from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from model import *
import uuid
import geoopt
from geooptplus import PoincareBall
import pandas as pd

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd-fc', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd-conv', type=float, default=1e-2, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--data', default='cora', help='dataset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--gamma', type=int, default=0., help='gamma of lr scheduler')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--final_agg', action='store_true', default=False, help='final weight alignment.')
parser.add_argument('--margin', type=float, default=2., help='margin of margin loss.')
parser.add_argument('--grad-clip', type=float, default=None, help='clip gradient.')
parser.add_argument('--lr-reduce-freq', type=float, default=None, help='reduce lr frequency')
parser.add_argument('--print-epochs', type=int, default=1, help='print every epoch evaluation.')
parser.add_argument('--c', type=float, default=1., help='curvature')
parser.add_argument('--task', type=str, default='nc', help='nc/lp')
parser.add_argument('--optim', type=str, default='Adam', help='Adam/SGD')
parser.add_argument('--act', type=str, default='relu', help='none/relu/lrelu')
parser.add_argument('--alpha', type=float, default=0.1, help='')
parser.add_argument('--beta', type=float, default=0.6, help='')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
task = args.task
if task == 'nc':
    if args.data in ['cora', 'citeseer', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_data(args, './data/{}'.format(args.data), normalize_feats=True)
    elif args.data in ['disease_nc', 'airport']:
        adj, features, labels, idx_train, idx_val, idx_test = load_data(args, './data/{}'.format(args.data), normalize_feats=True)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(args.data))
else:
    print('LP Task')
    

cudaid = "cuda:"+ str(args.dev)
device = torch.device(cudaid)
args.n_nodes, args.feat_dim = features.shape
features = features.to(device)
adj = adj.to(device)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid, checkpt_file)

# feature normalization
manifold = PoincareBall(c=args.c)
features = manifold.expmap0(features)

# Activation function
act_fn = None
if args.act == 'none':
    act_fn = None
elif args.act == 'relu':
    act_fn = nn.ReLU()
elif args.act == 'lrelu':
    act_fn = nn.LeakyReLU()

# Define model
model = HGCN(nfeat = args.feat_dim,
                nlayers = args.layer,
                nhidden = args.hidden,
                nclass = int(labels.max()) + 1,
                dropout = args.dropout,
                final_agg = args.final_agg,
                act_fn = act_fn,
                c = args.c,
                params = [args.alpha, args.beta]).to(device)
parameters = [{'params':model.params_convs,'weight_decay':args.wd_conv},{'params':model.params_fcs,'weight_decay':args.wd_fc},]
if args.optim =='Adam': 
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
elif args.optim == 'SGD':
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.99, nesterov=True)
if not args.lr_reduce_freq:
    args.lr_reduce_freq = args.epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_reduce_freq),
                                                   gamma=float(args.gamma))

# Regularization settings
coef_list = []

def train():
    model.train()
    optimizer.zero_grad()
    output, loss_reg = model.encode(features, adj)

    # compute loss
    loss_train = None
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train += sum(map(lambda x, y: x * y, coef_list, loss_reg))
    
    acc_train, f1_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    if args.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return loss_train.item(), acc_train, f1_train


def validate():
    model.eval()
    with torch.no_grad():
        output, loss_reg = model.encode(features, adj)

        # compute loss
        loss_val = None
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        loss_val += sum(map(lambda x, y: x * y, coef_list, loss_reg))
        
        acc_val, f1_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val, f1_val

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output, loss_reg = model.encode(features, adj)

        # compute loss
        loss_test = None
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        loss_test += sum(map(lambda x, y: x * y, coef_list, loss_reg))

        acc_test, f1_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test, f1_test
    
t_total = time.time()
bad_counter = 0
best = 0
best_epoch = 0
acc = 0
save_train_acc = 0
save_val_acc = 0
save_val_f1 = 0
print("Training with seed {}...".format(args.seed))
for epoch in range(args.epochs):
    loss_tra, acc_tra, f1_tra = train()
    loss_val, acc_val, f1_val = validate()
    best_metric = acc_val
    if args.print_epochs:
        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch), 'train','loss:{:.3f}'.format(loss_tra),'acc:{:.2f}'.format(acc_tra * 100),
                '| val','loss:{:.3f}'.format(loss_val),'acc:{:.2f}'.format(acc_val * 100),'f1:{:.2f}'.format(f1_val * 100),
                '| best val acc:{:.2f}'.format(best * 100))
    if best_metric >= best:
        if_update = True
        if best_metric == best:
            if acc_tra >= save_train_acc:
                if_update = True
            else:
                if_update = False
        else:
            if_update = True
        if if_update:
            best = best_metric
            best_epoch = epoch
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            save_train_acc = acc_tra
            save_val_acc = acc_val
            save_val_f1 = f1_val
            bad_counter = 0
        else:
            bad_counter += 1
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

if args.test:
    acc = test()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch, train acc:{:.2f}, validation acc:{:.2f}'.format(best_epoch, save_train_acc * 100., save_val_acc * 100.))
print("Test" if args.test else "Val", "acc.:{:.1f}".format(acc * 100.))

# df_path = './experiment/record.csv'
# df = pd.read_csv(df_path)
# df = pd.concat([df, pd.DataFrame({'data': [args.data], 'acc': [acc * 100.], 'alpha': [args.alpha], 'layers': [args.layer]})], ignore_index=True)
# df.to_csv(df_path, index=False)