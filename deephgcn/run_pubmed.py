import sys
import os
import copy
import json
import datetime

opt = dict()

opt['data'] = 'pubmed'
opt['layer'] = 8
opt['hidden'] = 64
opt['dropout'] = 0.4
opt['act'] = 'lrelu'
opt['wd-fc'] = 5e-4
opt['wd-conv'] = 1e-1
opt['lr'] = 1e-2
opt['dev'] = 0
opt['final_agg'] = ""
opt['test'] = ""
opt['print-epochs'] = 1
opt['optim'] = 'Adam'
opt['c'] = 1e-1

def generate_command(opt):
    cmd = 'python train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

evaluation_times = 1
for k in range(evaluation_times):
    seed = k + 1
    opt['seed'] = seed
    run(opt)