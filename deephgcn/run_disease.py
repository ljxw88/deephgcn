import sys
import os
import copy
import json
import datetime

opt = dict()

opt['data'] = 'disease_nc'
opt['layer'] = 6
opt['hidden'] = 256
opt['dropout'] = 0.1
opt['act'] = 'lrelu'
opt['wd-fc'] = 0.
opt['wd-conv'] = 0.
opt['lr'] = 1e-2
opt['dev'] = 0
opt['test'] = ""
opt['print-epochs'] = 1
opt['optim'] = 'Adam'
opt['c'] = 0.01

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