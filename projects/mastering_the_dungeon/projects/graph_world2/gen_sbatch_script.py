#!/usr/bin/env python3

##
## Copyright (c) Facebook, Inc. and its affiliates.
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##

import argparse
import subprocess

def gen(out_name, opt):
    fout_batch = open('{}.sh'.format(out_name), 'w')
    fout_batch.write('chmod +x *.sh\n')
    fout_batch.write('rm job-out-*\n')
    fout_batch.write('rm job-in-*\n')
    for i in range(opt['num_gpus']):
        sh_name = '{}_{}.sh'.format(out_name, i)
        fout = open(sh_name, 'w')
        if opt['slurm']:
            fout.write("srun -o checkpoint/slurm-gpu-job-%j.out --error=checkpoint/slurm-gpu-job-%j.err --gres=gpu:1 python3 train.py --job_num {}\n".format(i))
        else:
            fout.write("CUDA_VISIBLE_DEVICES={} python3 train.py --job_num {}\n".format(i, i))
        fout.close()
        fout_batch.write("./{} &\n".format(sh_name))
    fout_batch.close()
    subprocess.call("chmod +x {}.sh".format(out_name).split())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--slurm', action='store_true', default=False, help='whether use slurm or not')
    opt = vars(parser.parse_args())
    gen('batch_holder', opt)
