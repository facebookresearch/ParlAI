# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Runs Google's Seq2Seq training code at
https://github.com/google/seq2seq/blob/master/bin/train.py
"""

import os
import subprocess
import threading
import time

from parlai.core.params import ParlaiParser
from parlai.core.metrics import Metrics

from build import setup_args, build_parallel_data


class EvalThread(object):
    """Runs in the background, looking for output predictions and reporting
    ParlAI metrics for those outputs.
    """

    def __init__(self, output_dir):
        smpl_dir = os.path.join(output_dir, 'samples')
        self.thread = threading.Thread(target=self.run, args=(smpl_dir,),
                                       daemon=True)
        self.thread.start()

    def run(self, directory):
        seen = set()
        self.metrics = Metrics({})
        while True:
            for (dirpath, dirnames, filenames) in os.walk(directory):
                for f in filenames:
                    if f not in seen:
                        seen.add(f)
                        self.run_eval(os.path.join(dirpath, f))
            time.sleep(5)

    def run_eval(self, fn):
        self.metrics.clear()
        with open(fn) as read:
            read.readline() # skip first line
            read.readline() # skip second line
            pred = read.readline()
            target = read.readline()
            while pred and target:
                pred = pred.replace('SEQUENCE_END', '').strip()
                target = target.replace('SEQUENCE_END', '').strip()
                self.metrics.update({'text': pred}, [target])
                pred = read.readline()
                target = read.readline()
        print('***************************************************************')
        print('Running ParlAI eval on training sample predictions at {}'.format(fn))
        print(self.metrics.report())
        print('===============================================================')
        pass


if __name__ ==  '__main__':
    parser = setup_args()
    parser.add_argument('--tf_train', required=True,
        help='Path to TF train script, e.g. ~/seq2seq/bin/train.py')
    parser.add_argument('--config_paths', required=True,
        help='Path to TF configs (see Google documentation).')
    parser.add_argument('--debug', default=False,
        help='If set, will run tf command with `-m pdb -c c`.')
    output_dir = build_parallel_data(parser)
    opt = parser.parse_args()

    def fn(dt, src):
        """Returns output_dir/{task_name}_{dt}_{texts|labels}.txt"""
        return os.path.join(output_dir,
            '_'.join([opt['task'], dt, 'texts.txt' if src else 'labels.txt']))

    VOCAB_SOURCE = os.path.join(output_dir, opt['task'] + '_dict.tsv')
    VOCAB_TARGET = os.path.join(output_dir, opt['task'] + '_dict.tsv')
    TRAIN_SOURCES = fn('train', True)
    TRAIN_TARGETS = fn('train', False)
    DEV_SOURCES = fn('valid', True)
    DEV_TARGETS = fn('valid', False)
    TRAIN_STEPS = '1000000'
    import sys
    try:
        import seq2seq
    except:
        print('Please install Tensorflow Seq2Seq from Google at '
              'https://github.com/google/seq2seq/')
    cmd = ['python']
    if opt['debug']:
        cmd += ['-m', 'pdb', '-c', 'c']
    cmd += [opt['tf_train'],
        '--config_paths={}'.format(opt['config_paths']),
        '--model_params','''
            vocab_source: {s}
            vocab_target: {t}'''.format(s=VOCAB_SOURCE, t=VOCAB_TARGET),
        '--input_pipeline_train', '''
            class: ParallelTextInputPipeline
            params:
                source_files:
                    - {s}
                target_files:
                    - {t}'''.format(s=TRAIN_SOURCES, t=TRAIN_TARGETS),
        '--input_pipeline_dev', '''
            class: ParallelTextInputPipeline
            params:
                source_files:
                    - {s}
                target_files:
                    - {t}'''.format(s=DEV_SOURCES, t=DEV_TARGETS),
        '--batch_size', str(opt.get('batchsize')),
        '--train_steps', TRAIN_STEPS,
        '--output_dir', output_dir]

    eval_thread = EvalThread(output_dir)
    print('Launching tensorflow with command:\n`{}`'.format(' '.join(cmd)))
    results = subprocess.run(cmd,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
