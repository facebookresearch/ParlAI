#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generates a dictionary file from the training data.

Examples
--------

.. code-block:: shell

  # learn the vocabulary from one task, then train on another task.
  # TODO: missing task specification
  python parlai_hydra/scripts/build_dict.py  dict.file=premade.dict

  # Old command:
  # python -m parlai.scripts.build_dict -t convai2 --dict-file premade.dict
  # python -m parlai.scripts.train_model -t squad --dict-file premade.dict -m seq2seq

"""

import logging
import os

import hydra
import tqdm

from parlai_hydra.core.worlds import create_task
from parlai.utils.distributed import is_distributed
from parlai.utils.misc import TimeLogger

log = logging.getLogger(__name__)


def build_dict(cfg):
    if cfg.build_dict.skip_if_build and os.path.isfile(cfg.dict.file):
        # Dictionary already built, skip all loading or setup
        log.info("dictionary already built")
        return None

    if is_distributed():
        raise ValueError('Dictionaries should be pre-built before distributed train.')

    dictionary = hydra.utils.instantiate(cfg.dict)

    if os.path.isfile(cfg.dict.file):
        # Dictionary already built, return loaded dictionary agent
        log.info("dictionary already built")
        return dictionary

    # TODO: override
    # ordered_opt['numthreads'] = 1
    # ordered_opt['batchsize'] = 1

    # Set this to none so that image features are not calculated when Teacher is
    # instantiated while building the dict
    # TODO: change 'none' to 'no_image_model'
    # ordered_opt['image_mode'] = 'none'

    # ordered_opt['pytorch_teacher_batch_sort'] = False
    # TODO: how to check if task if ??? (not set)
    # if cfg.teacher.task['task'] == 'pytorch_teacher' or cfg.teacher.task is None:
    #     pytorch_teacher_task = ordered_opt.get('pytorch_teacher_task', '')
    #     if pytorch_teacher_task != '':
    #         ordered_opt['task'] = pytorch_teacher_task

    datatypes = ['train:ordered:stream']
    if cfg.build_dict.include_valid:
        datatypes.append('valid:stream')
    if cfg.build_dict.include_test:
        datatypes.append('test:stream')

    cnt = 0
    for dt in datatypes:
        world_dict = create_task(cfg, dictionary)
        # pass examples to dictionary
        log.info('running dictionary over data..')
        log_time = TimeLogger()
        total = world_dict.num_examples()
        if opt['dict_maxexs'] >= 0:
            total = min(total, opt['dict_maxexs'])

        log_every_n_secs = opt.get('log_every_n_secs', None)
        if log_every_n_secs:
            pbar = tqdm.tqdm(
                total=total, desc='Building dictionary', unit='ex', unit_scale=True
            )
        else:
            pbar = None
        while not world_dict.epoch_done():
            cnt += 1
            if cnt > cfg.build_dict.maxexs and cfg.build_dict.maxexs >= 0:
                log.info('Processed {} exs, moving on.'.format(cfg.build_dict.maxexs))
                # don't wait too long...
                break
            world_dict.parley()
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()

    dictionary.save(cfg.dict.file, sort=True)
    log.info(
        'dictionary built with {} tokens in {}s'.format(
            len(dictionary), round(log_time.total_time(), 2)
        )
    )
    return dictionary


@hydra.main(config_path='conf/build_dict.yaml')
def build_dict_main(cfg):
    build_dict(cfg)


if __name__ == '__main__':
    build_dict_main()
