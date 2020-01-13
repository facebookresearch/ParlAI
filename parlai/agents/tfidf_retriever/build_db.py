#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A script to read in and store ParlAI tasks in a sqlite database.

Adapted from Adam Fisch's work at github.com/facebookresearch/DrQA/
"""

import sqlite3
import os

from tqdm import tqdm

from collections import deque
import random
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.utils.logging import logger

fmt = '%(asctime)s: [ %(message)s ]'
logger.set_format(fmt)

# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def store_contents(opt, task, save_path, context_length=-1, include_labels=True):
    """
    Preprocess and store a corpus of documents in sqlite.

    Args:
        task: ParlAI tasks of text (and possibly values) to store.
        save_path: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute('CREATE TABLE documents (id INTEGER PRIMARY KEY, text, value);')
    if not task:
        logger.info('No data to initialize table: just creating table.')
        logger.info('Add more data by passing observations to the agent.')
        logger.info('Committing...')
        conn.commit()
        conn.close()
        return

    ordered_opt = opt.copy()
    dt = opt.get('datatype', '').split(':')
    ordered_opt['datatype'] = ':'.join([dt[0], 'ordered'] + dt[1:])
    ordered_opt['batchsize'] = 1
    ordered_opt['numthreads'] = 1
    ordered_opt['task'] = task
    teacher = create_task_agent_from_taskname(ordered_opt)[0]

    episode_done = False
    current = []
    triples = []
    context_length = context_length if context_length >= 0 else None
    context = deque(maxlen=context_length)
    with tqdm(total=teacher.num_episodes()) as pbar:
        while not teacher.epoch_done():
            # collect examples in episode
            while not episode_done:
                action = teacher.act()
                current.append(action)
                episode_done = action['episode_done']

            for ex in current:
                if 'text' in ex:
                    text = ex['text']
                    context.append(text)
                    if len(context) > 1:
                        text = '\n'.join(context)

                # add labels to context
                labels = ex.get('labels', ex.get('eval_labels'))
                label = None
                if labels is not None:
                    label = random.choice(labels)
                    if include_labels:
                        context.append(label)
                # use None for ID to auto-assign doc ids--we don't need to
                # ever reverse-lookup them
                triples.append((None, text, label))

            c.executemany('INSERT OR IGNORE INTO documents VALUES (?,?,?)', triples)
            pbar.update()

            # reset flags and content
            episode_done = False
            triples.clear()
            current.clear()
            context.clear()

    logger.info(
        'Read %d examples from %d episodes.'
        % (teacher.num_examples(), teacher.num_episodes())
    )
    logger.info('Committing...')
    conn.commit()
    conn.close()
