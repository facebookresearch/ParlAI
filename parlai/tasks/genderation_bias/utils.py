#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utils for Controllable Gen Teacher.
"""
from collections import deque
import random

from parlai.core.loader import load_teacher_module
from parlai.core.message import Message
from parlai.core.opt import Opt

from typing import List, Tuple

PUNCTUATION_LST = [
    (' .', '.'),
    (' !', '!'),
    (' ?', '?'),
    (' ,', ','),
    (" ' ", "'"),
    (" . . . ", "... "),
    (" ( ", " ("),
    (" ) ", ") "),
    (" ; ", "; "),
]


def format_text(text: str, lower: bool = True) -> str:
    """
    Space punctuation and lowercase text

    :param text:
        text to lowercase
    :param lower:
        whether to lowercase or not

    :return text:
        return formatted text.
    """
    if lower:
        text = text.lower()
    for punc in PUNCTUATION_LST:
        text = text.replace(punc[1], punc[0])

    return text


def get_word_list_token(
    text: str, word_lists: Tuple[List[str], List[str]], four_class: bool = False
) -> str:
    """
    Return a control token corresponding to gender within text.

    :param text:
        text to consider for control token
    :param word_lists:
        tuple of lists for male-specific and female-specific words
    :param four_class:
        whether to utilize four-class method (f0m0 --> f1m1) identifying.
        False defaults to three-class method (MALE/FEMALE/NEUTRAL)

    :return token:
        return control token corresponding to input text.
    """
    m_list, f_list = word_lists
    text = format_text(text)
    m_cnt = 0
    f_cnt = 0
    for word in text.split(' '):
        if word in m_list:
            m_cnt += 1
        if word in f_list:
            f_cnt += 1

    if four_class:
        if f_cnt == 0 and m_cnt == 0:
            return 'f0m0'
        elif f_cnt == 0 and m_cnt > 0:
            return 'f0m1'
        elif f_cnt > 0 and m_cnt == 0:
            return 'f1m0'
        else:
            return 'f1m1'
    # three class regime
    if m_cnt > f_cnt:
        return 'MALE'
    if f_cnt > m_cnt:
        return 'FEMALE'

    return 'NEUTRAL'


def flatten_and_classify(
    episode: List[Message],
    context_length: int,
    word_lists: Tuple[List[str], List[str]],
    include_labels: bool = True,
    delimiter: str = '\n',
    four_class: bool = False,
):
    """
    Flatten the dialogue history of an episode, explore into N new examples.

    Additionally, add control token corresponding to gender identified in the
    episode.

    :param episode:
        list of examples to flatten
    :param context_length:
        max number of utterances to use while flattening
    :param word_lists:
        tuple of lists for male-specific and female-specific words
    :param include_labels:
        whether to include labels while flattening
    :param delimiter:
        delimiter to use while flattening
    :param four_class:
        boolean param indicating whether to use four-class classification
        vs. three-class classification.
    """
    context = deque(maxlen=context_length if context_length > 0 else None)
    new_episode = []

    for ex in episode:
        context.append(ex.get('text', ''))
        # add context
        if len(context) > 1:
            ex.force_set('text', delimiter.join(context))
        # set episode_done to be True
        ex.force_set('episode_done', True)
        labels = ex.get('labels', ex.get('eval_labels', None))
        if labels is not None and include_labels:
            context.append(random.choice(labels))

        # word list
        control_tok = get_word_list_token(
            random.choice(labels), word_lists, four_class=four_class
        )
        ex.force_set('text', ex['text'] + ' ' + control_tok)
        new_episode.append(ex)

    return new_episode


def get_original_task_module(opt: Opt, multi_possible: bool = False):
    """
    Returns task module of "original" task.

    Original task in this case means the task we want to use
    with the control teacher.

    :param opt:
        opt dict
    :param multi_possible:
        specify True if multiple tasks are possible.

    :return task_module:
        return module associated with task.
    """
    modules = []
    tasks = opt['task'].split(',')
    if not multi_possible:
        assert len(tasks) == 1

    for task in tasks:
        if len(task.split(':')) < 3:
            raise RuntimeError(
                '\n\n********************************************************\n'
                'Must specify original task using the following format:\n'
                '`--task internal:flattened:task:<ORIGINAL TASK NAME>`'
                '\n********************************************************\n'
            )
        original_task = ':'.join(task.split(':')[2:])
        task_module = load_teacher_module(original_task)
        modules.append(task_module)

    if multi_possible:
        return modules

    return modules[0]
