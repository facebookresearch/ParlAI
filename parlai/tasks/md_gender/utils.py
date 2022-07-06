#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from collections import defaultdict
from parlai.tasks.md_gender.build import build

"""
Gender utilities for the multiclass gender classification tasks.
"""

MASK_TOKEN = '[MASK]'

MASC = 'male'
FEM = 'female'
NEUTRAL = 'gender-neutral'
NONBINARY = 'non-binary'
UNKNOWN = 'unknown'


SELF_UNKNOWN_LABELS = [f'SELF:{MASC}', f'SELF:{FEM}']
PARTNER_UNKNOWN_LABELS = [f'PARTNER:{MASC}', f'PARTNER:{FEM}']
UNKNOWN_LABELS = {'self': SELF_UNKNOWN_LABELS, 'partner': PARTNER_UNKNOWN_LABELS}

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

SELF_CANDS = [f'SELF:{MASC}', f'SELF:{FEM}']
PARTNER_CANDS = [f'PARTNER:{NEUTRAL}', f'PARTNER:{MASC}', f'PARTNER:{FEM}']
ABOUT_CANDS = [
    f'ABOUT:{NEUTRAL}',
    f'ABOUT:{FEM}',
    f'ABOUT:{MASC}',
    f'ABOUT:{NONBINARY}',
]
ALL_CANDS = {'self': SELF_CANDS, 'partner': PARTNER_CANDS, 'about': ABOUT_CANDS}
EMPTY_LABELS = {'self': 'SELF:{}', 'partner': 'PARTNER:{}', 'about': 'ABOUT:{}'}


def get_data_stats(data, key='label', lst=True):
    counts = defaultdict(int)
    for ex in data:
        if lst:
            label = ex[key][0]
        else:
            label = ex[key]
        counts[label] += 1

    print('Total dataset counts:')
    tups = sorted([(k, v) for k, v in counts.items()], key=lambda x: x[0])
    for k, v in tups:
        print(f'{k}: {v}')


def add_common_args(parser):
    """
    Add arguments common across all of the datasets.
    """
    agent = parser.add_argument_group('Gender Multiclass args')
    agent.add_argument(
        '--balance',
        type='bool',
        default=False,
        help='Whether to balance the data between classes during training',
    )
    agent.add_argument(
        '--balance-valid',
        type='bool',
        default=False,
        help='Whether to balance the validation data',
    )
    agent.add_argument(
        '--add-unknown-classes',
        type='bool',
        default=False,
        help='Add unknown classes as neutral',
    )
    agent.add_argument(
        '--unknown-temp',
        type=float,
        default=1.0,
        help='Rate at which to sample examples from the unknown class',
    )
    return parser


def balance_data(data_list, key='labels', shuffle=True, exclude_labels=None):
    """
    Given a list of acts, balance the list by label.
    """
    if len(data_list) == 0:
        # empty set
        return data_list

    def get_lst_sample(lst, sample_size):
        if len(lst) == sample_size:
            return lst

        sampled = []
        sample_times = sample_size // len(lst)
        for _ in range(sample_times):
            sampled += lst
        differential = sample_size - len(sampled)
        if differential > 0:
            extra_examples = random.sample(lst, differential)
            sampled += extra_examples

        return sampled

    separate_data = {}
    excluded_data = []
    for x in data_list:
        label = x[key]
        if isinstance(label, list):
            label = label[0]
        if exclude_labels is not None and label in exclude_labels:
            # exclude this from the balancing, but
            # add it later
            excluded_data.append(x)
        else:
            separate_data.setdefault(label, [])
            separate_data[label].append(x)

    max_len = max(len(value) for value in separate_data.values())
    new_data = []
    for _, data in separate_data.items():
        exs = get_lst_sample(data, max_len)
        new_data += exs

    assert len(new_data) == max_len * len(separate_data)

    # now add back data that was excluded from balancing
    new_data += excluded_data

    if shuffle:
        random.shuffle(new_data)

    return new_data


def get_inferred_about_data(task, opt, threshold=0.8):
    """
    Load inferred ABOUT data from teh ABOUT classifier.
    """
    root = os.path.join(
        opt['datapath'], 'md_gender', 'data_to_release', 'inferred_about'
    )
    task_str = task.split(':')[-1]
    dt = opt['datatype'].split(':')[0]
    with open(os.path.join(root, f'{task_str}_{dt}_binary.txt'), 'r') as f:
        lines = f.read().splitlines()
    examples = []
    for line in lines:
        text, label, score = line.split('\t')
        if threshold is not None and float(score) < threshold:
            # replace label with NEUTRAL
            label = f'ABOUT:{NEUTRAL}'
        if not text or not label:
            continue
        examples.append(
            {
                'text': text,
                'labels': [label],
                'class_type': 'about',
                'label_candidates': ABOUT_CANDS,
                'episode_done': True,
            }
        )

    return examples


def format_text(text, lower=True):
    """
    Add spaces around punctuation.
    """
    if lower:
        text = text.lower()
    for punc in PUNCTUATION_LST:
        text = text.replace(punc[1], punc[0])

    return text


def unformat_text(text):
    """
    Remove spaces from punctuation.
    """
    for punc in PUNCTUATION_LST:
        text = text.replace(punc[0], punc[1])

    return text


def get_explicitly_gendered_words(opt):
    """
    Load list of explicitly gendered words from.

    <https://github.com/uclanlp/gn_glove/blob/main/wordlist/>.

    Examples include brother, girl, actress, husbands, etc.
    """
    build(opt)
    folder = os.path.join(opt['datapath'], 'md_gender', 'data_to_release', 'word_list')
    male_words = os.path.join(folder, 'male_word_file.txt')
    female_words = os.path.join(folder, 'female_word_file.txt')

    with open(male_words, 'r') as f:
        male = f.read().splitlines()

    with open(female_words, 'r') as f:
        female = f.read().splitlines()

    return male, female


def mask_gendered_words(text, gendered_list, mask_token=MASK_TOKEN):
    """
    Given a string of text, mask out gendered words from a list.
    """
    text = format_text(text, lower=False)
    to_ret = []
    orig_text = text.split(' ')
    lowered_text = text.lower().split(' ')
    for word, word_lower in zip(orig_text, lowered_text):
        if word_lower in gendered_list:
            to_ret.append(mask_token)
        else:
            to_ret.append(word)

    return unformat_text(' '.join(to_ret))


CONTRACTIONS_LIST = [
    ("i am", "i'm"),
    ("you are", "you're"),
    ("we are", "we're"),
    ("they are", "they're"),
    ("who are", "who're"),
    ("i have", "i've"),
    ("you have", "you've"),
    ("we have", "we've"),
    ("could have", "could've"),
    ("would have", "would've"),
    ("should have", "should've"),
    ("might have", "might've"),
    ("who have", "who've"),
    ("there have", "there've"),
    ("he is", "he's"),
    ("she is", "she's"),
    ("it is", "it's"),
    ("what is", "what's"),
    ("that is", "that's"),
    ("who is", "who's"),
    ("there is", "there's"),
    ("here is", "here's"),
    ("one is", "one's"),
    ("i will", "i'll"),
    ("you will", "you'll"),
    ("she will", "she'll"),
    ("he will", "he'll"),
    ("it will", "it'll"),
    ("we will", "we'll"),
    ("they will", "they'll"),
    ("that will", "that'll"),
    ("there will", "there'll"),
    ("this will", "this'll"),
    ("what will", "what'll"),
    ("who will", "who'll"),
    ("i would", "i'd"),
    ("you would", "you'd"),
    ("he would", "he'd"),
    ("she would", "she'd"),
    ("we would", "we'd"),
    ("they would", "they'd"),
    ("it would", "it'd"),
    ("there would", "there'd"),
    ("what would", "what'd"),
    ("who would", "who'd"),
    ("that would", "that'd"),
    ("let us", "let's"),
    ("cannot", "can't"),
    ("do not", "don't"),
    ("is not", "isn't"),
    ("will not", "won't"),
    ("should not", "shouldn't"),
    ("could not", "couldn't"),
    ("would not", "wouldn't"),
    ("are not", "aren't"),
    ("does not", "doesn't"),
    ("was not", "wasn't"),
    ("were not", "weren't"),
    ("has not", "hasn't"),
    ("have not", "haven't"),
    ("had not", "hadn't"),
    ("must not", "mustn't"),
    ("did not", "didn't"),
    ("might not", "mightn't"),
    ("need not", "needn't"),
]


CONTRACTION_SPACES = [(" " + x[0] + " ", " " + x[1] + " ") for x in CONTRACTIONS_LIST]

CONTRACTION_LEFT_SPACES = [(" " + x[0], " " + x[1]) for x in CONTRACTIONS_LIST]

CONTRACTION_RIGHT_SPACES = [(x[0] + " ", x[1] + " ") for x in CONTRACTIONS_LIST]
