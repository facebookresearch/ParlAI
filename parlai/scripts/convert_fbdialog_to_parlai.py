#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Convert FB Dialog format into the ParlAI text format.

Examples
--------

.. code-block:: shell

  python convert_fbdialog_to_parlai.py --infile infile --outfile /tmp/dump
"""

from parlai.core.params import ParlaiParser
from parlai.core.utils import msg_to_str
import random


def setup_data(path, cloze=False, double=False):
    r"""
    Read data in the fbdialog format.

    Returns ``((x,y,r,c), new_episode?)`` tuples.

    ``x`` represents a query, ``y`` represents the labels, ``r`` represents
    any reward, and ``c`` represents any label_candidates.

    The example above will be translated into the following tuples:

    ::

        x: 'Sam went to the kitchen\nPat gave Sam the milk\nWhere is the milk?'
        y: ['kitchen']
        r: '1'
        c: ['hallway', 'kitchen', 'bathroom']
        new_episode = True (this is the first example in the episode)


    ::

        x: 'Sam went to the hallway\\nPat went to the bathroom\\nWhere is the
            milk?'
        y: ['hallway']
        r: '1'
        c: ['hallway', 'kitchen', 'bathroom']
        new_episode = False (this is the second example in the episode)

    :param cloze:
        if cloze, add special question to the end
    """
    print("[loading fbdialog data:" + path + "]")
    with open(path) as read:
        start = True
        x = ''
        reward = 0
        last_conv_id = None
        for line in read:
            line = line.strip().replace('\\n', '\n')
            if len(line) == 0:
                # empty response
                continue

            # first, get conversation index -- '1' means start of episode
            space_idx = line.find(' ')
            if space_idx == -1:
                # empty line, both individuals are saying whitespace
                conv_id = int(line)
            else:
                conv_id = int(line[:space_idx])

            # split line into constituent parts, if available:
            # x<tab>y<tab>reward<tab>label_candidates
            # where y, reward, and label_candidates are optional
            split = line[space_idx + 1:].split('\t')

            # remove empty items and strip each one
            for i in range(len(split)):
                word = split[i].strip()
                if len(word) == 0:
                    split[i] = ''
                else:
                    split[i] = word
            # Empty reward string same as None
            if len(split) > 2 and split[2] == '':
                split[2] = None

            # now check if we're at a new episode
            if last_conv_id is None or conv_id <= last_conv_id:
                x = x.strip()
                if x:
                    yield [x, None, reward], start
                start = True
                reward = 0
                # start a new episode
                if cloze:
                    x = 'Fill in the blank in the last sentence.\n{x}'.format(
                        x=split[0]
                    )
                else:
                    x = split[0]
            else:
                if x:
                    # otherwise add current x to what we have so far
                    x = '{x}\n{next_x}'.format(x=x, next_x=split[0])
                else:
                    x = split[0]
            last_conv_id = conv_id
            if len(split) > 2 and split[2]:
                reward += float(split[2])

            if len(split) > 1 and split[1]:
                # only generate an example if we have a y
                split[0] = x
                # split labels
                split[1] = split[1].split('|')
                if len(split) > 3:
                    # split label_candidates
                    split[3] = split[3].split('|')
                if len(split) > 2:
                    split[2] = reward
                else:
                    split.append(reward)
                if start:
                    yield split, True
                    start = False
                else:
                    yield split, False
                # reset x in case there is unlabeled data still left
                x = ''
                reward = 0
        if x:
            yield [x, None, reward], start


def _read_episode(data_loader):
    """
    Read one episode at a time from the provided iterable over entries.

    :param data_loader:
        an iterable which returns tuples in the format described in the
        class docstring.
    """
    episode = []
    last_cands = None
    for entry, new in data_loader:
        if new and len(episode) > 0:
            yield tuple(episode)
            episode = []
            last_cands = None

        # intern all strings so we don't store them more than once
        # TODO: clean up the if .. sys.intern else None by refactoring
        new_entry = []
        if len(entry) > 0:
            # process text if available
            if entry[0] is not None:
                new_entry.append(entry[0])
            else:
                new_entry.append(None)
            # TODO: unindent all of these one level.
            if len(entry) > 1:
                # process labels if available
                if entry[1] is None:
                    new_entry.append(None)
                elif hasattr(entry[1], '__iter__') and type(entry[1]) is not str:
                    # TODO: this could use the abc collections
                    # make sure iterable over labels, not single string
                    new_entry.append(tuple(e for e in entry[1]))
                else:
                    raise TypeError(
                        'Must provide iterable over labels, not a single string.'
                    )
            if len(entry) > 2:
                # process reward if available
                if entry[2] is not None:
                    new_entry.append(entry[2])
                else:
                    new_entry.append(None)
            if len(entry) > 3:
                # process label candidates if available
                if entry[3] is None:
                    new_entry.append(None)
                elif last_cands and entry[3] is last_cands:
                    # if cands are shared, say "same" so we
                    # don't store them again
                    # TODO: This is bad, and it's not actually used anywhere
                    # DEPRECATIONDAY: make this more rational
                    new_entry.append('same as last time')
                elif hasattr(entry[3], '__iter__') and type(entry[3]) is not str:
                    # make sure iterable over candidates, not single string
                    last_cands = entry[3]
                    new_entry.append(tuple(e for e in entry[3]))
                else:
                    raise TypeError(
                        'Must provide iterable over label candidates, '
                        'not a single string.'
                    )
            if len(entry) > 4 and entry[4] is not None:
                new_entry.append(entry[4])

        episode.append(tuple(new_entry))

    if len(episode) > 0:
        yield tuple(episode)

def build_table(entry):
    """
    Packs an entry into an action-observation dictionary.

    :param entry: a tuple in the form described in the class docstring.
    """
    table = {}
    if entry[0] is not None:
        table['text'] = entry[0]
    if len(entry) > 1:
        if entry[1] is not None:
            table['labels'] = entry[1]
    if len(entry) > 2:
        if entry[2] is not None:
            table['reward'] = entry[2]
    if len(entry) > 3:
        if entry[3] is not None:
            table['label_candidates'] = entry[3]
    if len(entry) > 4 and entry[4] is not None:
        table['image'] = entry[4]

    if 'labels' in table and 'label_candidates' in table:
        if table['labels'][0] not in table['label_candidates']:
            raise RuntimeError('true label missing from candidate labels')
    return table


def dump_data(opt):
    ignorefields = opt.get('ignore_fields', '')
    if opt.get('outfile') is None:
        raise RuntimeError('Please specify outfile when converting to ParlAI format')
    if opt.get('infile') is None:
        raise RuntimeError('Please specify infile when converting to ParlAI format')
    outfile = opt['outfile']
    print('[ starting to convert.. ]')
    print('[ saving output to {} ]'.format(outfile))
    with open(outfile, 'w') as fw:
        data_loader = setup_data(opt.get('infile'), cloze=opt.get('cloze'))
        if opt.get('additional_data_loader'):
            data_loader = opt.get('additional_data_loader')((data_loader))
        for ep in _read_episode(data_loader):
            for i, ex in enumerate(ep):
                ex = build_table(ex)
                ex['episode_done'] = i == len(ep) - 1
                ex['labels'] = ex.get('labels', ex.pop('eval_labels', None))
                txt = msg_to_str(ex, ignore_fields=ignorefields)
                fw.write(txt + '\n')
                if ex.get('episode_done', False):
                    fw.write('\n')


def main():
    random.seed(42)
    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument('-of', '--outfile', default=None, type=str,
                        help='Output file where to save, by default will be \
                                created in /tmp')
    parser.add_argument('-inf', '--infile', default=None, type=str,
                        help='Input file to load in fbdialog format')
    parser.add_argument('-if', '--ignore-fields', default='id', type=str,
                        help='Ignore these fields from the message (returned\
                                with .act() )')
    parser.add_argument('--cloze', default=False, type='bool',
                        help='whether dataset is cloze')
    opt = parser.parse_args()
    dump_data(opt)


if __name__ == '__main__':
    main()
