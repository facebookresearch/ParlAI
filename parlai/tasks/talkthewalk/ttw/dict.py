# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
import argparse

from nltk.tokenize import TweetTokenizer

UNK_TOKEN = '__UNK__'
START_TOKEN = '__START__'
END_TOKEN = '__END__'
PAD_TOKEN = '__PAD__'
SPECIALS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]


def split_tokenize(text):
    """Splits tokens based on whitespace after adding whitespace around
    punctuation.
    """
    return (text.lower().replace('.', ' . ').replace('. . .', '...')
            .replace(',', ' , ').replace(';', ' ; ').replace(':', ' : ')
            .replace('!', ' ! ').replace('?', ' ? ')
            .split())


class Dictionary:
    def __init__(self, file=None, min_freq=0, split=False):
        self.i2tok = list()
        self.tok2i = dict()
        self.tok2cnt = dict()
        self.split = split

        self.null_token = UNK_TOKEN
        self.start_token = START_TOKEN
        self.end_token = END_TOKEN

        self.txt2vec = self.encode


        for tok in SPECIALS:
            self.tok2i[tok] = len(self.tok2i)
            self.i2tok.append(tok)
            self.tok2cnt[tok] = 100000000

        if file is not None:
            with open(file) as f:
                for line in f:
                    try:
                        tok, cnt = line.split('\t')
                    except:
                        tok, cnt = line.split(' ')
                    if int(cnt) >= min_freq:
                        self.tok2i[tok] = len(self.i2tok)
                        self.tok2cnt[tok] = int(cnt)
                        self.i2tok.append(tok)

        self.tokenizer = TweetTokenizer()

    def __len__(self):
        return len(self.i2tok)

    def __getitem__(self, tok):
        return self.tok2i.get(tok, self.tok2i[UNK_TOKEN])

    def encode(self, msg, include_end=False):
        if self.split:
            ret = [self[tok] for tok in split_tokenize(msg)]
        else:
            ret = [self[tok] for tok in self.tokenizer.tokenize(msg)]
        return ret + [self[END_TOKEN]] if include_end else ret

    def decode(self, toks):
        res = []
        for tok in toks:
            tok = self.i2tok[tok]
            if tok != END_TOKEN:
                res.append(tok)
            else:
                break
        return ' '.join(res)

    def add(self, msg):
        for tok in split_tokenize(msg):
            if tok not in self.tok2i:
                self.tok2cnt[tok] = 0
                self.tok2i[tok] = len(self.i2tok)
                self.i2tok.append(tok)
            self.tok2cnt[tok] += 1

    def save(self, file):
        toklist = [(tok, cnt) for tok, cnt in self.tok2cnt.items()]
        sorted_list = sorted(toklist, key=lambda x: x[1], reverse=True)

        with open(file, 'w') as f:
            for tok in sorted_list:
                if tok[0] not in SPECIALS:
                    f.write(tok[0] + '\t' + str(tok[1]) + '\n')


class LandmarkDictionary(object):
    def __init__(self):
        self.i2landmark = ['Coffee Shop', 'Shop', 'Restaurant', 'Bank', 'Subway',
                           'Playfield', 'Theater', 'Bar', 'Hotel', 'Empty']
        self.landmark2i = {value: index for index, value in enumerate(self.i2landmark)}

    def encode(self, name):
        return self.landmark2i[name] + 1

    def decode(self, i):
        return self.i2landmark[i-1]

    def __len__(self):
        return len(self.i2landmark) + 1

class ActionAwareDictionary:

    def __init__(self):
        self.aware_id2act = ['ACTION:TURNLEFT', 'ACTION:TURNRIGHT', 'ACTION:FORWARD']
        self.aware_act2id = {v: k for k, v in enumerate(self.aware_id2act)}

    def encode(self, msg):
        if msg in self.aware_act2id:
            return self.aware_act2id[msg]+1
        return -1

    def decode(self, id):
        return self.aware_id2act[id-1]

    def __len__(self):
        return len(self.aware_id2act) + 1


class ActionAgnosticDictionary:
    def __init__(self):
        self.agnostic_id2act = ['LEFT', 'UP', 'RIGHT', 'DOWN', 'STAYED']
        self.agnostic_act2id = {v: k for k, v in enumerate(self.agnostic_id2act)}
        self.act_to_orientation = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def encode(self, msg):
        return self.agnostic_act2id[msg] + 1

    def decode(self, id):
        return self.agnostic_id2act[id-1]

    def encode_from_location(self, old_loc, new_loc):
        """Determine if tourist went up, down, left, or right"""
        step_to_dir = {
            0: {
                1: 'UP',
                -1: 'DOWN',
                0: 'STAYED'
            },
            1: {
                0: 'LEFT',
            },
            -1: {
                0: 'RIGHT'
            }
        }

        step = [new_loc[0] - old_loc[0], new_loc[1] - old_loc[1]]
        act = self.agnostic_act2id[step_to_dir[step[0]][step[1]]]
        return act + 1

    def __len__(self):
        return len(self.agnostic_id2act) + 1


class TextrecogDict:

    def __init__(self, textfeatures):
        obs_vocab = set([])
        for neighborhood in textfeatures.keys():
            # build textrecognition vocab
            for k, obs in textfeatures[neighborhood].items():
                obs_vocab |= set([o['lex_recog'] for o in obs])

        self.obs_i2s = list(obs_vocab)
        self.obs_s2i = {k: i for i, k in enumerate(self.obs_i2s)}

    def encode(self, text):
        return self.obs_s2i[text] + 1

    def decode(self, index):
        return self.obs_i2s[index-1]

    def __len__(self):
        return len(self.obs_i2s) + 1


if __name__ == '__main__':
    """Build language dictionary and save to args.data_dir/dict.txt"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')

    args = parser.parse_args()

    train_set = json.load(open(os.path.join(args.data_dir, 'talkthewalk.train.json')))
    valid_set = json.load(open(os.path.join(args.data_dir, 'talkthewalk.valid.json')))
    test_set = json.load(open(os.path.join(args.data_dir, 'talkthewalk.test.json')))

    dictionary = Dictionary()
    for set in [train_set, valid_set, test_set]:
        for config in set:
            for msg in config['dialog']:
                if msg['id'] == 'Tourist':
                    if msg['text'] not in ['ACTION:TURNLEFT', 'ACTION:TURNRIGHT', 'ACTION:FORWARD']:
                        if len(msg['text'].split(' ')) > 2:
                            dictionary.add(msg['text'])

    dictionary.save(os.path.join(args.data_dir, 'dict.txt'))
