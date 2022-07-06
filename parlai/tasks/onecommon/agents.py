#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
from .build import build

import os

EOS_TOKEN = '<eos>'
SELECTION_TOKEN = '<selection>'
YOU_TOKEN = 'YOU:'
THEM_TOKEN = 'THEM:'
SILENCE_TOKEN = '__SILENCE__'

INPUT_TAG = 'input'
DIALOGUE_TAG = 'dialogue'
REFERENTS_TAG = 'referents'
OUTPUT_TAG = 'output'
REAL_IDS_TAG = 'real_ids'

REF_BEGIN_IDX = 0
REF_END_IDX = 1
REF_EOS_IDX = 2
REF_BEGIN_TARGET_IDX = 3

N_OBJECT = 7


def get_tag(tokens, tag):
    """
    Extracts the value inside the given tag.
    """
    start = tokens.index('<' + tag + '>') + 1
    stop = tokens.index('</' + tag + '>')
    return tokens[start:stop]


class OneCommonTeacher(FixedDialogTeacher):
    """
    OneCommon teacher that loads the data from https://github.com/Alab-NII/Reference-
    Resolution.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.datatype = opt['datatype'].split(':')[0]
        build(opt)

        data_path = os.path.join(
            opt['datapath'],
            'onecommon',
            'onecommon-1.0',
            'aaai2020',
            'experiments',
            'data',
            'onecommon',
            self.datatype + '_reference_0.txt',
        )

        if shared and 'episodes' in shared:
            self.episodes = shared['episodes']
        else:
            self._setup_data(data_path)

        self.expected = {}

        self.reset()

    def reset(self):
        self.expected = {}
        super().reset()

    def num_examples(self):
        num_exs = sum(len(episode['dialogue']) // 2 for episode in self.episodes)
        return num_exs

    def num_episodes(self):
        return len(self.episodes)

    def share(self):
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def observe(self, observation):
        if 'metrics' not in observation:
            observation['metrics'] = {}

        # Selection accuracy
        if (
            'output' in observation
            and observation['output'] is not None
            and 'output' in self.expected
            and self.expected['output'] is not None
        ):
            obs_out = int(observation['output'])
            exp_out = int(self.expected['output'])
            observation['metrics']['target_sel_accuracy'] = float(obs_out == exp_out)

        # Reference accuracy
        if (
            'referents' in observation
            and observation['referents'] is not None
            and len(observation['referents']) > 0
            and 'referents' in self.expected
            and self.expected['referents'] is not None
            and len(self.expected['referents']) > 0
        ):
            obs_refs = observation['referents']
            exp_refs = self.expected['referents']
            assert len(obs_refs) == len(exp_refs)

            exact_match = 0.0
            equal_cnt = total_cnt = 0.0
            for obs_ref, exp_ref in zip(obs_refs, exp_refs):
                for obs_tgt, exp_tgt in zip(obs_ref['target'], exp_ref['target']):
                    # Expected sorted referents
                    equal_cnt += float(int(obs_tgt) == int(exp_tgt))
                    total_cnt += 1.0
                if equal_cnt == total_cnt:
                    exact_match += 1.0

            observation['metrics']['referent_exact_match'] = exact_match / len(exp_refs)
            observation['metrics']['referent_accuracy'] = equal_cnt / total_cnt

        return super().observe(observation)

    def get(self, episode_idx, entry_idx):
        episode = self.episodes[episode_idx]
        dialogue = episode['dialogue']
        referents = episode['referents']

        entry_idx *= 2  # Every two utterance is an entry

        action = {}
        action['context'] = episode['context']

        # Fill in teacher's message (THEM)
        sentence = dialogue[entry_idx]
        if sentence is None:
            action['text'] = SILENCE_TOKEN
        else:
            action['text'] = ' '.join(sentence[1:])

        # Fill in learner's response (YOU)
        entry_idx += 1
        sentence = dialogue[entry_idx]
        if sentence is None:
            action['labels'] = [SELECTION_TOKEN]
            action['referents'] = []
        else:
            action['labels'] = [' '.join(sentence[1:])]
            action['referents'] = referents[entry_idx]
        self.expected['referents'] = action['referents']

        # Fill in output at end of dialogue
        if entry_idx < len(dialogue) - 1:
            action['output'] = None
            action['episode_done'] = False
        else:
            action['output'] = episode['output']
            action['episode_done'] = True
            self.expected['output'] = action['output']

        return action

    def _setup_data(self, data_path):
        print('loading: ' + data_path)
        with PathManager.open(data_path) as data_file:
            raw_data = data_file.readlines()

        self.episodes = []
        for data in raw_data:
            words = data.strip().split()
            context = list(map(float, get_tag(words, INPUT_TAG)))
            dialogue, spans = self._split_dialogue(get_tag(words, DIALOGUE_TAG))
            referents = self._split_referents(get_tag(words, REFERENTS_TAG), spans)
            output = int(get_tag(words, OUTPUT_TAG)[0])
            self.episodes.append(
                {
                    'context': context,
                    'dialogue': dialogue,
                    'referents': referents,
                    'output': output,
                }
            )

    def _split_dialogue(self, words, separator=EOS_TOKEN):
        sentences = []
        spans = []
        start = 0
        for stop in range(len(words)):
            if words[stop] == separator:
                sentences.append(words[start:stop])
                spans.append((start, stop))
                start = stop + 1
        if stop >= start:
            sentences.append(words[start:])
            spans.append((start, len(words) - 1))

        # Dataset contains consecutive turn
        # concatenate utterances for those cases
        dialogue = []
        utterance = sentences[0]
        for i in range(1, len(sentences)):
            if sentences[i - 1][0] == sentences[i][0]:
                utterance += sentences[i][1:]
            else:
                dialogue.append(utterance)
                utterance = sentences[i]
        dialogue.append(utterance)

        if dialogue[0][0] == YOU_TOKEN:
            # Dialogue starts with YOU
            dialogue.insert(0, None)
            spans.insert(0, None)
        if dialogue[-1][0] == THEM_TOKEN:
            # Dialogue starts with THEM
            dialogue.append(None)
            spans.append(None)

        return dialogue, spans

    def _split_referents(self, raw_referents, spans):
        """
        Split the referents.

        The first 3 values are begin idx, end idx, and eos idx The next N_OBJECT values
        are booleans of if the object is referred e.g. 3 4 10 0 1 0 0 0 0 0 means idx 3
        to 4 is a markable of an utterance with <eos> at idx 10, and it refers to the
        2nd dot
        """

        referent_len = 3 + N_OBJECT
        splitted_referents = []
        for i in range(len(raw_referents) // referent_len):
            val = raw_referents[i * referent_len : (i + 1) * referent_len]
            splitted_referents.append(list(map(int, val)))

        referents = []
        idx = 0
        for span in spans:
            if span is None:
                referents.append(None)
                continue

            # span is a (bos index, eos index) of an utterance
            refs = []
            while idx < len(splitted_referents):
                if splitted_referents[idx][REF_EOS_IDX] == span[1]:
                    ref = {
                        'begin': splitted_referents[idx][REF_BEGIN_IDX] - (span[0] + 1),
                        'end': splitted_referents[idx][REF_END_IDX] - (span[0] + 1),
                        'target': splitted_referents[idx][REF_BEGIN_TARGET_IDX:],
                    }
                    refs.append(ref)
                    idx += 1
                else:
                    break
            referents.append(refs)

        return referents


class DefaultTeacher(OneCommonTeacher):
    pass
