#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Taskmaster-2 implementation for ParlAI.

No official train/valid/test splits are available as of 2020-05-18, so we make our own
splits.
"""

import os
import pandas as pd
import hashlib
from collections import Counter
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.core.metrics import AverageMetric, F1Metric, BleuMetric
from parlai.utils.misc import warn_once
import json
import parlai.utils.logging as logging
from typing import Optional, Tuple
from parlai.core.message import Message
from parlai.utils.io import PathManager

import parlai.tasks.taskmaster2.build as build_

DOMAINS = [
    'flights',
    'food-ordering',
    'hotels',
    'movies',
    'restaurant-search',
    'sports',
    'music',
]

ONTO_TOKEN = "Onto:"
CALL_TOKEN = "Call:"
RESP_TOKEN = "Result:"


class _Abstract(DialogTeacher):
    """
    Abstract data loader.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument('--include-ontology', type=bool, default=False)
        argparser.add_argument(
            '--domains',
            nargs='+',
            default=DOMAINS,
            choices=DOMAINS,
            help='Uses last passed in configuration.',
        )
        return argparser

    def __init__(self, opt: Opt, shared=None):
        self.fold = opt['datatype'].split(':')[0]
        opt['datafile'] = self.fold
        self.dpath = os.path.join(opt['datapath'], 'taskmaster-2')
        if shared is None:
            warn_once(
                "Taskmaster2 is a beta dataset, and format may significantly change."
            )
            build_.build(opt)
        super().__init__(opt, shared)

    def _h(self, x):
        """
        Hash function.
        """
        h = int(hashlib.sha1(x.encode('utf-8')).hexdigest(), 16) % 10
        if h == 0:
            return 'valid'
        elif h == 1:
            return 'test'
        else:
            return 'train'

    def _normalize_annotation(self, anno):
        return anno

    def _load_data(self, fold, domains):
        # load up the ontology
        ontology = {}
        for section in domains:
            parts = []
            fn = os.path.join(self.dpath, section + '.onto.json')
            with PathManager.open(fn, 'r') as f:
                o = json.load(f)
            assert len(o) == 1
            o = list(o.values())[0]
            for sub in o:
                prefix = sub['prefix']
                parts += [
                    self._normalize_annotation(f'{prefix}.{a}')
                    for a in sub['annotations']
                ]
            ontology[section] = ' ; '.join(parts)

        chunks = []
        for section in domains:
            with PathManager.open(os.path.join(self.dpath, section + '.json')) as f:
                subset = pd.read_json(f)
            subset['domain'] = section
            chunks.append(subset)
        chunks = pd.concat(chunks, axis=0)
        # shuffle deterministically for randomness in few-shot training
        chunks = chunks.sample(frac=1.0, random_state=42)
        chunks['fold'] = self._label_fold(chunks)
        # only the fold we need here
        chunks = chunks[chunks.fold == fold].reset_index()
        chunks['ontology'] = chunks['domain'].apply(ontology.get)
        return chunks

    def _segments2text(self, segments):
        output = []
        slots = {}
        for segment in segments:
            val = segment['text']
            for anno_ in segment['annotations']:
                anno = anno_['name']
                anno = self._normalize_annotation(anno)
                output.append(f'{anno} = {val}')
                slots[anno] = val
        return " ; ".join(output), slots

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ):
        if 'metrics' in model_response and 'type' in teacher_action:
            # keep copies of metrics across both api calls/responses
            prefix = teacher_action['type']
            keys = list(model_response['metrics'].keys())
            for k in keys:
                self.metrics.add(f'{prefix}_{k}', model_response['metrics'][k])

        if 'text' not in model_response or not labels or 'type' not in teacher_action:
            return

        domain = teacher_action['domain']

        if teacher_action['type'] == 'apicall':
            # also count slot accuracy
            text = model_response['text']
            slot_guesses = set(
                text.replace(CALL_TOKEN + " ", "").split(' ; ')
            )  # prevent cheating via repeated guesses
            correct = 0
            for slot_guess in slot_guesses:
                if ' = ' not in slot_guess:
                    continue
                try:
                    slot, guess = slot_guess.split(' = ')
                except ValueError:
                    continue
                if teacher_action['slots'].get(slot) == guess:
                    self.metrics.add('slot_p', AverageMetric(1))
                    self.metrics.add(f'{domain}_slot_p', AverageMetric(1))
                    correct += 1
                else:
                    self.metrics.add('slot_p', AverageMetric(0))
                    self.metrics.add(f'{domain}_slot_p', AverageMetric(0))
                    logging.debug(
                        f"Bad slot guess '{slot_guess}' != {teacher_action['slots']}"
                    )
            if teacher_action['slots']:
                self.metrics.add(
                    'slot_r', AverageMetric(correct, len(teacher_action['slots']))
                )
                self.metrics.add(
                    f'{domain}_slot_r',
                    AverageMetric(correct, len(teacher_action['slots'])),
                )
                self.metrics.add(
                    'jga', AverageMetric(correct == len(teacher_action['slots']))
                )

        elif teacher_action['type'] == 'apiresp':
            # keep track of statistics by domain
            f1_metric = F1Metric.compute(model_response['text'], labels)
            bleu_metric = BleuMetric.compute(model_response['text'], labels)
            self.metrics.add(f'{domain}_lex_f1', f1_metric)
            self.metrics.add(f'{domain}_lex_bleu', bleu_metric)

            delex_text = model_response['text']
            delex_label = labels[0]
            # compute delexicalized string metrics
            for slot, value in teacher_action['slots'].items():
                delex_text = delex_text.replace(value, slot)
                delex_label = delex_label.replace(value, slot)
            f1_metric = F1Metric.compute(delex_text, (delex_label,))
            self.metrics.add('delex_f1', f1_metric)
            self.metrics.add(f'{domain}_delex_f1', f1_metric)
            bleu_metric = BleuMetric.compute(delex_text, [delex_label])
            self.metrics.add('delex_bleu', bleu_metric)
            self.metrics.add(f'{domain}_delex_bleu', bleu_metric)

    def setup_data(self, fold):
        domains = self.opt.get('domains', DOMAINS)
        chunks = self._load_data(fold, domains)
        domains_cnt = Counter()
        for _, row in chunks.iterrows():
            domains_cnt[row['domain']] += 1
            first = True
            utterances = row['utterances'][:]
            if (
                len(utterances) >= 3
                and utterances[0]['speaker'] == 'USER'
                and utterances[1]['speaker'] == 'ASSISTANT'
                and utterances[2]['speaker'] == 'ASSISTANT'
                and "help you?" in utterances[1]['text']
            ):
                # skip this one
                utterances.pop(1)
            if self.opt['include_ontology']:
                yield {'text': f"{ONTO_TOKEN} {row['ontology']}", 'label': ''}, True
                first = False
            while utterances:
                utt = utterances.pop(0)
                segtxt, slots = self._segments2text(utt.get('segments', []))
                if utt['speaker'] == 'USER':
                    yield {
                        'text': utt['text'],
                        'label': f'{CALL_TOKEN} {segtxt}',
                        'domain': row['domain'],
                        'slots': slots,
                        'type': 'apicall',
                    }, first
                    first = False
                elif utt['speaker'] == 'ASSISTANT':
                    yield {
                        'text': f'{RESP_TOKEN} {segtxt}',
                        'label': utt['text'],
                        'domain': row['domain'],
                        'slots': slots,
                        'type': 'apiresp',
                    }, first
                    first = False
        logging.debug(f"Fold {fold} domains: {domains_cnt}")


class DelexTeacher(_Abstract):
    def _label_fold(self, chunks):
        return chunks.conversation_id.apply(self._h)

    def _delexicalize(self, text, slots):
        for key, value in slots.items():
            text = text.replace(value, key)
        return text

    def setup_data(self, fold):
        domains_cnt = Counter()
        chunks = self._load_data(fold)
        for _, row in chunks.iterrows():
            domains_cnt[row['domain']] += 1
            first = True
            utterances = row['utterances'][:]
            if (
                len(utterances) >= 3
                and utterances[0]['speaker'] == 'USER'
                and utterances[1]['speaker'] == 'ASSISTANT'
                and utterances[2]['speaker'] == 'ASSISTANT'
                and "help you?" in utterances[1]['text']
            ):
                # skip this one
                utterances.pop(1)

            user_utterances = []
            asst_utterances = []
            while utterances:
                utt = utterances.pop(0)
                _, slots = self._segments2text(utt.get('segments', []))
                if utt['speaker'] == 'USER':
                    if asst_utterances:
                        yield {
                            'text': ' __BREAK__ '.join(user_utterances),
                            'label': ' __BREAK__ '.join(asst_utterances),
                            'domain': row['domain'],
                        }, first
                        first = False
                        user_utterances = []
                        asst_utterances = []
                    user_utterances.append(self._delexicalize(utt['text'], slots))
                elif utt['speaker'] == 'ASSISTANT':
                    asst_utterances.append(self._delexicalize(utt['text'], slots))
                    if not user_utterances:
                        user_utterances.append('__SILENCE__')
            if asst_utterances:
                yield {
                    'text': ' __BREAK__ '.join(user_utterances),
                    'label': ' __BREAK__ '.join(asst_utterances),
                    'domain': row['domain'],
                }, first


class TextOnlyTeacher(DelexTeacher):
    def _delexicalize(self, text, slots):
        return text


class FullShotTeacher(_Abstract):
    """
    The full shot teacher uses a standard 80-10-10 split, without regarding domain.
    """

    def _label_fold(self, chunks):
        return chunks.conversation_id.apply(self._h)


class FewShotTeacher(_Abstract):
    """
    Few shot teacher tests for generalization to new domains.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument(
            '--holdout',
            default=DOMAINS[0],
            choices=DOMAINS,
            help='Domain which is held out from test',
        )
        argparser.add_argument(
            '--n-shot',
            default=100,
            type=int,
            help='Number of few shot examples to provide in training fold.',
        )
        return super().add_cmdline_args(argparser)

    def _label_fold(self, chunks):
        folds = []
        num_shots = 0
        for _, row in chunks.iterrows():
            if row['domain'] != self.opt['holdout']:
                # if it's not in the holdout, always mark it train
                folds.append('train')
            else:
                # keep the same valid/test sets as in fullshot, and only leak
                # a small number of the training examples (i.e. throw away the
                # vast majority of our data but keep test sets the same)

                f = self._h(row['conversation_id'])
                if f != 'train':
                    folds.append(f)
                elif num_shots < self.opt['n_shot']:
                    folds.append('train')
                    num_shots += 1
                else:
                    folds.append('throwaway')
        return folds


class DefaultTeacher(FullShotTeacher):
    pass
