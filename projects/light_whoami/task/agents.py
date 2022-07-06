#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Teachers used in the Am I Me or You task.
"""
from abc import ABC
from collections import deque
import re
from typing import Optional
from parlai.core.build_data import modelzoo_path
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.tasks.light_dialog.agents import SimpleMultiTeacher
from parlai.utils.data import DatatypeHelper
import parlai.utils.logging as logging

import copy
import os
import random
import torch
import torch.nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from projects.light_whoami.task.utils import (
    DEFAULT_DELIM,
    CONTEXT_KEYS,
    extract_characters,
    WHO_AM_I,
    WHO_ARE_YOU,
    SELF,
    PARTNER,
    WHO_IS_THIS,
    maybe_annotate,
)
import projects.light_whoami.task.mutators  # type: ignore


class BaseSimpleMultiTeacher(SimpleMultiTeacher):
    pass


########################################
# RPA Classification Training Teachers #
########################################


class SpeakerClassifierTeacher(SimpleMultiTeacher, ABC):
    """
    Speaker Classifier Teacher.

    This teacher is used for the RPA classifier training.

    This teacher should NOT be used directly.
    """

    @staticmethod
    def add_speaker_classifier_cmdline_args(parser):
        group = parser.add_argument_group('Speaker Classifier args')
        group.add_argument(
            '--speaker-label-type',
            type=str,
            default='speaker',
            choices=['speaker', 'listener'],
            help='Whether the label should be the person who is speaking, or the '
            'person whom is being spoken to.',
        )
        group.add_argument(
            '--classifier-label-type',
            type=str,
            default='character',
            choices=['character', 'role'],
            help='Whether the label should be the character, or role in the '
            'conversation (i.e., self or partner)',
        )
        group.add_argument(
            '--num-utterances',
            type=int,
            default=-1,
            help='Number of utterances to include in the example prior to final utterance. '
            'Default <=0 will use all utterances in an episode.',
        )
        group.add_argument(
            '--annotate-speaker',
            type='bool',
            default=True,
            help='If true, annotate each utterance with speaker name',
        )
        group.add_argument(
            '--speaker-separator',
            type='bool',
            default=True,
            help='Whether to surround speaker annotation with special tokens',
        )
        group.add_argument(
            '--speaker-annotation-position',
            type=str,
            choices=['prefix', 'suffix'],
            default='suffix',
            help='If annotating speaker, where to add speaker annotation relative to text.',
        )
        group.add_argument(
            '--include-light-context',
            type='bool',
            default=True,
            help='Whether to prepend each episode with light context',
        )
        group.add_argument(
            '--exclude-from-context',
            type=str,
            default=None,
            help='comma-separated list of keys for items to exclude from the context. '
            f"Choices: {','.join(CONTEXT_KEYS)}",
        )
        group.add_argument(
            '--exclude-from-context-delimiter',
            type=str,
            default=',',
            help='how to split the exclude_from_context specifications',
        )
        group.add_argument(
            '--inline-candidate-type',
            type=str,
            default='all',
            choices=['all', 'conversation'],
            help='What to include as inline label candidates. All will use all '
            'characters from the data split; conversation will be just the characters '
            'within the conversation.',
        )
        group.add_argument(
            '--num-train-inline-candidates',
            type=int,
            default=100,
            help='If --inline-canidate-type is `all`, how many candidates to subsample for training '
            'Set to -1 to keep all cands',
        )
        group.add_argument(
            '--left-to-right',
            type='bool',
            default=False,
            help='If true, generate an example for every word in the label; '
            'i.e., classify from left to right',
        )
        group.add_argument(
            '--delimit-label-control',
            type='bool',
            default=True,
            help='If False, do not insert the delimiter between the context and label control',
        )

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        SpeakerClassifierTeacher.add_speaker_classifier_cmdline_args(parser)
        return parser

    def __init__(self, opt, shared=None):
        """
        Ensure that certain opt parameters are set correctly.
        """
        if opt['annotate_speaker']:
            assert opt[
                'light_use_person_names'
            ], 'must set --light_use_person_names True if annotating speaker'
        self.annotate_speaker = opt['annotate_speaker']
        self.speaker_annotation_position = opt['speaker_annotation_position']
        self.speaker_label_type = opt['speaker_label_type']
        self.classifier_label_type = opt['classifier_label_type']
        self.speaker_separator = opt['speaker_separator']
        self.include_light_context = opt['include_light_context']
        self.num_utterances = opt['num_utterances']
        self.inline_candidate_type = opt['inline_candidate_type']
        self.num_train_inline_candidates = opt['num_train_inline_candidates']
        self.delimiter = opt.get('delimiter', DEFAULT_DELIM)
        self.use_speech_prefix = opt['light_use_speech_prefix']
        self.left_to_right = opt['left_to_right']
        self.delimit_label_control = opt['delimit_label_control']
        if opt.get('exclude_from_context'):
            self.exclude_from_context = opt['exclude_from_context'].split(
                opt['exclude_from_context_delimiter']
            )
        else:
            self.exclude_from_context = []
        assert all(c in CONTEXT_KEYS for c in self.exclude_from_context)
        # manually set persona to all
        opt['light_use_persona'] = 'all'
        super().__init__(opt, shared)

    def share(self):
        shared = super().share()
        shared['candidates'] = self.candidates
        return shared

    @classmethod
    def _build_candidates(cls, episodes: List[Message]) -> List[str]:
        """
        Build up set of candidates (i.e., characters) from the data.
        """
        candidates = set()
        for ep in episodes:
            character_mapping = extract_characters(ep[0]['text'])
            for c in character_mapping.values():
                candidates.add(c)
        return list(candidates)

    @classmethod
    def _explode_episode(
        cls,
        episode: List[Message],
        exclude_from_context: List[str],
        use_speech_prefix: bool,
    ) -> Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]:
        """
        Extract context, characters, and list of utterances from an episode.

        Additionally return initial start and end indices to use when constructing
        new eps.

        :param episode:
            list of examples
        :param exclude_from_context:
            list of context keys to exclude from the light context
        :param use_speech_prefix:
            if true, prepend label text with speech prefix.

        :return (context, characters, utterances):
            context: string context
            characters: dict mapping char key to character
            utterances: list of tuples (speaker_name, utterance, listener_name)
        """
        utterances = []
        context = episode[0]['text']
        characters = extract_characters(context)
        me = characters['_self_name']
        you = characters['_partner_name']
        if not context.split('\n')[-1].startswith('_'):
            # begin conversation with partner
            utterances.append((you, context.split('\n')[-1], me))
            context = '\n'.join(context.split('\n')[:-1])

        if exclude_from_context:
            context = '\n'.join(
                [
                    c
                    for c in context.split('\n')
                    if not any(c.startswith(x) for x in exclude_from_context)
                ]
            )

        for i, ex in enumerate(episode):
            if i != 0:
                # skip context
                utterances.append((you, ex['text'], me))
            prefix = '_self_say ' if use_speech_prefix else ''
            utterances.append((me, f"{prefix}{ex['labels'][0]}", you))

        return context, characters, utterances

    def _build_rpa_episodes(self, ep: List[Message]) -> List[Message]:
        """
        Construct new episodes from old, LIGHT ones.

        enumerate over all possible start and label positions

        :param ep:
            episode to explode and build into new eps

        :return episodes:
            return a list of episodes after enumerating over possible labels.
        """
        episodes = []
        context, characters, utterances = self._explode_episode(
            ep, self.exclude_from_context, self.use_speech_prefix
        )
        candidates = (
            self.candidates
            if self.inline_candidate_type == 'all'
            else list(characters.values())
        )

        # determine initial start and end indices
        num_utts = self.num_utterances
        if num_utts < 0:
            num_utts = len(utterances) - 1

        start_idx, end_idx = (0, num_utts - 1)

        # Enumerate over all possible start, end positions
        while end_idx < len(utterances) - 1:
            # Step 0: (maybe) annotate the prior utterances of dialogue
            prev_utts = [
                maybe_annotate(
                    *utt[:-1],
                    self.annotate_speaker,
                    self.speaker_separator,
                    self.speaker_annotation_position,
                )
                for utt in utterances[start_idx:end_idx]
            ]
            if self.include_light_context:
                prev_utts = [context] + prev_utts

            # Step 1: enumerate over each successive utterance
            for speaker, label, listener in utterances[end_idx:]:
                # Step 2: determine the label control / task type
                if self.classifier_label_type == 'character':
                    speaker_label = (
                        speaker if self.speaker_label_type == 'speaker' else listener
                    )
                    label_control = (
                        WHO_AM_I
                        if self.speaker_label_type == 'speaker'
                        else WHO_ARE_YOU
                    )
                else:
                    speaker_label = (
                        SELF if speaker == characters['_self_name'] else PARTNER
                    )
                    label_control = WHO_IS_THIS
                # Step 3: Determine what label candidates to use
                if (
                    self.num_train_inline_candidates > 0
                    and DatatypeHelper.is_training(self.datatype)
                    and self.inline_candidate_type == 'all'
                ):
                    label_cands = [speaker, listener]
                    while speaker in label_cands and listener in label_cands:
                        label_cands = random.sample(
                            candidates, self.num_train_inline_candidates - 2
                        )
                    label_cands += [speaker, listener]
                    random.shuffle(label_cands)
                else:
                    label_cands = candidates

                # Step 4: Build the Message
                if self.left_to_right:
                    label_words = label.split(' ')
                    for i in range(1, len(label_words) + 1):
                        if self.delimit_label_control:
                            text = self.delimiter.join(
                                prev_utts + [label_control, ' '.join(label_words[:i])]
                            )
                        else:
                            text = self.delimiter.join(
                                prev_utts[:-1]
                                + [f"{prev_utts[-1]} {label_control}"]
                                + [' '.join(label_words[:i])]
                            )
                        message = Message(
                            {
                                'text': text,
                                'labels': [speaker_label],
                                'label_candidates': label_cands,
                                'episode_done': True,
                            }
                        )
                        episodes.append([message])
                else:
                    if self.delimit_label_control:
                        text = self.delimiter.join(prev_utts + [label_control, label])
                    else:
                        text = self.delimiter.join(
                            prev_utts[:-1]
                            + [f"{prev_utts[-1]} {label_control}"]
                            + [label]
                        )
                    message = Message(
                        {
                            'text': text,
                            'labels': [speaker_label],
                            'label_candidates': label_cands,
                            'episode_done': True,
                        }
                    )
                    episodes.append([message])

            if start_idx == end_idx:
                # edge case where num_utterances == 1
                break
            else:
                start_idx += 1
                end_idx += 1

        return episodes

    def _setup_data(self, path: str):
        """
        Override to reset the labels and label candidates.

        From a context and list of utterances, construct every possible episode
        given the following constraints:

        - Number of preceding utterances is equal to the value of `--num-utterances`
        - The utterance to "classify" follows the preceding utterances at some point

        So, every possible start position and end position will be considered.
        """
        super()._setup_data(path)
        self.candidates = self._build_candidates(self.episodes)
        random.seed(42)
        new_eps = []
        for ep in self.episodes:
            episodes = self._build_rpa_episodes(ep)
            new_eps += episodes

        self.episodes = new_eps
        self.num_eps = len(self.episodes)
        self.num_exs = len(self.episodes)


class WhoIsSpeakingTeacher(SpeakerClassifierTeacher):
    """
    Label is the speaker.
    """

    def __init__(self, opt, shared=None):
        opt['speaker_label_type'] = 'speaker'
        super().__init__(opt, shared)


class WhoIsSpeakingLeftToRightTeacher(SpeakerClassifierTeacher):
    """
    Label is the speaker.

    Left to right (partial sequences)
    """

    def __init__(self, opt, shared=None):
        opt['speaker_label_type'] = 'speaker'
        opt['left_to_right'] = True
        super().__init__(opt, shared)


class WhoIsListeningTeacher(SpeakerClassifierTeacher):
    """
    Label is the listener.
    """

    def __init__(self, opt, shared=None):
        opt['speaker_label_type'] = 'listener'
        super().__init__(opt, shared)


class AmIMeOrYouTeacher(SpeakerClassifierTeacher):
    """
    Label is PARTNER or SELF.
    """

    def __init__(self, opt, shared=None):
        opt['classifier_label_type'] = 'role'
        super().__init__(opt, shared)


class AllUtteranceLengthsTeacher(SpeakerClassifierTeacher):
    """
    This teacher will enumerate **every single** combination of episodes with utterance
    lengths.

    Finds the minimum number of utterances through the entire data, and goes from there.
    """

    def _setup_data(self, path):
        super()._setup_data(path)
        max_num_utts = min(
            len(ep[0]['text'].split(self.delimiter)) for ep in self.episodes
        )
        episodes = copy.deepcopy(self.episodes)
        for i in range(0, max_num_utts - 1):
            self.num_utterances = i
            super()._setup_data(path)
            episodes += copy.deepcopy(self.episodes)
        self.episodes = episodes
        self.num_eps = len(self.episodes)
        self.num_exs = len(self.episodes)


######################################
# RPA Classifier Evaluation Teachers #
######################################


class ResponseClassifierTeacher(SimpleMultiTeacher):
    """
    This teacher examines model responses and classifies whether the model responded as
    the correct character.

    Evaluation is entirely dependent on the classifier specified.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        from projects.light_whoami.agents.rpa_rerank import RPAReranker

        RPAReranker.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        from projects.light_whoami.agents.rpa_rerank import RPAReranker

        assert opt.get('predictor_model_file') and os.path.exists(
            modelzoo_path(opt['datapath'], opt['predictor_model_file'])
        ), f"must specify a proper RPA predictor to use this teacher, file path invalid: {opt['predictor_model_file']}"

        if not shared:
            self.classifier = RPAReranker(opt)
        else:
            self.classifier = shared['classifier']
        self.context = []
        self.delimiter = opt.get('delimiter', DEFAULT_DELIM)

        super().__init__(opt, shared)

    def share(self):
        shared = super().share()
        shared['classifier'] = self.classifier
        return shared

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ) -> None:
        """
        Compute RPA for a model response.

        :param teacher_action:
            The message last sent from this teacher.
        :param labels:
            The previous correct labels
        :param model_response:
            The raw response from the model
        """
        if not model_response or not model_response.get('text'):
            return
        self.context.append(teacher_action['text'])
        context = self.delimiter.join(self.context)
        characters = extract_characters(context)
        correct_character = characters['_self_name']
        model_text = model_response['text']
        classifier_act = self.classifier.classify(context, model_text)
        predicted_character = classifier_act['text']
        correct_prediction = int(predicted_character == correct_character)
        self.metrics.add('character_accuracy', AverageMetric(correct_prediction))
        scores = F.softmax(classifier_act['sorted_scores'].float(), dim=0)
        if teacher_action['episode_done']:
            self.context = []
        else:
            assert labels
            self.context.append(labels[0])

        return predicted_character == correct_character

    def _setup_data(self, path: str):
        super()._setup_data(path)

        candidates = SpeakerClassifierTeacher._build_candidates(self.episodes)
        for ep in self.episodes:
            for ex in ep:
                ex.force_set('character_candidates', candidates)


class MultiObjectiveTeacher(SimpleMultiTeacher):
    """
    Include characters, inline candidates to train a generator with multiple objectives.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        SimpleMultiTeacher.add_cmdline_args(parser, partial_opt)
        SpeakerClassifierTeacher.add_speaker_classifier_cmdline_args(parser)
        parser.set_defaults(num_train_inline_candidates=-1)
        return parser

    def _setup_data(self, path):
        super()._setup_data(path)
        logging.info('Building Candidates')
        self.candidates = SpeakerClassifierTeacher._build_candidates(self.episodes)
        logging.info('Setting up character labels')
        for ep in self.episodes:
            context_str = ep[0]['text']
            for ex in ep:
                n_cands = self.opt['num_train_inline_candidates']
                if n_cands > 0 and DatatypeHelper.is_training(self.datatype):
                    speaker, listener = extract_characters(context_str).values()
                    label_cands = [speaker, listener]
                    while speaker in label_cands and listener in label_cands:
                        label_cands = random.sample(self.candidates, n_cands - 2)
                    label_cands += [speaker, listener]
                    random.shuffle(label_cands)
                else:
                    label_cands = self.candidates
                ex.force_set('character_candidates', label_cands)


class MultiObjectiveResponseClassifierTeacher(
    ResponseClassifierTeacher, MultiObjectiveTeacher
):
    def _setup_data(self, path: str):
        SimpleMultiTeacher._setup_data(self, path)
        ResponseClassifierTeacher._setup_data(self, path)
        MultiObjectiveTeacher._setup_data(self, path)


class DefaultTeacher(SpeakerClassifierTeacher):
    """
    Default to default options.
    """

    pass
