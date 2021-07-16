#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from parlai.core.opt import Opt
from parlai.utils.strings import normalize_reply
from parlai.core.teachers import MultiTaskTeacher

from .build import build
import os
import json
from typing import Optional
from parlai.core.params import ParlaiParser
import copy
import random
import math
from parlai.utils.logging import logger
from parlai.core.message import Message
from parlai.tasks.convai2.agents import NormalizedTeacherTrait, SelfOriginalTeacher

NOPERSONA = '__NO__PERSONA__BEAM__MIN__LEN__20__'
DUMMY_TEXT = '__SILENCE__'


def get_sessionbase_dir_path(opt, dpath, task_name):
    assert task_name in ['msc_personasummary', 'msc_dialogue']
    dpath = os.path.join(dpath, 'msc', task_name, f'session_{opt.get("session_id", 0)}')
    return dpath


def get_predicted_summary_path(dpath, is_session_level=True):
    if is_session_level:
        return os.path.join(
            dpath, 'msc', 'msc_dialogue', 'sessionlevel_summaries_subsample5.json'
        )
    else:
        return os.path.join(dpath, 'msc', 'msc_dialogue', 'summaries_subsample5.json')


class SessionBasePersonaSummaryTeacher(DialogTeacher):
    """
    Teacher that summarizes the persona lines.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('MSC Persona Summary Teacher options')
        agent.add_argument('--session-id', type=int, default=1, help="session id")
        agent.add_argument(
            '--summary-num-turns',
            type=int,
            default=-1,
            help="number of turns to infer persona",
        )
        agent.add_argument(
            '--nopersona-subsampling-weight',
            type=float,
            default=1,
            help="subampling ratio ",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.summary_num_turns = opt['summary_num_turns']
        assert (
            self.summary_num_turns < 0 or self.summary_num_turns % 2 == 0
        ), "Please choose an even number for turns"
        self.session_id = opt['session_id']
        assert opt['session_id'] <= 4, f"No data beyong session {opt['session_id']}!"
        assert (
            opt['session_id'] <= 3 or 'train' not in opt['datatype']
        ), f"No train data beyong session {opt['session_id']}!"
        self.nopersona_subsampling_weight = opt['nopersona_subsampling_weight']
        if 'test' in opt['datatype']:
            logger.warning(f'WARNING: Do not subsampling for {opt["datatype"]}')
            self.nopersona_subsampling_weight = 1
        assert (
            self.nopersona_subsampling_weight >= 0
            and self.nopersona_subsampling_weight <= 1
        ), "invalid subsampling weight"

        dpath = build(opt)
        opt['datafile'] = get_sessionbase_dir_path(opt, dpath, 'msc_personasummary')
        self.id = f'msc_personasummary_{self.session_id}'
        super().__init__(opt, shared)

    def setup_data(self, data_path):
        print('loading: ' + data_path)
        if self.datatype.startswith('train'):
            path_to_open = os.path.join(data_path, 'train.txt')
        elif self.datatype.startswith('valid'):
            path_to_open = os.path.join(data_path, 'valid.txt')
        else:
            path_to_open = os.path.join(data_path, 'test.txt')

        with PathManager.open(path_to_open) as f:
            raw_data = [json.loads(line.strip()) for line in f]

        data = []
        negative_data = []
        for dialog_dict in raw_data:
            current_episode = dialog_dict['dialog']
            init_personachat = dialog_dict['init_personachat']
            for end_idx in range(len(current_episode)):
                if self.summary_num_turns > 0:
                    start_index = max(0, end_idx - self.summary_num_turns + 1)
                else:
                    start_index = 0
                end_line_persona = (
                    current_episode[end_idx]['persona_text']
                    if 'persona_text' in current_episode[end_idx]
                    else NOPERSONA
                )
                dialog_texts = [
                    current_episode[i]['text'] for i in range(start_index, end_idx + 1)
                ]

                action = {
                    'id': self.id,
                    'text': '\n'.join(dialog_texts),
                    'labels': [end_line_persona],
                    'initial_data_id': dialog_dict['initial_data_id'],
                    'init_personas': init_personachat['init_personas'],
                    'utt_idx': end_idx,
                    'speaker_idx': end_idx % 2 + 1,
                    'session_id': self.session_id,
                }
                if end_line_persona == NOPERSONA:
                    negative_data.append(action)
                else:
                    data.append(action)

        size_to_sample = math.ceil(
            self.nopersona_subsampling_weight * len(negative_data)
        )
        data.extend(random.sample(negative_data, size_to_sample))
        random.shuffle(data)

        for episode in data:
            yield Message(episode), True


class SessionBaseMscTeacher(DialogTeacher):
    """
    Teacher that generate text in the multi-session chat.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Multi-Session Chat Task options')
        agent.add_argument(
            '--session-id',
            type=int,
            default=2,
            help="session id, session_id = 1 refers to convai2 teacher and it's not supported here",
        )
        agent.add_argument(
            '--previous-persona-type',
            type=str,
            default="raw_history",
            choices=[
                'none',
                'goldsum_self',
                'goldsum_both',
                'goldsum_their',
                'predsum_self',
                'predsum_both',
                'predsum_their',
                'predsum_utt_self',
                'predsum_utt_both',
                'predsum_utt_their',
                'init_self',
                'init_both',
                'init_their',
                'raw_history',
            ],
            help="type of previous context to include as context. "
            "the 'goldsum_' prefix refers to gold persona summaries from crowdworkers; "
            "the 'predsum_' prefix refers to predicted persona summaries from a summarization model; "
            "the 'init_' prefix refers to the original persona lines used to ground the PersonaChat conversations. ",
        )
        agent.add_argument(
            '--your-persona-first',
            type=bool,
            default=False,
            help="whether to prepend your persona first or not",
        )
        agent.add_argument(
            '--session-openning',
            type=bool,
            default=False,
            help="whether to only include session openning or not",
        )
        agent.add_argument(
            '--label-speaker-id',
            type=str,
            default="both",
            choices=['self', 'both', 'their'],
            help="the speaker id of the 'labels' field,",
        )
        agent.add_argument(
            '--include-time-gap',
            type=bool,
            default=False,
            help="whether to include time passed since last conversation in the context",
        )
        agent.add_argument(
            '--history-time-gaps-token',
            type=str,
            default=None,
            help="time tokens in the previous raw dialogue history, e.g. 'time:' ",
        )
        agent.add_argument(
            '--history-person-tokens',
            type=str,
            default=None,
            help="person tokens in the previous raw dialogue history, e.g. 'p1:,p2:' ",
        )
        agent.add_argument(
            '--previous-session-delimiter',
            type=str,
            default=None,
            help="delimiter between previous sessions in the context, such as '__NEXT_SESSION__' ",
        )
        return parser

    def __init__(self, opt, shared=None):
        assert opt['session_id'] <= 5, f"No data beyong session {opt['session_id']}!"
        assert (
            opt['session_id'] <= 4 or 'train' not in opt['datatype']
        ), f"No train data beyong session {opt['session_id']}!"
        assert (
            not opt['previous_persona_type'].startswith('predsum')
            or opt['session_id'] <= 4
            or (
                opt['session_id'] == 5
                and ('valid' in opt['datatype'] or 'test' in opt['datatype'])
            )
        ), f"No predicted summary for session {opt['session_id']}"
        self.previous_persona_type = opt['previous_persona_type']
        self.session_openning = opt.get('session_openning', False)
        if self.session_openning:
            opt['label_speaker_id'] = 'their'
        # NOTE: session_id = 1: personachat
        self.session_id = opt['session_id']
        self.label_speaker_id = opt["label_speaker_id"]
        self.your_persona_first = opt['your_persona_first']
        self.include_last_time_gap = opt['include_time_gap']
        self.history_time_gaps_token = opt['history_time_gaps_token']
        if self.history_time_gaps_token:
            self.include_last_time_gap = False
        self.history_person_tokens = opt['history_person_tokens']
        self.use_predicted_summary = self.previous_persona_type.startswith('predsum')
        self.previous_session_delimiter = opt.get('previous_session_delimiter', None)
        if self.history_person_tokens is not None:
            self.history_person_tokens = self.history_person_tokens.split(",")
        self.msc_dpath = build(opt)
        opt['datafile'] = get_sessionbase_dir_path(opt, self.msc_dpath, 'msc_dialogue')

        self.id = f'msc_dialogue_{self.session_id}'
        super().__init__(opt, shared)

    def normalize_replies(self, x):
        xs = [xt.strip() for xt in x.split('\n')]
        xs2 = []
        for x in xs:
            if 'your persona:' in x:
                # Normalize the sentence appearing after 'your persona:'
                x = x[len('your persona: ') :]
                x = normalize_reply(x)
                x = 'your persona: ' + x
            elif "partner's persona: " in x:
                x = x[len("partner's persona: ") :]
                x = normalize_reply(x)
                x = "partner's persona: " + x
            elif x != DUMMY_TEXT:
                x = normalize_reply(x)
            xs2.append(x)
        return "\n".join(xs2)

    def setup_data(self, datafile):
        print('loading: ' + datafile)
        if self.datatype.startswith('train'):
            path_to_open = os.path.join(datafile, 'train.txt')
        elif self.datatype.startswith('valid'):
            path_to_open = os.path.join(datafile, 'valid.txt')
        else:
            path_to_open = os.path.join(datafile, 'test.txt')

        with PathManager.open(path_to_open) as f:
            raw_data = [json.loads(line.strip()) for line in f]

        data = []
        label_speaker_id_range = {}
        predicted_summary_dict = {}
        if self.use_predicted_summary:
            is_session_level = not ('utt_' in self.previous_persona_type)
            predsum_path = get_predicted_summary_path(self.msc_dpath, is_session_level)
            logger.warning(f"use the predicted summary from {predsum_path}")
            with PathManager.open(predsum_path) as jsonfile:
                predicted_summary_dict = json.load(jsonfile)

        def _get_time_gap(time_num, time_unit, time_token=""):
            time_gap = str(time_num) + ' ' + time_unit
            return f'{time_token} {time_gap}' if len(time_token) > 0 else time_gap

        def _compile_persona_dialog_input(
            dialog, personas, previous_dialogs, label_speaker_id
        ):
            new_dialog = copy.deepcopy(dialog)
            new_previous_dialogs = copy.deepcopy(previous_dialogs)
            your_persona = ""
            partner_persona = ""
            if label_speaker_id == 'self':
                your_persona = '\n'.join([f'your persona: {x}' for x in personas[1]])
                partner_persona = '\n'.join(
                    [f"partner's persona: {x}" for x in personas[0]]
                )
            elif label_speaker_id == 'their':
                your_persona = '\n'.join([f'your persona: {x}' for x in personas[0]])
                partner_persona = '\n'.join(
                    [f"partner's persona: {x}" for x in personas[1]]
                )
                for prev_dialog in new_previous_dialogs:
                    prev_dialog['dialog'].insert(0, {"text": DUMMY_TEXT})
                    if len(prev_dialog['dialog']) % 2 == 1 and (
                        self.history_person_tokens is None
                    ):
                        prev_dialog['dialog'].append({"text": DUMMY_TEXT})
                new_dialog.insert(0, {"text": DUMMY_TEXT})

            return your_persona, partner_persona, new_dialog, new_previous_dialogs

        for dialog_dict in raw_data:
            initial_data_id = dialog_dict['metadata']['initial_data_id']
            if self.label_speaker_id == 'both':
                label_speaker_id_range = ['their', 'self']
            else:
                label_speaker_id_range = [self.label_speaker_id]

            for label_speaker_id in label_speaker_id_range:
                if self.use_predicted_summary:
                    personas_to_complie = predicted_summary_dict[
                        str(self.session_id - 1)
                    ][initial_data_id]
                elif self.previous_persona_type.startswith('init'):
                    personas_to_complie = dialog_dict['init_personas']
                else:
                    personas_to_complie = dialog_dict['personas']

                (
                    your_persona,
                    partner_persona,
                    new_dialog,
                    new_previous_dialogs,
                ) = _compile_persona_dialog_input(
                    dialog_dict['dialog'],
                    personas_to_complie,
                    dialog_dict['previous_dialogs'],
                    label_speaker_id,
                )
                previous_sessions_msgs = []
                if self.previous_persona_type == 'raw_history':
                    for d_id in range(len(new_previous_dialogs)):
                        previous_dialog_msg = [
                            x['text'] for x in new_previous_dialogs[d_id]['dialog']
                        ]
                        if self.history_person_tokens:
                            previous_dialog_msg = [
                                self.history_person_tokens[i % 2] + ' ' + text
                                for i, text in enumerate(previous_dialog_msg)
                                if text != DUMMY_TEXT
                            ]
                        if self.history_time_gaps_token:
                            time_gap_i = _get_time_gap(
                                new_previous_dialogs[d_id]['time_num'],
                                new_previous_dialogs[d_id]['time_unit'],
                                time_token=self.history_time_gaps_token,
                            )
                            previous_sessions_msgs.append(
                                '\n'.join(previous_dialog_msg + [time_gap_i])
                            )
                        else:
                            previous_sessions_msgs.append(
                                '\n'.join(previous_dialog_msg)
                            )

                if self.previous_session_delimiter is not None:
                    previous_sessions_msgs = [
                        val
                        for pair in zip(
                            previous_sessions_msgs,
                            [self.previous_session_delimiter]
                            * len(previous_sessions_msgs),
                        )
                        for val in pair
                    ]
                previous_sessions_msgs = '\n'.join(previous_sessions_msgs)

                episode = []
                for i in range(0, len(new_dialog) - 1, 2):
                    text = new_dialog[i]['text']
                    partner_persona_one_line = partner_persona.replace('\n', '').split(
                        "partner's persona: "
                    )
                    your_persona_one_line = your_persona.replace('\n', '').split(
                        "your persona: "
                    )
                    action = {
                        'id': self.id,
                        'text': self.normalize_replies(text),
                        'labels': [self.normalize_replies(new_dialog[i + 1]['text'])],
                        'session_id': self.session_id,
                        'initial_data_id': initial_data_id,
                        'personas': f'{partner_persona}\n{your_persona}',
                        'personas_one_line': f"partner's persona: {' '.join(partner_persona_one_line)}\nyour persona: {' '.join(your_persona_one_line)}",
                    }
                    episode.append(action)
                    if self.session_openning:
                        break

                persona_context_str = ""
                if 'self' in self.previous_persona_type:
                    persona_context_str = your_persona
                elif 'their' in self.previous_persona_type:
                    persona_context_str = partner_persona
                elif 'both' in self.previous_persona_type:
                    if self.your_persona_first:
                        persona_context_str = (
                            (your_persona + '\n') if len(your_persona) > 0 else ""
                        ) + partner_persona
                    else:
                        persona_context_str = (
                            (partner_persona + '\n') if len(partner_persona) > 0 else ""
                        ) + your_persona
                elif self.previous_persona_type == 'raw_history':
                    persona_context_str = previous_sessions_msgs

                if self.include_last_time_gap:
                    time_gap = _get_time_gap(
                        dialog_dict['previous_dialogs'][-1]['time_num'],
                        dialog_dict['previous_dialogs'][-1]['time_unit'],
                    )
                    persona_context_str = (
                        (persona_context_str + '\n')
                        if len(persona_context_str) > 0
                        else ""
                    ) + f'[{time_gap}]'

                if persona_context_str and len(persona_context_str) > 0:
                    episode[0]['text'] = persona_context_str + '\n' + episode[0]['text']

                data.append(episode)

        for episode in data:
            start_idx = 0
            for i, turn in enumerate(episode):
                yield Message(turn), i == start_idx


class PersonaSummaryTeacher(MultiTaskTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('MSC Summary Teacher Args')
        parser.add_argument(
            '--include-last-session',
            type=bool,
            default=False,
            help="whether to include session 4 for valid and test splits",
        )
        SessionBasePersonaSummaryTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        msc_tasks = [
            'msc:SessionBasePersonaSummary:session_id=1',
            'msc:SessionBasePersonaSummary:session_id=2',
            'msc:SessionBasePersonaSummary:session_id=3',
        ]
        if opt.get('include_last_session', False) and 'train' not in opt['datatype']:
            msc_tasks += ['msc:SessionBasePersonaSummary:session_id=4']
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(msc_tasks)
        super().__init__(opt, shared)


class Session1NormalizedTrait(NormalizedTeacherTrait):
    """
    Trait for flatten persona into one line.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Session Level NormalizedTeacher arguments')
        agent.add_argument(
            '--is-convai2-session-level',
            type=bool,
            default=False,
            help="whether to flatten the persona lines into a single persona line per speaker",
        )
        return agent

    def __init__(self, opt, shared=None):
        self.is_convai2_session_level = opt.get('is_convai2_session_level', False)
        super().__init__(opt, shared)

    def normalize_replies(self, x):
        xs = x.split('\n')
        your_personas = []
        partner_personas = []
        non_personas = []
        for x in xs:
            if x.startswith('your persona: '):
                # Normalize the sentence appearing after 'your persona:'
                x = x[len('your persona: ') :]
                x = normalize_reply(x)
                your_personas.append(x)
            elif x.startswith("partner's persona: "):
                x = x[len("partner's persona: ") :]
                x = normalize_reply(x)
                partner_personas.append(x)
            else:
                x = normalize_reply(x)
                non_personas.append(x)
        xs2 = []
        if not self.is_convai2_session_level:
            your_personas = ['your persona: ' + yx for yx in your_personas]
            partner_personas = ["partner's persona: " + px for px in partner_personas]
        else:
            if your_personas:
                your_personas = ['your persona: ' + " ".join(your_personas)]
            if partner_personas:
                partner_personas = ["partner's persona: " + " ".join(partner_personas)]
        if self.your_persona_first:
            xs2.extend(your_personas)
            xs2.extend(partner_personas)
        else:
            xs2.extend(partner_personas)
            xs2.extend(your_personas)
        xs2.extend(non_personas)
        return '\n'.join(xs2)


class Session1SelfTeacher(Session1NormalizedTrait, SelfOriginalTeacher):
    """
    Convai2 as Session 1.
    """

    pass


class MscTeacher(MultiTaskTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('Multi Session Chat (MSC) Teacher Args')
        parser.add_argument(
            '--include-session1',
            type=bool,
            default=True,
            help="whether to include session 1 (convai2:normalized)",
        )
        parser.add_argument(
            '--include-last-session',
            type=bool,
            default=False,
            help="whether to include session 5",
        )
        SessionBaseMscTeacher.add_cmdline_args(parser, partial_opt)
        Session1SelfTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        msc_tasks = [
            'msc:SessionBaseMsc:session_id=2',
            'msc:SessionBaseMsc:session_id=3',
            'msc:SessionBaseMsc:session_id=4',
        ]
        if opt.get('include_session1', False) and not opt['session_openning']:
            if opt['previous_persona_type'] in [
                'predsum_self',
                'predsum_both',
                'predsum_their',
            ]:
                msc_tasks = [
                    'msc:Session1Self:is_convai2_session_level=True'
                ] + msc_tasks
            else:
                msc_tasks = [
                    'msc:Session1Self:is_convai2_session_level=False'
                ] + msc_tasks
        if opt.get('include_last_session', False) and 'train' not in opt['datatype']:
            msc_tasks += ['msc:SessionBaseMsc:session_id=5']
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(msc_tasks)
        super().__init__(opt, shared)


class DefaultTeacher(MscTeacher):
    pass
