#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional

import parlai.utils.logging as logging
from parlai.utils.io import PathManager
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from .build import build


DEFAULT_TRAIN_EXPERIENCER_ONLY = False


class EmpatheticDialoguesTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = self._get_datafile(opt)
        self.id = 'empathetic_dialogues'
        self.experiencer_side_only = self._get_experiencer_side_only(opt)
        super().__init__(opt, shared)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('EmpatheticDialogues teacher arguments')
        agent.add_argument(
            '--train-experiencer-only',
            type='bool',
            default=DEFAULT_TRAIN_EXPERIENCER_ONLY,
            # i.e. do not include the other side of the conversation where the Listener
            # (responder) utterance would be the text and the Speaker (experiencer)
            # utterance would be the label
            help='In the train set, only use Speaker (experiencer) utterances as text and Listener (responder) utterances as labels.',
        )
        return parser

    def _get_base_datatype(self, opt) -> str:
        return opt['datatype'].split(':')[0]

    def _get_datafile(self, opt) -> str:
        """
        Get the datafile path.

        Useful for subclassed teachers.
        """
        base_datatype = self._get_base_datatype(opt)
        return os.path.join(
            opt['datapath'],
            'empatheticdialogues',
            'empatheticdialogues',
            base_datatype + '.csv',
        )

    def _get_experiencer_side_only(self, opt):
        """
        Determine which side(s) of the conversation to use.
        """
        base_datatype = self._get_base_datatype(opt)
        return (
            opt.get('train_experiencer_only', DEFAULT_TRAIN_EXPERIENCER_ONLY)
            and base_datatype == 'train'
        ) or base_datatype != 'train'

    def setup_data(self, path):

        logging.debug('loading: ' + path)
        with PathManager.open(path) as f:
            df = f.readlines()

        turn_idx = 1
        responder_text_dialogue = []
        experiencer_text_dialogue = []
        data = []
        for i in range(1, len(df)):

            cparts = df[i - 1].strip().split(",")
            sparts = df[i].strip().split(",")

            if cparts[0] == sparts[0]:

                # Check that the turn number has incremented correctly
                turn_idx += 1
                assert (
                    int(cparts[1]) + 1 == int(sparts[1]) and int(sparts[1]) == turn_idx
                )

                contextt = cparts[5].replace("_comma_", ",")
                label = sparts[5].replace("_comma_", ",").strip()
                prompt = sparts[2]
                sit = sparts[3].replace("_comma_", ",")
                if len(sparts) == 9:
                    if sparts[8] != '':
                        inline_label_candidates = [
                            cand.replace("_comma_", ",").replace("_pipe_", "|")
                            for cand in sparts[8].split('|')
                        ]
                    else:
                        inline_label_candidates = None
                elif len(sparts) == 8:
                    inline_label_candidates = None
                else:
                    raise ValueError(f'Line {i:d} has the wrong number of fields!')

                dialogue_parts = Message(
                    {
                        'text': contextt,
                        'labels': [label],
                        'emotion': prompt,
                        'situation': sit,
                    }
                )
                if inline_label_candidates is not None:
                    inline_label_candidates = [
                        lc.strip() for lc in inline_label_candidates
                    ]
                    dialogue_parts.force_set(
                        'label_candidates', inline_label_candidates
                    )

                if int(sparts[1]) % 2 == 0:
                    # experiencer is the "text" and responder is the "label"
                    experiencer_text_dialogue.append(dialogue_parts)
                else:
                    # responder is the "text" and experiencer is the "label"
                    responder_text_dialogue.append(dialogue_parts)

            else:

                # We've finished the previous episode, so add it to the data
                turn_idx = 1
                data += self._select_dialogues_to_add(
                    experiencer_text_dialogue, responder_text_dialogue
                )
                experiencer_text_dialogue = []
                responder_text_dialogue = []

        # Add in the final episode
        data += self._select_dialogues_to_add(
            experiencer_text_dialogue, responder_text_dialogue
        )

        for episode in data:
            for entry_idx, entry in enumerate(episode):
                new_episode = entry_idx == 0
                yield entry, new_episode

    def _select_dialogues_to_add(
        self,
        experiencer_text_dialogue: List[Message],
        responder_text_dialogue: List[Message],
    ) -> List[List[Message]]:
        """
        Return conversation halves to add to self.data.

        Given lists corresponding to the conversation turns from both sides of the
        conversation, return only the list(s) that will be used by the teacher.
        Optionally include both sides of the conversation.
        """
        selected_dialogues = []
        if len(experiencer_text_dialogue) > 0:
            selected_dialogues.append(experiencer_text_dialogue)
        if len(responder_text_dialogue) > 0 and not self.experiencer_side_only:
            selected_dialogues.append(responder_text_dialogue)
        return selected_dialogues


class DefaultTeacher(EmpatheticDialoguesTeacher):
    pass
