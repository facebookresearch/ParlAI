#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import os
from typing import Any, List

from parlai.utils.io import PathManager
from parlai.core.message import Message
from parlai.core.teachers import DialogTeacher
from .build import build


DEFAULT_TRAIN_EXPERIENCER_ONLY = False


class EmpatheticDialoguesTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        base_datatype = self.datatype.split(':')[0]
        opt['datafile'] = os.path.join(
            opt['datapath'],
            'empatheticdialogues',
            'empatheticdialogues',
            base_datatype + '.csv',
        )
        self.id = 'empathetic_dialogues'
        super().__init__(opt, shared)

        self.experiencer_side_only = (
            opt.get('train_experiencer_only', DEFAULT_TRAIN_EXPERIENCER_ONLY)
            and base_datatype == 'train'
        ) or base_datatype != 'train'

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

    def setup_data(self, path):

        print('loading: ' + path)
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
                label = sparts[5].replace("_comma_", ",")
                prompt = sparts[2]
                sit = sparts[3].replace("_comma_", ",")
                if len(sparts) == 9:
                    if sparts[8] != '':
                        inline_label_candidates = [
                            cand.replace("_comma_", ",").replace("_pipe_", "|")
                            for cand in sparts[8].split('|')
                        ]
                    else:
                        inline_label_candidates = []
                elif len(sparts) == 8:
                    inline_label_candidates = []
                else:
                    raise ValueError(f'Line {i:d} has the wrong number of fields!')

                context_emb, cand_emb = None, None  # Deprecated fields
                ft_ctx, ft_cand = None, None  # Deprecated fields
                is_political = None  # Deprecated field

                dialogue_parts = [
                    contextt,
                    label,
                    prompt,
                    sit,
                    context_emb,
                    cand_emb,
                    ft_ctx,
                    ft_cand,
                    inline_label_candidates,
                    is_political,
                ]

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
                episode_done = entry_idx == len(episode) - 1
                action = Message(
                    {
                        'situation': entry[3],
                        'emotion': entry[2],
                        'text': entry[0],
                        'labels': [entry[1]],
                        'prepend_ctx': entry[6],  # Deprecated
                        'prepend_cand': entry[7],  # Deprecated
                        'deepmoji_ctx': entry[4],  # Deprecated
                        'deepmoji_cand': entry[5],  # Deprecated
                        'episode_done': episode_done,
                        'label_candidates': entry[8],
                    }
                )
                yield action, episode_done

    def _select_dialogues_to_add(
        self,
        experiencer_text_dialogue: List[List[Any]],
        responder_text_dialogue: List[List[Any]],
    ) -> List[List[List[Any]]]:
        """
        Return conversation halves to add to self.data.

        Given lists corresponding to the conversation turns from both sides of the
        conversation, return only the list(s) that will be used by the teacher.
        """
        selected_dialogues = []
        if len(experiencer_text_dialogue) > 0:
            selected_dialogues.append(experiencer_text_dialogue)
        if len(responder_text_dialogue) > 0 and not self.experiencer_side_only:
            selected_dialogues.append(responder_text_dialogue)
        return selected_dialogues


class DefaultTeacher(EmpatheticDialoguesTeacher):
    pass
