#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from parlai.core.teachers import FixedDialogTeacher
from .build import build
import numpy as np


DEFAULT_TRAIN_EXPERIENCER_ONLY = False


class EmpatheticDialoguesTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        self.datatype = opt.get('datatype', 'train').split(':')[0]
        self.datapath = os.path.join(
            self.opt['datapath'],
            'empatheticdialogues',
            'empatheticdialogues',
            self.datatype + '.csv',
        )
        self.experiencer_side_only = (
            opt.get('train_experiencer_only', DEFAULT_TRAIN_EXPERIENCER_ONLY)
            and self.datatype == 'train'
        ) or self.datatype != 'train'
        print(
            f'[EmpatheticDialoguesTeacher] Only use experiencer side? '
            f'{self.experiencer_side_only}, datatype: {self.datatype}'
        )

        if shared:
            self.data = shared['data']
        else:
            build(opt)
            self._setup_data(self.datatype)

        self.num_exs = sum([len(d) for d in self.data])
        self.num_eps = len(self.data)
        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('EmpatheticDialogues teacher arguments')
        agent.add_argument(
            '--train-experiencer-only',
            type='bool',
            default=DEFAULT_TRAIN_EXPERIENCER_ONLY,
            # i.e. do not include the other side of the conversation where the Listener
            # (responder) utterance would be the text and the Speaker (experiencer)
            # utterance would be the label
            help='In the train set, only use Speaker (experiencer) utterances as text and Listener (responder) utterances as labels.',
        )

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _setup_data(self, datatype):

        if self.opt.get('deepmoji') is not None:
            self.embed = np.load(self.opt['deepmoji'] + datatype + ".npy")

        if self.opt.get('fasttextloc') is not None and self.opt.get('prepend', -1) > 0:
            try:
                import fastText
            except ImportError:
                raise ImportError("Please run 'pip install fasttext'.")
            ftpath = self.opt['fasttextloc']
            ftmodel = fastText.FastText.load_model(ftpath)

        df = open(self.datapath).readlines()

        turn_idx = 1
        responder_text_dialogue = []
        experiencer_text_dialogue = []
        self.data = []
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

                context_emb, cand_emb = None, None
                if self.opt.get('deepmoji') is not None:
                    context_emb = self.embed[i - 2]
                    cand_emb = self.embed[i - 1]

                ft_ctx, ft_cand = None, None
                if (
                    self.opt.get('fasttextloc') is not None
                    and self.opt.get('prepend', -1) > 0
                ):
                    ft_ctx = ""
                    gettop, _ = ftmodel.predict(contextt, k=self.opt['prepend'])
                    for f in gettop:
                        ft_ctx = f.split("_")[-1] + " " + ft_ctx
                    ft_cand = ""
                    gettop, _ = ftmodel.predict(label, k=self.opt['prepend'])
                    for f in gettop:
                        ft_cand = f.split("_")[-1] + " " + ft_cand

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
                if len(experiencer_text_dialogue) > 0:
                    self.data.append(experiencer_text_dialogue)
                if len(responder_text_dialogue) > 0 and not self.experiencer_side_only:
                    self.data.append(responder_text_dialogue)
                experiencer_text_dialogue = []
                responder_text_dialogue = []

        # Add in the final episode
        if len(experiencer_text_dialogue) > 0:
            self.data.append(experiencer_text_dialogue)
        if len(responder_text_dialogue) > 0 and not self.experiencer_side_only:
            self.data.append(responder_text_dialogue)

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        ep_i = ep[entry_idx]
        episode_done = entry_idx >= (len(ep) - 1)
        action = {
            'situation': ep_i[3],
            'emotion': ep_i[2],
            'text': ep_i[0],
            'labels': [ep_i[1]],
            'prepend_ctx': ep_i[6],
            'prepend_cand': ep_i[7],
            'deepmoji_ctx': ep_i[4],
            'deepmoji_cand': ep_i[5],
            'episode_done': episode_done,
            'label_candidates': ep_i[8],
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class EmotionClassificationSituationTeacher(EmpatheticDialoguesTeacher):
    """
    Class for detecting the emotion based on the situation.
    """

    def __init__(self, opt, shared=None):
        opt['train_experiencer_only'] = True
        # So that we only have one episode per train conversation
        super().__init__(opt, shared)
        if not shared:
            self._get_situations()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def _get_situations(self):
        new_data = []
        for ep in self.data:
            new_data.append(ep[0])
        self.data = new_data

    def get(self, episode_idx, entry_idx=0):
        ex = self.data[episode_idx]
        episode_done = True

        return {'labels': [ex[2]], 'text': ex[3], 'episode_done': episode_done}


class DefaultTeacher(EmpatheticDialoguesTeacher):
    pass
