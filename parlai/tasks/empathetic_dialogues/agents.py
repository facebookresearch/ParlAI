#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from parlai.core.teachers import FixedDialogTeacher
from .build import build
import numpy as np


class EmpatheticDialogueTeacher(FixedDialogTeacher):

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        if shared:
            self.data = shared['data']
        else:
            build(opt)
            fold = opt.get('datatype', 'train').split(':')[0]
            self._setup_data(fold)

        self.num_exs = sum([(len(d)+1)//2 for d in self.data])
        self.num_eps = len(self.data)
        self.reset()

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _setup_data(self, fold):
        self.turns = 0
        if self.opt.get('deepmoji') is not None:
            self.embed = np.load(self.opt['deepmoji'] + fold + ".npy")

        if (
            self.opt.get('fasttextloc') is not None and
            self.opt.get('prepend', -1) > 0
        ):
            try:
                import fastText
            except ImportError:
                raise ImportError("Please run 'pip install fasttext'.")
            ftpath = self.opt['fasttextloc']
            ftmodel = fastText.FastText.load_model(ftpath)

        fpath = os.path.join(
            self.opt['datapath'], 'empatheticdialogues', 'empatheticdialogues',
            fold + '.csv',
        )
        df = open(fpath).readlines()

        self.data = []
        dialog = []
        for i in range(1, len(df)):

            cparts = df[i-1].strip().split(",")
            sparts = df[i].strip().split(",")

            if cparts[0] == sparts[0]:

                contextt = cparts[5].replace("_comma_", ",")
                label = sparts[5].replace("_comma_", ",")
                prompt = sparts[2]
                sit = sparts[3].replace("_comma_", ",")
                if len(sparts) == 9:
                    inline_label_candidates = [
                        cand.replace("_comma_", ",").replace("_pipe_", "|")
                        for cand in sparts[8].split('|')
                    ]
                elif len(sparts) == 8:
                    inline_label_candidates = []
                else:
                    raise ValueError(
                        f'Line {i:d} has the wrong number of fields!'
                    )

                context_emb, cand_emb = None, None
                if self.opt.get('deepmoji') is not None:
                    context_emb = self.embed[i-2]
                    cand_emb = self.embed[i-1]

                ft_ctx, ft_cand = None, None
                if (
                    self.opt.get('fasttextloc') is not None and
                    self.opt.get('prepend', -1) > 0
                ):
                    ft_ctx = ""
                    gettop, _ = ftmodel.predict(contextt, k=self.opt['prepend'])
                    for f in gettop:
                        ft_ctx = f.split("_")[-1] + " " + ft_ctx
                    ft_cand = ""
                    gettop, _ = ftmodel.predict(label, k=self.opt['prepend'])
                    for f in gettop:
                        ft_cand = f.split("_")[-1] + " " + ft_cand

                dialog.append((
                    contextt, label, prompt, sit, context_emb, cand_emb, ft_ctx,
                    ft_cand, inline_label_candidates,
                ))

            else:

                if len(dialog) > 0:
                    self.data.append(dialog)
                dialog = []

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        i = entry_idx*2
        ep_i = ep[i]
        episode_done = (i >= (len(ep)-2))
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


class DefaultTeacher(EmpatheticDialogueTeacher):
    pass
