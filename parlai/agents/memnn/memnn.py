#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import torch

from parlai.core.torch_ranker_agent import TorchRankerAgent

from .modules import MemNN, opt_to_kwargs


class MemnnAgent(TorchRankerAgent):
    """Memory Network agent.

    Tips:
    - time features are necessary when memory order matters
    - multiple hops allow multiple steps of reasoning, but also seem to make it
        easier to learn to read the memories if you have at least two hops
    - 'adam' seems to work very poorly compared to 'sgd' for hogwild training
    """

    @staticmethod
    def add_cmdline_args(argparser):
        arg_group = argparser.add_argument_group('MemNN Arguments')
        arg_group.add_argument(
            '--init-model', type=str, default=None,
            help='load dict/model/opts from this path')
        arg_group.add_argument(
            '-esz', '--embedding-size', type=int, default=128,
            help='size of token embeddings')
        arg_group.add_argument(
            '-hops', '--hops', type=int, default=3,
            help='number of memory hops')
        arg_group.add_argument(
            '--memsize', type=int, default=32,
            help='size of memory, set to 0 for "nomemnn" model which just '
                 'embeds query and candidates and picks most similar candidate')
        arg_group.add_argument(
            '-tf', '--time-features', type='bool', default=True,
            help='use time features for memory embeddings')
        arg_group.add_argument(
            '-pe', '--position-encoding', type='bool', default=False,
            help='use position encoding instead of bag of words embedding')
        TorchRankerAgent.add_cmdline_args(argparser)
        MemnnAgent.dictionary_class().add_cmdline_args(argparser)
        return arg_group

    @staticmethod
    def model_version():
        """Return current version of this model, counting up from 0.

        Models may not be backwards-compatible with older versions.
        Version 1 split from version 0 on Sep 7, 2018.
        To use version 0, use --model legacy:memnn:0
        (legacy agent code is located in parlai/agents/legacy_agents).
        """
        # TODO: Update date that Version 2 split and move version 1 to legacy
        return 2

    def __init__(self, opt, shared=None):
        # all instances may need some params
        super().__init__(opt, shared)

        self.id = 'MemNN'
        self.memsize = opt['memsize']
        if self.memsize < 0:
            self.memsize = 0
        self.use_time_features = opt['time_features']

        if not shared:
            if opt['time_features']:
                for i in range(self.memsize):
                    self.dict[self._time_feature(i)] = 100000000 + i

    def build_model(self):
        """Build MemNN model."""
        kwargs = opt_to_kwargs(self.opt)
        self.model = MemNN(len(self.dict), self.opt['embedding_size'],
                           padding_idx=self.NULL_IDX, **kwargs)

    def score_candidates(self, batch, cand_vecs):
        mems = self._build_mems(batch.memory_vecs)
        scores = self.model(batch.text_vec, mems, cand_vecs)
        return scores

    @lru_cache(maxsize=None)  # bounded by opt['memsize'], cache string concats
    def _time_feature(self, i):
        """Return time feature token at specified index."""
        return '__tf{}__'.format(i)

    def get_dialog_history(self, *args, **kwargs):
        """Override options in get_dialog_history from parent."""
        kwargs['add_p1_after_newln'] = True  # will only happen if -pt True
        return super().get_dialog_history(*args, **kwargs)

    def vectorize(self, *args, **kwargs):
        """Override options in vectorize from parent."""
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        kwargs['split_lines'] = True
        return super().vectorize(*args, **kwargs)

    def _build_mems(self, mems):
        """Build memory tensors.

        During building, will add time features to the memories if enabled.

        :param: list of length batchsize containing inner lists of 1D tensors
                containing the individual memories for each row in the batch.

        :returns: 3d padded tensor of memories (bsz x num_mems x seqlen)
        """
        if mems is None:
            return None
        bsz = len(mems)
        if bsz == 0:
            return None

        num_mems = max(len(mem) for mem in mems)
        if num_mems == 0 or self.memsize <= 0:
            return None
        elif num_mems > self.memsize:
            # truncate to memsize most recent memories
            num_mems = self.memsize
            mems = [mem[-self.memsize:] for mem in mems]

        try:
            seqlen = max(len(m) for mem in mems for m in mem)
            if self.use_time_features:
                seqlen += 1  # add time token to each sequence
        except ValueError:
            return None

        padded = torch.LongTensor(bsz, num_mems, seqlen).fill_(0)

        for i, mem in enumerate(mems):
            # tf_offset = len(mem) - 1
            for j, m in enumerate(mem):
                padded[i, j, :len(m)] = m
                # if self.use_time_features:
                #     padded[i, j, -1] = self.dict[self._time_feature(tf_offset - j)]

        # NOTE: currently below we are adding tf's to every memory,
        # including emtpy ones. above commented-out code adds only to filled
        # ones but is significantly slower to run.
        if self.use_time_features:
            nm = num_mems - 1
            for i in range(num_mems):
                # put lowest time feature in most recent memory
                padded[:, nm - i, -1] = self.dict[self._time_feature(i)]

        if self.use_cuda:
            padded = padded.cuda()

        return padded
