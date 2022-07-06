#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Agents for handling the generation aspects of Wizard.
"""
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from itertools import chain
from functools import lru_cache

import torch as th
import numpy as np

from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs

from parlai.agents.transformer.transformer import TransformerGeneratorAgent

from .modules import EndToEndModel
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE

TOKEN_DIALOG = '__dialog__'


DEFAULT_OPTS = {
    "learningrate": 5e-4,
    "optimizer": "adam",
    "lr_scheduler": "invsqrt",
    "warmup_updates": 5000,
    "clip_norm": 0.1,
    "ffn_size": 512,
    "embedding_size": 256,
    "n_heads": 2,
    "dropout": 0.2,
    "n_layers": 5,
    "betas": "0.9,0.98",
    "truncate": 128,
    "add_token_knowledge": True,
    "dict_textfields": "text,labels,chosen_topic,checked_sentence,knowledge,title",
}


class _GenericWizardAgent(TransformerGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.set_defaults(**DEFAULT_OPTS)
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return parser

    def batchify(self, obs_batch):
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]

        checked_sentences = []
        for obs in reordered_observations:
            checked_sentence = '{} {} {}'.format(
                obs.get('title', ''), TOKEN_KNOWLEDGE, obs.get('checked_sentence', '')
            )
            checked_sentences.append(checked_sentence)

        batch['checked_sentence'] = checked_sentences

        return batch


class TwoStageAgent(_GenericWizardAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if shared is not None:
            # make sure the dialogue token appears
            self.dict[TOKEN_DIALOG] = 9999999

    def _set_text_vec(self, obs, history, truncate):
        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            fields = []
            dialogue_history = history.get_history_str()
            if 'chosen_topic' in obs:
                fields += [obs['title']]
            if 'checked_sentence' in obs:
                fields += [TOKEN_KNOWLEDGE, obs['checked_sentence']]
            if dialogue_history:
                fields += [TOKEN_DIALOG, dialogue_history]
            obs['text'] = ' '.join(fields)
            obs['text_vec'] = self.dict.txt2vec(obs['text'])

        # check truncation
        if 'text_vec' in obs:
            obs['text_vec'] = th.LongTensor(
                self._check_truncate(obs['text_vec'], truncate, True)
            )

        return obs


class EndToEndAgent(_GenericWizardAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self._vectorize_text = lru_cache(int(2**20))(self._vectorize_text)

        # knowledge truncate defaults to the same as --truncate
        self.knowledge_truncate = opt.get('knowledge_truncate')
        if not self.knowledge_truncate:
            self.knowledge_truncate = opt['truncate']
        self.max_knowledge = opt.get('max_knowledge')
        self.knowledge_alpha = opt['knowledge_alpha']

    def compute_loss(self, batch, return_output=False):
        # first compute our regular forced decoding loss
        token_loss, model_output = super().compute_loss(batch, return_output=True)
        notnull = batch.label_vec.ne(self.NULL_IDX)
        num_tokens = notnull.long().sum().item()

        encoder_states = model_output[2]
        ctx_know_attn = encoder_states[2]

        if self.knowledge_alpha == 0.0:
            loss = token_loss
        else:
            _, know_pred = ctx_know_attn.max(1)
            know_acc = (know_pred == batch.cs_ids).float().sum().item()
            know_chance = batch.ck_mask.sum(1).float().reciprocal().sum().item()
            self.metrics['know_chance'] += know_chance
            self.metrics['bsz'] += batch.text_vec.size(0)
            self.metrics['know_acc'] += know_acc
            know_loss = th.nn.functional.cross_entropy(
                ctx_know_attn, batch.cs_ids, reduction='mean'
            )
            self.metrics['know_loss'] += know_loss.item() * batch.text_vec.size(0)
            # in the original paper the loss was scaled by num_tokens for both
            # know_loss and token_loss
            know_loss /= num_tokens
            loss = (
                1 - self.knowledge_alpha
            ) * token_loss + self.knowledge_alpha * know_loss
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['bsz'] = 0.0
        self.metrics['know_acc'] = 0.0
        self.metrics['know_loss'] = 0.0
        self.metrics['know_chance'] = 0.0

    def report(self):
        r = super().report()
        bsz = max(self.metrics['bsz'], 1)
        for k in ['know_loss', 'know_acc', 'know_chance']:
            # round and average across all items since last report
            r[k] = round_sigfigs(self.metrics[k] / bsz, 4)
        return r

    def _parse_knowledge(self, obs):
        if 'knowledge_parsed' in obs:
            # make a copy of the list to prevent the future padding step from
            # being destructive
            return list(obs['knowledge_parsed'])

        if 'checked_sentence' not in obs:
            # interactive time. we're totally on our own
            obs_know = [
                k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
            ]
            obs_know = [k for k in obs_know if k]
            obs['knowledge_parsed'] = obs_know
            return obs['knowledge_parsed']

        checked_sentence = '{} {} {}'.format(
            obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence']
        )
        # grab all the nonempty knowledge
        obs_know = [
            k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
        ]
        obs_know = [k for k in obs_know if k]

        # we want the correct knowledge to always be in index 0
        try:
            i = obs_know.index(checked_sentence)
        except ValueError:
            # uh oh, couldn't find the sentence in the knowledge. This happens for
            # one or two examples in the training set. We can just artificially
            # put it back in
            i = 0
            obs_know[0] = checked_sentence
        obs_know[0], obs_know[i] = obs_know[i], obs_know[0]

        obs['knowledge_parsed'] = obs_know
        obs['checked_sentence_parsed'] = checked_sentence
        return obs['knowledge_parsed']

    def batchify(self, obs_batch):
        """
        Wizard custom batchify, which passes along the knowledge/title.

        Following the docstring of TorchAgent.batchify, it calls super, then
        uses an extended version of the torch_agent.Batch namedtuple.

        The purpose of extending the info is to keep track of some custom
        metrics.
        """
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]
        is_training = 'labels' in reordered_observations[0]

        # first parse and compile all the knowledge together
        all_knowledges = []  # list-of-lists knowledge items for each observation
        knowledge_counts = []  # how much knowledge each observation gets
        for obs in reordered_observations:
            obs_know = self._parse_knowledge(obs)
            # downsample if desired
            if (
                is_training
                and self.max_knowledge
                and len(obs_know) > self.max_knowledge
            ):
                # offset by one so that we don't choose 0
                keepers = 1 + np.random.choice(
                    len(obs_know) - 1, self.max_knowledge, False
                )
                # correct answer is always the first one
                keepers[0] = 0
                obs_know = [obs_know[i] for i in keepers]
            all_knowledges.append(obs_know)
            knowledge_counts.append(len(obs_know))

        # now we want to actually pack this into a tensor, along with the mask
        N = len(reordered_observations)
        K = max(knowledge_counts)
        # round out the array so everything is equally sized
        for i in range(N):
            all_knowledges[i] += [''] * (K - knowledge_counts[i])
        flattened_knowledge = list(chain(*all_knowledges))

        knowledge_vec = [
            self._vectorize_text(
                # the beginning of the sentence is more useful
                k,
                truncate=self.knowledge_truncate,
                add_end=True,
                truncate_left=False,
            )
            for k in flattened_knowledge
        ]
        knowledge_vec, _ = padded_tensor(
            knowledge_vec, pad_idx=self.NULL_IDX, left_padded=True
        )
        knowledge_vec[:, -1] = self.END_IDX
        T = knowledge_vec.size(-1)
        knowledge_vec = knowledge_vec.view(N, K, T)

        # knowledge mask is a N x K tensor saying which items we're allowed to
        # attend over
        bsz = len(reordered_observations)
        ck_mask = th.zeros(bsz, K, dtype=th.uint8)
        for i, klen in enumerate(knowledge_counts):
            ck_mask[i, :klen] = 1
        ck_mask = ck_mask != 0  # for pytorch 1.0/1.2 uint8/bool compatibility
        # and the correct labels
        cs_ids = th.LongTensor(bsz).zero_()

        if self.use_cuda:
            knowledge_vec = knowledge_vec.cuda()
            ck_mask = ck_mask.cuda()
            cs_ids = cs_ids.cuda()

        batch['know_vec'] = knowledge_vec
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = cs_ids
        batch['use_cs_ids'] = is_training
        batch['knowledge'] = np.array(flattened_knowledge).reshape(N, K)
        return batch

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group("EndToEnd Agent")
        group.add_argument(
            '--knowledge-alpha',
            type=float,
            default=0.95,
            help='Weight on the knowledge-attn loss',
        )
        group.add_argument(
            '--knowledge-truncate',
            type=int,
            default=32,
            help='Knowledge truncation field. Defaults to same as --truncate.',
        )
        group.add_argument(
            '--max-knowledge',
            type=int,
            help='Reduce the amount of negative knowledge at train time.',
        )
        parser.add_argument(
            '--knowledge-alpha',
            type=float,
            default=0.95,
            help='Weight on the knowledge-attn loss',
        )
        return parser

    def _model_input(self, batch):
        return (
            batch.text_vec,
            batch.know_vec,
            batch.ck_mask,
            batch.cs_ids,
            batch.use_cs_ids,
        )

    def build_model(self):
        self.model = EndToEndModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model = self.model.cuda()
        return self.model
