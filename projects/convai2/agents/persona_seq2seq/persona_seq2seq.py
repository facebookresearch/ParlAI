# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from parlai.core.utils import PaddingUtils
from .modules import PersonaSeq2seq

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

from collections import deque

import os
import math

class PersonaSeq2seqAgent(Seq2seqAgent):
    """Same as the Seq2seqAgent, except extracts the personas from the input
    and uses them for attention.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        agent = Seq2seqAgent.add_cmdline_args(argparser)
        agent.add_argument('-pnl', '--persona-numlayers', type=int, default=1,
            help='Number of layers for the persona encoder.')
        agent.add_argument('-pbi', '--persona-bidirectional', type='bool',
            default=True,
            help='Whether to use a bidirectional encoder for the personas.')
        agent.add_argument('-patt', '--persona-attention', default='persona',
            choices=['persona', 'context'],
            help='Whether to use the persona or the context for attention, '
                 'or how to combine them if you want to use both.')
        agent.add_argument('-penc', '--persona-encoding', default='separate',
            choices=['concat', 'separate', 'max', 'maxsum', 'bow'],
            help='How to encode the personas before doing attention over them. '
                 'TODO: elaborate')

    def __init__(self, opt, shared=None):
        """Sets model class to modified seq2seq."""
        self.model_class = PersonaSeq2seq
        super().__init__(opt, shared)

    def reset(self):
        """Also reset personas."""
        super().reset()
        self.persona = []
        self.other_persona = []

    def extract_persona(self, text):
        """Extract `your persona` and `partner\'s persona` from text."""
        # your persona
        personas = []
        ind = text.find('your persona: ')
        while ind >= 0:
            new_line = text.find('\n')
            persona = text[ind:new_line]
            personas.append(persona)
            text = text[new_line + 1:]
            ind = text.find('your persona: ')
        if len(personas) > 0:
            self.persona = personas

        # other persona
        other_personas = []
        ind = text.find('partner\'s persona: ')
        while ind >= 0:
            new_line = text.find('\n')
            other_persona = text[ind:new_line]
            other_personas.append(other_persona)
            text = text[new_line + 1:]
            ind = text.find('partner\'s persona: ')
        if len(other_personas) > 0:
            self.other_persona = other_personas

        return text, self.persona, self.other_persona


    def observe(self, observation):
        """Extracts personas before doing standard observe."""
        obs = observation
        if 'text' in obs:
            obs['text'], obs['persona'], obs['other_persona'] = self.extract_persona(obs['text'])
        return super().observe(obs)


    def predict(self, xs, ys=None, cands=None, valid_cands=None, is_training=False, ps=None):
        """Produce a prediction from our model.

        Copied from parent, but uses personas for attention in addition to
        context.
        """
        text_cand_inds, loss_dict = None, None
        if is_training:
            self.model.train()
            self.zero_grad()
            predictions, scores, _ = self.model(xs, ys, ps=ps)
            loss = self.criterion(scores.view(-1, scores.size(-1)), ys.view(-1))
            # save loss to metrics
            target_tokens = ys.ne(self.NULL_IDX).long().sum().data[0]
            self.metrics['loss'] += loss.double().data[0]
            self.metrics['num_tokens'] += target_tokens
            loss /= target_tokens  # average loss per token
            # loss /= xs.size(0)  # average loss per sentence
            loss.backward()
            self.update_params()
        else:
            self.model.eval()
            predictions, _scores, text_cand_inds = self.model(
                xs, ys=None, cands=cands, valid_cands=valid_cands, ps=ps)

            if ys is not None:
                # calculate loss on targets
                _, scores, _ = self.model(xs, ys, ps=ps)
                loss = self.criterion(scores.view(-1, scores.size(-1)), ys.view(-1))
                target_tokens = ys.ne(self.NULL_IDX).long().sum().data[0]
                self.metrics['loss'] += loss.double().data[0]
                self.metrics['num_tokens'] += target_tokens

        return predictions, text_cand_inds

    def vectorize(self, observations):
        """Same as parent but also contructs personas."""
        xs, ys, labels, valid_inds, cands, valid_cands, is_training = super().vectorize(observations)
        ps = None
        if xs is not None:
            exs = [observations[i] for i in valid_inds]
            ps = []
            for ex in exs:
                ex_ps = []
                for p in ex.get('persona', []) + ex.get('other_persona', []):
                    tensor = torch.LongTensor(self.parse(p))
                    if self.use_cuda:
                        tensor = tensor.cuda()
                    ex_ps.append(Variable(tensor))
                ps.append(ex_ps)
        return xs, ys, labels, valid_inds, cands, valid_cands, is_training, ps

    def batch_act(self, observations):
        """Same as parent but also calls predict with personas."""
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, labels, valid_inds, cands, valid_cands, is_training, ps = self.vectorize(observations)

        if xs is None:
            # no valid examples, just return empty responses
            return batch_reply

        # produce predictions, train on targets if availables
        predictions, text_cand_inds = self.predict(xs, ys, cands, valid_cands, is_training, ps)

        if is_training:
            report_freq = 0
        else:
            report_freq = 0.01
        PaddingUtils.map_predictions(
            predictions.cpu().data, valid_inds, batch_reply, observations,
            self.dict, self.END_IDX, report_freq=report_freq, labels=labels,
            answers=self.answers, ys=ys.data if ys is not None else None)

        if text_cand_inds is not None:
            text_cand_inds = text_cand_inds.cpu().data
            for i in range(len(valid_cands)):
                order = text_cand_inds[i]
                _, batch_idx, curr_cands = valid_cands[i]
                curr = batch_reply[batch_idx]
                curr['text_candidates'] = [curr_cands[idx] for idx in order
                                           if idx < len(curr_cands)]

        return batch_reply
