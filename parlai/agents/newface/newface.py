#!/usr/bin/env python3

# Copyright (c) 2019-present, Shaojie Jiang.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# This programme is modified on top of the Seq2Seq implementation of Facebook Inc.,
# please visit http://parl.ai/ for more details.
#
# Should you have any problems using this programme, please contact Shaojie Jiang
# via shaojiejiang.1991@gmail.com

from parlai.core.torch_generator_agent import TorchGeneratorAgent
#from .modules import Seq2seq, opt_to_kwargs, HLoss
from parlai.core.utils import NEAR_INF, padded_tensor, round_sigfigs, warn_once
from parlai.core.torch_agent import Output
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent

import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import math
import numpy as np
from collections import Counter


class NewfaceAgent(Seq2seqAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('NewFace Arguments')
        agent.add_argument('-ft', '--frequency-type', default='out',
                           choices=['out', 'gt', 'none'],
                           help='What to use for calculating token frequency.')
        agent.add_argument('-wt', '--weighing-time', default='pre',
                           choices=['pre', 'post', 'none'],
                           help='When to apply weight to losses.')
        agent.add_argument('-cp', '--confidence-penalty', default='none',
                           choices=['cp', 'cpf', 'cpfw', 'cpfwn', 'none'],
                           help='Which kind of confidence penalty to use: '
                                "'cp' is the confidence-penalty function reported in https://arxiv.org/abs/1809.01941. "
                                "'cpf' is the parameter-free version proposed in https://arxiv.org/abs/1902.09191. "
                                "'cpfw' means using the parameter-free version as the weight of FACE. "
                                "'cpfwn' is a new design that normalizes the weight to the range of [1, +inf], which is "
                                "more favorable as the weight of FACE.")
        agent.add_argument('-b', '--beta', type=float, default=2.5,
                           help='Penalty strength for type "cp".')

        super(NewfaceAgent, cls).add_cmdline_args(argparser)
        NewfaceAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        return 2

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'NEWFACE'
        if getattr(self, 'word_freq', None) is None:
            self.word_freq = np.zeros(len(self.dict))
        self.ft = opt['frequency_type']
        self.wt = opt['weighing_time']
        self.cp = opt['confidence_penalty']
        self.beta = opt['beta']
        self.masked_entropy = HLoss(ignore_index=self.NULL_IDX)
        self.ideal_entropy = math.log(1 / len(self.dict))

    def weighted_loss(self):
        return

    def compute_loss(self, batch, return_output=False):
        """
        Computes and returns the loss for the given batch. Easily overridable for
        customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        scores, preds, _ = self.model(batch.text_vec, ys=batch.label_vec)
        score_view = scores.view(-1, scores.size(-1))
        preds_clean = self.clean_preds(preds)

        if self.ft == 'gt':
            self.update_frequency(self.clean_preds(batch.label_vec))
        elif self.ft == 'out':
            self.update_frequency(preds_clean)

        # calculate loss w/ or w/o pre-/post-weight
        if self.wt == 'pre':
            self.criterion.weight = self.loss_weight()
            loss = self.criterion(score_view, batch.label_vec.view(-1))
        elif self.wt == 'post':
            self.criterion.reduction = 'none'
            loss = self.criterion(score_view, batch.label_vec.view(-1))
            device = loss.device
            freq_pred = self.word_freq[preds.view(-1).cpu().numpy()]
            freq_pred = torch.FloatTensor(freq_pred).to(device)
            freq_GT = self.word_freq[batch.label_vec.view(-1).cpu().numpy()]
            freq_GT = torch.FloatTensor(freq_GT).to(device)
            total_freq = self.word_freq.sum()
            weight = 1 + F.relu(freq_pred - freq_GT) / total_freq
            loss = torch.matmul(loss, weight)
        else:
            loss = self.criterion(score_view, batch.label_vec.view(-1))

        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        #
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum().item()
        if self.cp != 'none':
            entropy = self.masked_entropy(score_view, batch.label_vec.view(-1))
            mean_entropy = entropy / target_tokens
            if self.cp == 'cp':
                loss -= self.beta * mean_entropy
            elif self.cp == 'cpf':
                loss += 1 / mean_entropy
            elif self.cp == 'cpfw':
                # TODO: normalize weight to [1, ++]?
                loss *= (1 + 1 / mean_entropy)
            elif self.cp == 'cpfwn':
                loss *= (self.ideal_entropy / mean_entropy)
        correct = ((batch.label_vec == preds) * notnull).sum().item()
        self.metrics['correct_tokens'] += correct
        self.metrics['nll_loss'] += loss.item()
        self.metrics['num_tokens'] += target_tokens
        self.metrics['preds'].extend(preds_clean)

        loss /= target_tokens  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        bsz = batch.text_vec.size(0)
        self.model.eval()
        cand_scores = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss = self.compute_loss(batch)  # noqa: F841  we need the side effects
            self.metrics['loss'] += loss.item()

        preds = None
        if self.skip_generation:
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning
            )
        elif self.beam_size == 1:
            # greedy decode
            _, preds, *_ = self.model(*self._model_input(batch), bsz=bsz)
        elif self.beam_size > 1:
            out = self.beam_search(
                self.model,
                batch,
                self.beam_size,
                start=self.START_IDX,
                end=self.END_IDX,
                pad=self.NULL_IDX,
                min_length=self.beam_min_length,
                min_n_best=self.beam_min_n_best,
                block_ngram=self.beam_block_ngram
            )
            beam_preds_scores, _, beams = out
            preds, scores = zip(*beam_preds_scores)

            if self.beam_dot_log is True:
                self._write_beam_dots(batch.text_vec, beams)

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            f_scores, f_preds, _ = self.model(batch.text_vec, ys=batch.label_vec)
            score_view = f_scores.view(-1, f_scores.size(-1))
            self.criterion.reduction = 'sum'
            loss = self.criterion(score_view, batch.label_vec.view(-1))
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == f_preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens

        cand_choices = None
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(*self._model_input(batch))
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = padded_tensor(
                    batch.candidate_vecs[i], self.NULL_IDX, self.use_cuda
                )
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        text = [self._v2t(p) for p in preds] if preds is not None else None
        self.metrics['preds'].extend(self.clean_preds(preds))
        return Output(text, cand_choices)

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            if hasattr(self.model, 'module'):
                model['model'] = self.model.module.state_dict()
                model['longest_label'] = self.model.module.longest_label
            else:
                model['model'] = self.model.state_dict()
                model['longest_label'] = self.model.longest_label
            model['optimizer'] = self.optimizer.state_dict()
            model['optimizer_type'] = self.opt['optimizer']
            model['word_freq'] = self.word_freq

            with open(path, 'wb') as write:
                torch.save(model, write)

            # save opt file
            with open(path + '.opt', 'w') as handle:
                # save version string
                self.opt['model_version'] = self.model_version()
                json.dump(self.opt, handle)

    def load(self, path):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        self.word_freq = states['word_freq']
        # set loaded states if applicable
        self.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.model.longest_label = states['longest_label']
        return states

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        super().reset_metrics()
        self.metrics['loss'] = 0.0
        self.metrics['num_tokens'] = 0
        self.metrics['correct_tokens'] = 0
        self.metrics['preds'] = []

    def clean_preds(self, preds):
        res = []
        # OAD:
        #         preds = preds.cpu().tolist()
        if type(preds) == tuple:
            preds = [p.cpu().tolist() for p in preds]
        else:  # should be tensor:
            preds = preds.cpu().tolist()

        for pred in preds:
            if self.END_IDX in pred:
                ind = pred.index(self.END_IDX) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == self.START_IDX:
                pred = pred[1:]
            res.append(pred)
        return res

    def calc_diversity(self, metrics):
        unigram = set()
        bigram = set()
        num_tok = 0
        for vec in self.metrics['preds']:
            v_len = len(vec)
            num_tok += v_len
            unigram.update(vec)
            bigram.update([tuple(vec[i:i + 2]) for i in range(v_len - 1)])
        metrics['d_1'] = round(len(unigram) / num_tok * 100, 2)
        metrics['d_2'] = round(len(bigram) / num_tok * 100, 2)
        if not self.model.training:
            metrics['num_d1'] = len(unigram)
            metrics['num_d2'] = len(bigram)
            metrics['num_tok'] = num_tok

    def report(self):
        m = {}
        num_tok = self.metrics['num_tokens']
        if num_tok > 0:
            if self.metrics['correct_tokens'] > 0:
                m['token_acc'] = self.metrics['correct_tokens'] / num_tok
            m['loss'] = self.metrics['loss'] / num_tok
            try:
                m['ppl'] = math.exp(m['loss'])
            except OverflowError:
                m['ppl'] = float('inf')
        if self.metrics['total_skipped_batches'] > 0:
            m['total_skipped_batches'] = self.metrics['total_skipped_batches']
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        if self.metrics['preds']:
            self.calc_diversity(m)
        return m

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)

        # self.word_freq *= self.opt['decay_factor']
        for k, v in curr.items():
            self.word_freq[k] += v

    def loss_weight(self):
        RF = self.word_freq / self.word_freq.sum()  # relative frequency
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)  # normalization
        if self.use_cuda:
            return torch.FloatTensor(weight).cuda()
        else:
            return torch.FloatTensor(weight)

class HLoss(nn.Module):

    def __init__(self, ignore_index=-1):
        super(HLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, x, labels):
        mask = (labels != self.ignore_index).float()
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * torch.matmul(mask, b.sum(dim=1))
        return b

