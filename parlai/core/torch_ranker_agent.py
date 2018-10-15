#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import os
from functools import lru_cache

import torch
from torch import nn

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.thread_utils import SharedTable
from parlai.core.utils import round_sigfigs, padded_tensor

class TorchRankerAgent(TorchAgent):
    @staticmethod
    def add_cmdline_args(argparser):
        TorchAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('TorchRankerAgent')
        agent.add_argument(
            '-tc', '--train_candidates', type=str, default='batch',
            choices=['batch', 'inline', 'fixed', 'dict'],
            help='the source of candidates during training')
        agent.add_argument(
            '-ec', '--eval_candidates', type=str, default='batch',
            choices=['batch', 'inline', 'fixed', 'dict'],
            help='the source of candidates during training')
        agent.add_argument(
            '-cands', '--fixed-candidates-path', type=str,
            help='a text file of fixed candidates to use for all examples, '
                'one candidate per line')
    
    def __init__(self, opt, shared):
        if shared:
            super().__init__(opt, shared)
            self.model = shared['model']
            self.metrics = shared['metrics']
        else:
            # Must call _get_model_file first so that paths are updated if
            # necessary (e.g., .dict file)
            model_file = self._get_model_file(opt)
            self.metrics = {'loss': 0.0, 'examples': 0, 'rank': 0}
            opt['rank_candidates'] = True
            
            super().__init__(opt, shared)

            print(f'Building model of type {self.id}')
            self.build_model()
            if model_file:
                print('Loading existing model parameters from ' + model_file)
                self.load(model_file)

        self.rank_loss = nn.CrossEntropyLoss(reduction='sum')
        self._set_fixed_candidates(shared)

        if self.use_cuda:
            self.model.cuda()
            self.rank_loss.cuda()

        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        self._init_optim(optim_params)

    def score_candidates(self, batch, cand_vecs):
        """Given a batch and candidate set, return scores (for ranking)"""
        raise NotImplementedError(
            'Abstract class: user must implement score()')

    def build_model(self):
        """Build a new model (implemented by children classes)"""
        raise NotImplementedError(
            'Abstract class: user must implement build_model()')

    def train_step(self, batch):
        """Train on a single batch of examples."""
        if batch.text_vec is None:
            return
        batchsize = batch.text_vec.size(0)  
        self.model.train()
        self.optimizer.zero_grad()

        cands, cand_vecs, label_inds = self._build_candidates(batch, 
            source=self.opt['train_candidates'], mode='train')
        scores = self.score_candidates(batch, cand_vecs)
        loss = self.rank_loss(scores, label_inds)

        self.metrics['loss'] += loss.item()
        self.metrics['examples'] += batchsize
        _, ranks = scores.sort(1, descending=True)
        for b in range(batchsize):
            rank = (ranks[b] == label_inds[b]).nonzero().item()
            self.metrics['rank'] += 1 + rank

        loss.backward()
        self.update_params()

        # get predictions but not full rankings--too slow to get hits@1 score
        preds = [cands[row[0]] for row in ranks]

        return Output(preds)

    @torch.no_grad()
    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        batchsize = batch.text_vec.size(0)
        self.model.eval()

        cands, cand_vecs, label_inds = self._build_candidates(batch, 
            source=self.opt['eval_candidates'], mode='eval')

        scores = self.score_candidates(batch, cand_vecs)
        _, ranks = scores.sort(1, descending=True)

        if label_inds is not None:
            loss = self.rank_loss(scores, label_inds)
            self.metrics['loss'] += loss.item()
            self.metrics['examples'] += batchsize
            for b in range(batchsize):
                rank = (ranks[b] == label_inds[b]).nonzero().item()
                self.metrics['rank'] += 1 + rank

        cand_preds = [[cands[i] for i in row] for row in ranks]
        preds = [row[0] for row in cand_preds]
        return Output(preds, cand_preds)

    def _build_candidates(self, batch, source, mode):
        """Build a candidate set for this batch

        :param batch: a Batch object (defined in torch_agent.py)
        :param source: the source from which candidates should be built, one of
            ['batch', 'inline', 'fixed', 'vocab']
        :param mode: 'train' or 'eval'

        :return: tuple of tensors (label_inds, cands, cand_vecs)
            label_inds: A [bsz] LongTensor of the indices of the labelf or each
                example in the tensor of candidates
            cands: A [num_cands] list of (text) candidates 
            cand_vecs: A padded [num_cands, seqlen] LongTensor of vectorized 
                candidates

        Possible sources of candidates:
            * batch: the set of all labels in this batch
                Use all labels in the batch as the candidate set (with all but
                the example's label being treated as negatives). If labels
                are single tokens, filter to unique labels. (Ideally, we would
                always filter to unique labels, but we abstain for the sake of
                speed.)
            * inline: batch_size lists, one list per example
                If each example comes with a list of possible candidates, use
                those.
            * fixed: one global candidate list, provide in a file from the user
                If self.fixed_candidates is not None, use a set of fixed 
                candidates for all examples.
                Note: this setting is not recommended for training unless the
                universe of possible candidates is very small.
            * vocab: the set of all tokens in the vocabulary
        
        TODO: Consider allowing examples in a batch to have different candidate
        sets (requires changes to MemNN module). Currently, all examples in a 
        batch must have the same candidate set.
        """
        label_vecs = batch.label_vec # [bsz] list of lists of LongTensors
        batchsize = batch.text_vec.shape[0]

        if label_vecs is not None:
            assert label_vecs.dim() == 2
        label_inds = None

        if source == 'batch':
            self._warn_once(
                f'{mode}_batch_candidates',
                f'[ Executing {mode} mode with batch labels as set of '
                    'candidates. ]')
            if batchsize == 1:
                self._warn_once(
                    f'{mode}_batchsize_1',
                    f"[ Warning: using candidate source 'batch' and observed a "
                    "batch of size 1. This may be due to uneven batch sizes at "
                    "the end of an epoch. ]")
            if label_vecs is None:
                raise ValueError("If using candidate source 'batch', then "
                    "batch.label_vec can not be None.")

            # Labels are single tokens, enforce uniqueness
            if label_vecs.size(1) == 1:
                cand_vecs, label_inds = label_vecs.unique(return_inverse=True)
                cands = [batch.labels[ind] for ind in label_inds]
            # To save computation, treat all label_vecs as unique
            else:
                cands = batch.labels
                cand_vecs = label_vecs
                label_inds = label_vecs.new(range(batchsize))
                
        elif source == 'inline':
            # WARNING: this is currently the slowest of the options
            self._warn_once(
                f'{mode}_batch_candidates',
                f'[ Executing {mode} mode with provided inline set of candidates '
                    '(assumed to be identical for all candidates in a batch). ]')
            if batch.candidate_vecs is None:
                raise ValueError("If using candidate source 'inline', then "
                    "bath.candidate_vecs can not be None.")
            if batchsize > 1 and (
                torch.any(batch.candidate_vecs[0] != batch.candidate_vecs[1])):
                raise ValueError("All examples in a batch must have the same "
                    "candidate set.")

            cands = batch.candidates[0]
            cand_vecs, _ = padded_tensor(batch.candidate_vecs[0], 
                use_cuda=self.use_cuda)
            if label_vecs is not None:
                label_inds = label_vecs.new_empty(batchsize)
                # Warning: This computation is O(batchsize x num_candidates x seqlen)
                for i, label_vec in enumerate(label_vecs):
                    # Option 1: matrix multiplies
                    label_vec_pad = label_vec.new_zeros(cand_vecs.size(1))
                    label_vec_pad[0:label_vec.size(0)] = label_vec
                    label_inds[i] = (cand_vecs == label_vec_pad).all(1).nonzero()[0]
                    # Option 2: nested for loop
                    # for j, cand_vec in enumerate(cand_vecs):
                    #     if (label_vec == cand_vec[:label_vec.size(0)]).all():
                    #         if ((label_vec.shape == cand_vec.shape) or (cand_vec[label_vec.size(0)] == 0)):
                    #             label_inds[i] = j
                    #             break

        elif source == 'fixed':
            self._warn_once(
                f'{mode}_batch_candidates',
                    f'[ Executing {mode} mode with a common set of fixed '
                        'candidates. ]')
            if self.fixed_candidates is None:
                raise ValueError("If using candidate source 'fixed', then "
                    "you must provide the path to a file of candidates with "
                    "the flag --fixed-candidates-path")

            cands = self.fixed_candidates
            cand_vecs = self.fixed_candidates_vec
            if label_vecs is not None:
                label_inds = label_vecs.new_empty((batchsize))
                # Warning: This computation is O(batchsize x num_candidates)
                for i, label in enumerate(label_vecs):
                    label_inds[i] = (cand_vecs == label).all(1).nonzero()

        elif source == 'vocab':
            raise NotImplementedError

        return (cands, cand_vecs, label_inds)

    def share(self):
        """Share model parameters."""
        shared = super().share()
        shared['model'] = self.model
        if self.opt.get('numthreads', 1) > 1 and isinstance(self.metrics, dict):
            torch.set_num_threads(1)
            # move metrics and model to shared memory
            self.metrics = SharedTable(self.metrics)
            self.model.share_memory()
        shared['metrics'] = self.metrics

        # TODO: Consider moving shared candidates to shared memory?
        shared['fixed_candidates'] = self.fixed_candidates
        shared['fixed_candidates_vec'] = self.fixed_candidates_vec
        return shared

    def update_params(self):
        """Do optim step and clip gradients if needed."""
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

    def reset_metrics(self):
        """Reset metrics."""
        super().reset_metrics()
        self.metrics['loss'] = 0.0
        self.metrics['examples'] = 0
        self.metrics['rank'] = 0

    def report(self):
        """Report loss and mean_rank from model's perspective."""
        m = {}
        examples = self.metrics['examples']
        if examples > 0:
            m['examples'] = examples
            if self.metrics['loss'] > 0:
                m['loss'] = self.metrics['loss']
                m['mean_loss'] = self.metrics['loss'] / examples
            if self.metrics['rank'] > 0:
                m['mean_rank'] = self.metrics['rank'] / examples
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

    def vectorize(self, *args, **kwargs):
        """Override options in vectorize from parent."""
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        kwargs['split_lines'] = True
        return super().vectorize(*args, **kwargs)

    def _get_model_file(self, opt):
        model_file = None

        # first check load path in case we need to override paths
        if opt.get('init_model') and os.path.isfile(opt['init_model']):
            # check first for 'init_model' for loading model from file
            model_file = opt['init_model']

        if opt.get('model_file') and os.path.isfile(opt['model_file']):
            # next check for 'model_file', this would override init_model
            model_file = opt['model_file']

        if model_file is not None:
            # if we are loading a model, should load its dict too
            if (os.path.isfile(model_file + '.dict') or
                    opt['dict_file'] is None):
                opt['dict_file'] = model_file + '.dict'

        return model_file  

    def _set_fixed_candidates(self, shared):
        if shared:
            self.fixed_candidates = shared['fixed_candidates']
            self.fixed_candidates_vec = shared['fixed_candidates_vec']
        else:
            if self.opt['fixed_candidates_path']:
                print(f"Vectorizing fixed candidates set from "
                    f"{self.opt['fixed_candidates_path']}")
                with open(self.opt['fixed_candidates_path'], 'r') as f:
                    self.fixed_candidates = [line.strip() for line in 
                        f.readlines()]
                vectorize_func = self._vectorize_text
                candidate_vecs = [vectorize_func(cand) for cand in 
                    self.fixed_candidates]
                self.fixed_candidates_vec, _ = padded_tensor(candidate_vecs, 
                    use_cuda=self.use_cuda)
            else:
                self.fixed_candidates = None
                self.fixed_candidates_vec = None

    def _warn_once(self, flag, msg):
        if not hasattr(self, flag):
            setattr(self, flag, True)
            print(msg)


