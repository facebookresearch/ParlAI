#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import os

import torch
from torch import nn

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.thread_utils import SharedTable
from parlai.core.utils import round_sigfigs


class TorchRankerAgent(TorchAgent):
    @staticmethod
    def add_cmdline_args(argparser):
        TorchAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('TorchRankerAgent')
        agent.add_argument(
            '-tc', '--train-candidates', type=str, default='batch',
            choices=['batch', 'inline', 'fixed', 'vocab'],
            help='The source of candidates during training')
        agent.add_argument(
            '-ec', '--eval-candidates', type=str, default='batch',
            choices=['batch', 'inline', 'fixed', 'vocab'],
            help='The source of candidates during training')
        agent.add_argument(
            '-cands', '--fixed-candidates-path', type=str,
            help='A text file of fixed candidates to use for all examples, one '
                 'candidate per line')
        agent.add_argument(
            '-candvecs', '--fixed-candidate-vecs-path', type=str,
            help='A torch file containing vectorized fixed candidates to use for all '
                 'examples. If a file already exists at -fixed-candidate-vecs-path, '
                 'those vectors will be loaded. If not, they will be computed and '
                 'written to that path')

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

            print('Building model of type {}'.format(self.id))
            self.build_model()
            if model_file:
                print('Loading existing model parameters from ' + model_file)
                self.load(model_file)

        self.rank_loss = nn.CrossEntropyLoss(reduce=True, size_average=False)
        self.set_fixed_candidates(shared)

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

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.opt['train_candidates'], mode='train')
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

        cand_preds = []
        for i, ordering in enumerate(ranks):
            if cand_vecs.dim() == 2:
                cand_list = cands
            elif cand_vecs.dim() == 3:
                cand_list = cands[i]
            cand_preds.append([cand_list[rank] for rank in ordering])
        preds = [cand_preds[i][0] for i in range(batchsize)]
        return Output(preds, cand_preds)

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        batchsize = batch.text_vec.size(0)
        self.model.eval()

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.opt['eval_candidates'], mode='eval')

        scores = self.score_candidates(batch, cand_vecs)
        _, ranks = scores.sort(1, descending=True)

        if label_inds is not None:
            loss = self.rank_loss(scores, label_inds)
            self.metrics['loss'] += loss.item()
            self.metrics['examples'] += batchsize
            for b in range(batchsize):
                rank = (ranks[b] == label_inds[b]).nonzero().item()
                self.metrics['rank'] += 1 + rank

        cand_preds = []
        for i, ordering in enumerate(ranks):
            if cand_vecs.dim() == 2:
                cand_list = cands
            elif cand_vecs.dim() == 3:
                cand_list = cands[i]
            cand_preds.append([cand_list[rank] for rank in ordering])
        preds = [cand_preds[i][0] for i in range(batchsize)]
        return Output(preds, cand_preds)

    def _build_candidates(self, batch, source, mode):
        """Build a candidate set for this batch

        :param batch: a Batch object (defined in torch_agent.py)
        :param source: the source from which candidates should be built, one of
            ['batch', 'inline', 'fixed']
        :param mode: 'train' or 'eval'

        :return: tuple of tensors (label_inds, cands, cand_vecs)
            label_inds: A [bsz] LongTensor of the indices of the labels for each
                example from its respective candidate set
            cands: A [num_cands] list of (text) candidates
                OR a [batchsize] list of such lists if source=='inline'
            cand_vecs: A padded [num_cands, seqlen] LongTensor of vectorized candidates
                OR a [batchsize, num_cands, seqlen] LongTensor if source=='inline'

        Possible sources of candidates:
            * batch: the set of all labels in this batch
                Use all labels in the batch as the candidate set (with all but the
                example's label being treated as negatives). If labels are single
                tokens, filter to unique labels. (Ideally, we would always filter to
                unique labels, but we abstain for the sake of speed.)
                Note: with this setting, the candidate set is identical for all
                examplesin a batch.
            * inline: batch_size lists, one list per example
                If each example comes with a list of possible candidates, use those.
                Note: With this setting, each example will have its own candidate set.
            * fixed: one global candidate list, provide in a file from the user
                If self.fixed_candidates is not None, use a set of fixed candidates for
                all examples.
                Note: this setting is not recommended for training unless the
                universe of possible candidates is very small.
        """
        label_vecs = batch.label_vec  # [bsz] list of lists of LongTensors
        label_inds = None
        batchsize = batch.text_vec.shape[0]

        if label_vecs is not None:
            assert label_vecs.dim() == 2

        if source == 'batch':
            self._warn_once(
                flag=(mode + '_batch_candidates'),
                msg=('[ Executing {} mode with batch labels as set of candidates. ]'
                     ''.format(mode)))
            if batchsize == 1:
                self._warn_once(
                    flag=(mode + '_batchsize_1'),
                    msg=("[ Warning: using candidate source 'batch' and observed a "
                         "batch of size 1. This may be due to uneven batch sizes at "
                         "the end of an epoch. ]"))
            if label_vecs is None:
                raise ValueError(
                    "If using candidate source 'batch', then batch.label_vec cannot be "
                    "None.")

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
            self._warn_once(
                flag=(mode + '_batch_candidates'),
                msg=('[ Executing {} mode with provided inline set of candidates ]'
                     ''.format(mode)))
            if batch.candidate_vecs is None:
                raise ValueError(
                    "If using candidate source 'inline', then batch.candidate_vecs "
                    "cannot be None.")

            cands = batch.candidates
            # batch.candidate_vecs is a [batchsize] list of [num_cand] lists of
            # [seq_len] LongTensors
            max_len = max(len(t) for t_list in batch.candidate_vecs for t in t_list)
            cand_tensor_list = [
                self._cat_and_pad(cand_list, max_len=max_len, use_cuda=self.use_cuda)
                for cand_list in batch.candidate_vecs]
            # cand_tensor_list is a [batchsize] list of [num_cand, seq_len] LongTensors
            cand_vecs = torch.stack(cand_tensor_list, 0)
            # cands is a [batchsize, cand_len, seq_len] LongTensor
            if label_vecs is not None:
                label_inds = label_vecs.new_empty(batchsize)
                for i, label_vec in enumerate(label_vecs):
                    label_vec_pad = label_vec.new_zeros(cand_vecs[i].size(1))
                    label_vec_pad[0:label_vec.size(0)] = label_vec
                    label_inds[i] = (cand_vecs[i] == label_vec_pad).all(1).nonzero()[0]

        elif source == 'fixed':
            self._warn_once(
                flag=(mode + '_batch_candidates'),
                msg=('[ Executing {} mode with a common set of fixed candidates. ]'
                     ''.format(mode)))
            if self.fixed_candidates is None:
                raise ValueError(
                    "If using candidate source 'fixed', then you must provide the path "
                    "to a file of candidates with the flag --fixed-candidates-path")

            cands = self.fixed_candidates
            cand_vecs = self.fixed_candidate_vecs
            if label_vecs is not None:
                label_inds = label_vecs.new_empty((batchsize))
                for i, label in enumerate(label_vecs):
                    label_inds[i] = (cand_vecs == label).all(1).nonzero()

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
        shared['fixed_candidates'] = self.fixed_candidates
        shared['fixed_candidate_vecs'] = self.fixed_candidate_vecs
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
            m['loss'] = self.metrics['loss']
            m['mean_loss'] = self.metrics['loss'] / examples
            m['mean_rank'] = self.metrics['rank'] / examples
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

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

    def set_fixed_candidates(self, shared):
        """Load a set of fixed candidates and their vectors (or vectorize them here)

        Note: TorchRankerAgent by default converts candidates to vectors by vectorizing
        in the common sense (i.e., replacing each token with its index in the
        dictionary). If a child model wants to actually perform encoding, it can
        overwrite the vectorize_fixed_candidates() method to produce encoded vectors
        instead of just vectorized ones.
        """
        if shared:
            self.fixed_candidates = shared['fixed_candidates']
            self.fixed_candidate_vecs = shared['fixed_candidate_vecs']
        else:
            opt = self.opt
            if opt['fixed_candidates_path']:
                print("[ Loading fixed candidate set text from {} ]".format(
                    opt['fixed_candidates_path']))
                with open(opt['fixed_candidates_path'], 'r') as f:
                    self.fixed_candidates = [line.strip() for line in f.readlines()]

                if (opt['fixed_candidate_vecs_path'] and
                        os.path.isfile(opt['fixed_candidate_vecs_path'])):
                    print("[ Loading fixed candidate set vectors from {} ]".format(
                        opt['fixed_candidate_vecs_path']))
                    self.fixed_candidate_vecs = torch.load(
                        opt['fixed_candidate_vecs_path'])
                else:
                    cands = self.fixed_candidates
                    cand_batches = [cands[i:i + 512] for i in range(0, len(cands), 512)]
                    print("[ Vectorizing fixed candidates set from {} ({} batch(es) of "
                          "up to 512) ]".format(opt['fixed_candidates_path'],
                                          len(cand_batches)))
                    cand_vecs = []
                    for batch in cand_batches:
                        cand_vecs.extend(self.vectorize_fixed_candidates(batch))
                    self.fixed_candidate_vecs = self._cat_and_pad(cand_vecs,
                                                                  self.NULL_IDX)
                    if opt['fixed_candidate_vecs_path']:
                        print("[ Saving fixed candidate set vectors to {} ]".format(
                            opt['fixed_candidate_vecs_path']))
                        torch.save(self.fixed_candidate_vecs,
                                   opt['fixed_candidate_vecs_path'])

                if self.use_cuda:
                    self.fixed_candidate_vecs = self.fixed_candidate_vecs.cuda()
            else:
                self.fixed_candidates = None
                self.fixed_candidate_vecs = None

    def vectorize_fixed_candidates(self, cands_batch):
        return [self._vectorize_text(cand, truncate=self.truncate, truncate_left=False)
                for cand in cands_batch]

    def _cat_and_pad(self, tensor_list, max_len=None, use_cuda=False):
        """Concatenate a list of 1D LongTensors and pad it

        Args:
            tensor_list: a list of 1D LongTensors
            max_len: the length to which to pad; if None, the maximum length of the 1D
                tensors will be used
        """
        if not max_len:
            max_len = max([len(t) for t in tensor_list])
        response = torch.LongTensor(len(tensor_list), max_len).fill_(self.NULL_IDX)
        for i, tensor in enumerate(tensor_list):
            response[i, 0:len(tensor)] = tensor
        if use_cuda:
            response = response.cuda()
        return response

    def _warn_once(self, flag, msg):
        """
        Args:
            flag: The name of the flag
            msg: The message to display
        """
        warn_flag = '__warned_' + flag
        if not hasattr(self, warn_flag):
            setattr(self, warn_flag, True)
            print(msg)
