#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch import nn

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.thread_utils import SharedTable
from parlai.core.utils import round_sigfigs, padded_3d, padded_tensor, warn_once
from parlai.core.distributed_utils import is_distributed


class TorchRankerAgent(TorchAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        super(TorchRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('TorchRankerAgent')
        agent.add_argument(
            '-cands', '--candidates', type=str, default='inline',
            choices=['batch', 'inline', 'fixed', 'vocab'],
            help='The source of candidates during training '
                 '(see TorchRankerAgent._build_candidates() for details).')
        agent.add_argument(
            '-ecands', '--eval-candidates', type=str,
            choices=['batch', 'inline', 'fixed', 'vocab'],
            help='The source of candidates during evaluation (defaults to the same'
                 'value as --candidates if no flag is given)')
        agent.add_argument(
            '-fcp', '--fixed-candidates-path', type=str,
            help='A text file of fixed candidates to use for all examples, one '
                 'candidate per line')
        agent.add_argument(
            '--fixed-candidate-vecs', type=str, default='reuse',
            help="One of 'reuse', 'replace', or a path to a file with vectors "
                 "corresponding to the candidates at --fixed-candidates-path. "
                 "The default path is a /path/to/model-file.<cands_name>, where "
                 "<cands_name> is the name of the file (not the full path) passed by "
                 "the flag --fixed-candidates-path. By default, this file is created "
                 "once and reused. To replace it, use the 'replace' option.")
        agent.add_argument(
            '--prev-response-negatives', type='bool', default=False,
            help="If True, during training add the previous agent response as a "
                 "negative candidate for the current response (as a way to "
                 "organically teach the model to not repeat itself). This option is "
                 "ignored when candidate set is 'fixed' or 'vocab', as these already "
                 "include all candidates")
        agent.add_argument(
            '--prev-response-filter', type='bool', default=False,
            help="If True and --interactive=True, do not allow the model to output its "
                 "previous response. This is a hackier solution than using "
                 "--prev-response-negatives, but MUCH faster/simpler")
        agent.add_argument(
            '--interactive', type='bool', default=False,
            help="Mark as true if you are in a setting where you are only doing "
                 "evaluation, and always with the same fixed candidate set.")
        agent.add_argument(
            '--encode-fixed-candidates', type='bool', default=False,
            help="If True, encode fixed candidates in addition to vectorizing them.")

        # TODO: Move this functionality (init, loading, etc.) up to TorchAgent since
        # it has nothing to do with ranking models in particular
        agent.add_argument(
            '--init-model', type=str,
            help="The path to a saved model of the appropriate type to initialize with"
        )
        agent.add_argument(
            '--train-predict', type='bool', default=False,
            help='Get predictions and calculate mean rank during the train '
                 'step. Turning this on may slow down training.'
        )

    def __init__(self, opt, shared=None):
        # Must call _get_model_file() first so that paths are updated if necessary
        # (e.g., a .dict file)
        model_file, opt = self._get_model_file(opt)
        opt['rank_candidates'] = True
        if opt['eval_candidates'] is None:
            opt['eval_candidates'] = opt['candidates']

        # METADIALOG SPECIFIC
        if opt['interactive']:
            print("[ Setting interactive mode defaults... ]")
            opt['prev_response_filter'] = True
            opt['person_tokens'] = True

        super().__init__(opt, shared)

        if shared:
            self.model = shared['model']
            self.metrics = shared['metrics']
            self.fixed_candidates = shared['fixed_candidates']
            self.fixed_candidate_vecs = shared['fixed_candidate_vecs']
            self.vocab_candidates = shared['vocab_candidates']
            self.vocab_candidate_vecs = shared['vocab_candidate_vecs']
            states = None
        else:
            self.metrics = {'loss': 0.0, 'examples': 0, 'rank': 0,
                            'train_accuracy': 0.0}
            self.build_model()
            if model_file:
                print('Loading existing model parameters from ' + model_file)
                states = self.load(model_file)
            else:
                states = {}
            # Vectorize and save fixed/vocab candidates once upfront if applicable
            self.set_fixed_candidates(shared)
            self.set_vocab_candidates(shared)

        if opt['prev_response_negatives']:
            if opt['candidates'] in ['fixed', 'vocab']:
                msg = ("[ Option --prev-response-negatives=True is incompatible with "
                       "--candidates=['fixed','vocab']. Overriding it to False. ]")
                warn_once(msg)
                self.opt['prev_response_negatives'] = False
            self.prev_responses = None
        if opt['prev_response_filter']:
            if not opt['interactive']:
                msg = ("[ Option --prev-response-filter=True can only be used when "
                       "--interactive=True. Overriding it to False. ]")
                warn_once(msg)
                self.opt['prev_response_filter'] = False
            self.prev_response = None

        self.rank_loss = nn.CrossEntropyLoss(reduce=True, size_average=False)

        if self.use_cuda:
            self.model.cuda()
            self.rank_loss.cuda()

        if shared:
            # We don't use get here because hasattr is used on optimizer later.
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        else:
            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            self.init_optim(
                optim_params,
                states.get('optimizer'), states.get('optimizer_type')
            )
            self.build_lr_scheduler(states)

        if shared is None and is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.opt['gpu']],
                broadcast_buffers=False,
            )

    def score_candidates(self, batch, cand_vecs):
        """Given a batch and candidate set, return scores (for ranking)"""
        raise NotImplementedError(
            'Abstract class: user must implement score()')

    def build_model(self):
        """Build a new model (implemented by children classes)"""
        raise NotImplementedError(
            'Abstract class: user must implement build_model()')

    def get_batch_train_metrics(self, scores):
        batchsize = scores.size(0)
        # get accuracy
        targets = scores.new_empty(batchsize).long()
        targets = torch.arange(batchsize, out=targets)
        nb_ok = (scores.max(dim=1)[1] == targets).float().sum().item()
        self.metrics['train_accuracy'] += nb_ok
        # calculate mean rank
        above_dot_prods = scores - scores.diag().view(-1, 1)
        rank = (above_dot_prods > 0).float().sum().item()
        self.metrics['rank'] += rank

    def get_train_preds(self, scores, label_inds, cands, cand_vecs):
        # TODO: speed these calculations up
        batchsize = scores.size(0)
        _, ranks = scores.sort(1, descending=True)
        for b in range(batchsize):
            rank = (ranks[b] == label_inds[b]).nonzero().item()
            self.metrics['rank'] += 1 + rank

        # Get predictions but not full rankings for the sake of speed
        if cand_vecs.dim() == 2:
            preds = [cands[ordering[0]] for ordering in ranks]
        elif cand_vecs.dim() == 3:
            preds = [cands[i][ordering[0]] for i, ordering in enumerate(ranks)]
        return Output(preds)

    def train_step(self, batch):
        """Train on a single batch of examples."""
        if batch.text_vec is None:
            return
        batchsize = batch.text_vec.size(0)
        self.model.train()
        self.zero_grad()

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.opt['candidates'], mode='train')

        scores = self.score_candidates(batch, cand_vecs)
        _, ranks = scores.sort(1, descending=True)

        loss = self.rank_loss(scores, label_inds)

        # Update loss
        self.metrics['loss'] += loss.item()
        self.metrics['examples'] += batchsize

        for b in range(batchsize):
            rank = (ranks[b] == label_inds[b]).nonzero().item()
            self.metrics['rank'] += 1 + rank

        loss.backward()
        self.update_params()

        # Get train predictions
        if self.opt['candidates'] == 'batch':
            self.get_batch_train_metrics(scores)
            return Output()
        if not self.opt.get('train_predict', False):
            warn_once(
                "Some training metrics are omitted for speed. Set the flag "
                "`--train-predict` to calculate train metrics."
            )
            return Output()
        return self.get_train_preds(scores, label_inds, cands, cand_vecs)

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

        # Update metrics
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

        if self.opt['prev_response_filter']:
            assert(self.opt['interactive'])
            # Compare current prediction to previous (replacing if necessary)
            if self.prev_response and (preds[0] == self.prev_response):
                preds[0] = cand_preds[0][1]
            # Save current prediction for next turn
            self.prev_response = preds[0]

        if self.opt['interactive']:
            return Output(preds)
        else:
            return Output(preds, cand_preds)

    def _set_label_cands_vec(self, *args, **kwargs):
        """Sets the 'label_candidates_vec' field in the observation.

        Useful to override to change vectorization behavior"""
        obs = args[0]
        cands_key = ('candidates' if 'labels' in obs else
                     'eval_candidates' if 'eval_labels' in obs else None)
        if cands_key is None or self.opt[cands_key] != 'inline':
            # vectorize label candidates if and only if we are using inline
            # candidates
            return obs
        return super()._set_label_cands_vec(*args, **kwargs)

    def _build_candidates(self, batch, source, mode):
        """Build a candidate set for this batch

        :param batch: a Batch object (defined in torch_agent.py)
        :param source: the source from which candidates should be built, one of
            ['batch', 'inline', 'fixed', 'vocab']
        :param mode: 'train' or 'eval'

        :return: tuple of tensors (label_inds, cands, cand_vecs)
            label_inds: A [bsz] LongTensor of the indices of the labels for each
                example from its respective candidate set
            cands: A [num_cands] list of (text) candidates
                OR a [batchsize] list of such lists if source=='inline' or
                --prev-response-negatives==True
            cand_vecs: A padded [num_cands, seqlen] LongTensor of vectorized candidates
                OR a [batchsize, num_cands, seqlen] LongTensor if source=='inline' or
                --prev-response-negatives==True

        Possible sources of candidates:
            * batch: the set of all labels in this batch
                Use all labels in the batch as the candidate set (with all but the
                example's label being treated as negatives).
                Note: with this setting, the candidate set is identical for all
                examples in a batch. This option may be undesirable if it is possible
                for duplicate labels to occur in a batch, since the second instance of
                the correct label will be treated as a negative.
            * inline: batch_size lists, one list per example
                If each example comes with a list of possible candidates, use those.
                Note: With this setting, each example will have its own candidate set.
            * fixed: one global candidate list, provided in a file from the user
                If self.fixed_candidates is not None, use a set of fixed candidates for
                all examples.
                Note: this setting is not recommended for training unless the
                universe of possible candidates is very small.
            * vocab: one global candidate list, extracted from the vocabulary with the
                exception of self.NULL_IDX.
        """
        label_vecs = batch.label_vec  # [bsz] list of lists of LongTensors
        label_inds = None
        batchsize = batch.text_vec.shape[0]

        if label_vecs is not None:
            assert label_vecs.dim() == 2

        if source == 'batch':
            warn_once(
                '[ Executing {} mode with batch labels as set of candidates. ]'
                ''.format(mode)
            )
            if batchsize == 1:
                warn_once(
                    "[ Warning: using candidate source 'batch' and observed a "
                    "batch of size 1. This may be due to uneven batch sizes at "
                    "the end of an epoch. ]"
                )
            if label_vecs is None:
                raise ValueError(
                    "If using candidate source 'batch', then batch.label_vec cannot be "
                    "None.")

            cands = batch.labels
            cand_vecs = label_vecs
            label_inds = label_vecs.new_tensor(range(batchsize))

        elif source == 'inline':
            warn_once(
                '[ Executing {} mode with provided inline set of candidates ]'
                ''.format(mode)
            )
            if batch.candidate_vecs is None:
                raise ValueError(
                    "If using candidate source 'inline', then batch.candidate_vecs "
                    "cannot be None. If your task does not have inline candidates, "
                    "consider using one of --{m}={{'batch','fixed','vocab'}}."
                    "".format(m='candidates' if mode == 'train' else 'eval-candidates'))

            cands = batch.candidates
            cand_vecs = padded_3d(batch.candidate_vecs, self.NULL_IDX,
                                  use_cuda=self.use_cuda)
            if label_vecs is not None:
                label_inds = label_vecs.new_empty((batchsize))
                for i, label_vec in enumerate(label_vecs):
                    label_vec_pad = (label_vec.new_zeros(cand_vecs[i].size(1))
                                     .fill_(self.NULL_IDX))
                    if cand_vecs[i].size(1) < len(label_vec):
                        label_vec = label_vec[0:cand_vecs[i].size(1)]
                    label_vec_pad[0:label_vec.size(0)] = label_vec
                    label_inds[i] = self._find_match(cand_vecs[i], label_vec_pad)

        elif source == 'fixed':
            warn_once(
                "[ Executing {} mode with a common set of fixed candidates "
                "(n = {}). ]".format(mode, len(self.fixed_candidates))
            )
            if self.fixed_candidates is None:
                raise ValueError(
                    "If using candidate source 'fixed', then you must provide the path "
                    "to a file of candidates with the flag --fixed-candidates-path")

            cands = self.fixed_candidates
            cand_vecs = self.fixed_candidate_vecs
            if label_vecs is not None:
                label_inds = label_vecs.new_empty((batchsize))
                for i, label_vec in enumerate(label_vecs):
                    label_inds[i] = self._find_match(cand_vecs, label_vec)

        elif source == 'vocab':
            warn_once(
                '[ Executing {} mode with tokens from vocabulary as candidates. ]'
                ''.format(mode)
            )
            cands = self.vocab_candidates
            cand_vecs = self.vocab_candidate_vecs
            if label_vecs is not None:
                label_inds = label_vecs.new_empty((batchsize))
                for i, label_vec in enumerate(label_vecs):
                    label_inds[i] = self._find_match(cand_vecs, label_vec)

        if self.opt['prev_response_negatives'] and mode == 'train':
            cands, cand_vecs = self._add_prev_responses(
                batch, cands, cand_vecs, label_inds, source)

        return (cands, cand_vecs, label_inds)

    @staticmethod
    def _find_match(cand_vecs, label_vec):
        return ((cand_vecs == label_vec).sum(1) == cand_vecs.size(1)).nonzero()[0]

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
        shared['vocab_candidates'] = self.vocab_candidates
        shared['vocab_candidate_vecs'] = self.vocab_candidate_vecs
        shared['optimizer'] = self.optimizer
        return shared

    def reset_metrics(self):
        """Reset metrics."""
        super().reset_metrics()
        self.metrics['examples'] = 0
        self.metrics['loss'] = 0.0
        self.metrics['rank'] = 0
        self.metrics['train_accuracy'] = 0.0

    def report(self):
        """Report loss and mean_rank from model's perspective."""
        base = super().report()
        m = {}
        examples = self.metrics['examples']
        if examples > 0:
            m['examples'] = examples
            m['loss'] = self.metrics['loss']
            m['mean_loss'] = self.metrics['loss'] / examples
            m['mean_rank'] = self.metrics['rank'] / examples
            if self.opt['candidates'] == 'batch':
                m['train_accuracy'] = self.metrics['train_accuracy'] / examples
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def _get_model_file(self, opt):
        model_file = None

        # first check load path in case we need to override paths
        if opt.get('init_model'):
            if os.path.isfile(opt['init_model']):
                # check first for 'init_model' for loading model from file
                model_file = opt['init_model']
            else:
                raise RuntimeError(
                    "Specified --init-model={} could not be found."
                    .format(opt['init_model'])
                )

        if opt.get('model_file') and os.path.isfile(opt['model_file']):
            # next check for existing 'model_file'; this would override init_model
            model_file = opt['model_file']

        if model_file is not None:
            # if we are loading a model, should load its dict too
            if (os.path.isfile(model_file + '.dict') or
                    opt['dict_file'] is None):
                opt['dict_file'] = model_file + '.dict'

        return model_file, opt

    def set_vocab_candidates(self, shared):
        """Load the tokens from the vocab as candidates

        self.vocab_candidates will contain a [num_cands] list of strings
        self.vocab_candidate_vecs will contain a [num_cands, 1] LongTensor
        """
        if shared:
            self.vocab_candidates = shared['vocab_candidates']
            self.vocab_candidate_vecs = shared['vocab_candidate_vecs']
        else:
            if 'vocab' in (self.opt['candidates'], self.opt['eval_candidates']):
                cands = []
                vecs = []
                for ind in range(1, len(self.dict)):
                    cands.append(self.dict.ind2tok[ind])
                    vecs.append(ind)
                self.vocab_candidates = cands
                self.vocab_candidate_vecs = torch.LongTensor(vecs).unsqueeze(1)
                print("[ Loaded fixed candidate set (n = {}) from vocabulary ]"
                      "".format(len(self.vocab_candidates)))
                if self.use_cuda:
                    self.vocab_candidate_vecs = self.vocab_candidate_vecs.cuda()
            else:
                self.vocab_candidates = None
                self.vocab_candidate_vecs = None

    def set_fixed_candidates(self, shared):
        """Load a set of fixed candidates and their vectors (or vectorize them here)

        self.fixed_candidates will contain a [num_cands] list of strings
        self.fixed_candidate_vecs will contain a [num_cands, seq_len] LongTensor
            (or optionally a FloatTensor if a child class overrides
            vectorize_fixed_candidates() to produce encoded vectors instead of just
            vectorized ones)

        See the note on the --fixed-candidate-vecs flag for an explanation of the
        'reuse', 'replace', or path options.

        Note: TorchRankerAgent by default converts candidates to vectors by vectorizing
        in the common sense (i.e., replacing each token with its index in the
        dictionary). If a child model wants to additionally perform encoding, it can
        overwrite the vectorize_fixed_candidates() method to produce encoded vectors
        instead of just vectorized ones.
        """
        if shared:
            self.fixed_candidates = shared['fixed_candidates']
            self.fixed_candidate_vecs = shared['fixed_candidate_vecs']
        else:
            opt = self.opt
            cand_path = opt['fixed_candidates_path']
            if ('fixed' in (opt['candidates'], opt['eval_candidates']) and
                    cand_path):

                # Load candidates
                print("[ Loading fixed candidate set from {} ]".format(cand_path))
                with open(cand_path, 'r') as f:
                    cands = list(set(line.strip() for line in f.readlines()))

                # Load or create candidate vectors
                if os.path.isfile(opt['fixed_candidate_vecs']):
                    vecs_path = opt['fixed_candidate_vecs']
                else:
                    setting = opt['fixed_candidate_vecs']
                    model_dir, model_file = os.path.split(self.opt['model_file'])
                    model_name = os.path.splitext(model_file)[0]
                    cands_name = os.path.splitext(os.path.basename(cand_path))[0]
                    vecs_path = os.path.join(
                        model_dir, '.'.join([model_name, cands_name]))

                if setting == 'reuse' and os.path.isfile(vecs_path):
                    vecs = self.load_candidate_vecs(vecs_path)
                else:  # setting == 'replace' OR generating for the first time
                    vecs = self.make_candidate_vecs(cands)
                    self.save_candidate_vecs(vecs, vecs_path)

                self.fixed_candidates = cands
                self.fixed_candidate_vecs = vecs

                if self.use_cuda:
                    self.fixed_candidate_vecs = self.fixed_candidate_vecs.cuda()
            else:
                self.fixed_candidates = None
                self.fixed_candidate_vecs = None

    def make_candidate_vecs(self, cands):
        cand_batches = [cands[i:i + 512] for i in range(0, len(cands), 512)]
        print("[ Vectorizing fixed candidate set ({} batch(es) of up to 512) ]"
              "".format(len(cand_batches)))
        cand_vecs = []
        for batch in cand_batches:
            cand_vecs.extend(self.vectorize_fixed_candidates(batch))
        return padded_3d([cand_vecs], dtype=cand_vecs[0].dtype).squeeze(0)

    def vectorize_fixed_candidates(self, cands_batch):
        """Convert a batch of candidates from text to vectors

        :param cands_batch: a [batchsize] list of candidates (strings)
        :returns: a [num_cands] list of candidate vectors

        By default, candidates are simply vectorized (tokens replaced by token ids).
        To also encode candidates with a trained encoder, a child class may provide
        override this method with its own.
        """
        return [self._vectorize_text(cand, truncate=self.truncate,
                                     truncate_left=False) for cand in cands_batch]

    def save_candidate_vecs(self, vecs, path):
        print("[ Saving fixed candidate set vectors to {} ]".format(path))
        with open(path, 'wb') as f:
            torch.save(vecs, f)

    def load_candidate_vecs(self, path):
        print("[ Loading fixed candidate set vectors from {} ]".format(path))
        return torch.load(path)

    def _add_prev_responses(self, batch, cands, cand_vecs, label_inds, source):
        assert(source not in ['fixed', 'vocab'])
        self._extract_prev_responses(batch)

        # Add prev_responses as negatives
        prev_cands = self.prev_responses
        prev_cand_vecs = [torch.Tensor(self.dict.txt2vec(cand)) for cand in prev_cands]
        if source == 'batch':
            # Option 1: Change from one set of shared candidates to separate per example
            # cands = [cands + [prev_cand] for prev_cand in prev_cands]
            # list_of_lists_of_cand_vecs = [[vec for vec in cand_vecs] + [prev_cand_vec]
            #                            for prev_cand_vec in prev_cand_vecs]
            # cand_vecs = padded_3d(list_of_lists_of_cand_vecs, use_cuda=self.use_cuda,
            #                       dtype=cand_vecs[0].dtype)

            # Option 2: Just add all prev responses for the whole batch (doubles bs)
            cands += prev_cands
            cand_vecs, _ = padded_tensor([vec for vec in cand_vecs] + prev_cand_vecs,
                                         use_cuda=self.use_cuda)
        elif source == 'inline':
            raise NotImplementedError

        return cands, cand_vecs

    def _extract_prev_responses(self, batch):
        # Extract prev_responses for metadialog-formatted examples
        warn_once("WARNING: This code is specific to metadialog-formatted examples")

        # TODO: Pull out p1/p2 once elsewhere, not every time
        p1 = self.dict.txt2vec('__p1__')[0]
        p2 = self.dict.txt2vec('__p2__')[0]
        self.prev_responses = []
        # Do naively for now with a for loop
        for text_vec in batch.text_vec:
            p1s = (text_vec == p1).nonzero()
            p2s = (text_vec == p2).nonzero()
            if len(p1s) and len(p2s):
                response_vec = text_vec[p2s[-1] + 1:p1s[-1]]
            else:
                response_vec = [self.NULL_IDX]  # TODO: pull in actual N
            response = self.dict.vec2txt(response_vec)
            self.prev_responses.append(response)
