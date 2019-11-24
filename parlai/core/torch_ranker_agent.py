#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Torch Ranker Agents provide functionality for building ranking models.

See the TorchRankerAgent tutorial for examples.
"""

from abc import abstractmethod
from itertools import islice
import os
from tqdm import tqdm
import random

import torch

from parlai.utils.distributed import is_distributed
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, padded_3d, warn_once, padded_tensor


class TorchRankerAgent(TorchAgent):
    """
    Abstract TorchRankerAgent class; only meant to be extended.

    TorchRankerAgents aim to provide convenient functionality for building ranking
    models. This includes:

    - Training/evaluating on candidates from a variety of sources.
    - Computing hits@1, hits@5, mean reciprical rank (MRR), and other metrics.
    - Caching representations for fast runtime when deploying models to production.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add CLI args."""
        super(TorchRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('TorchRankerAgent')
        agent.add_argument(
            '-cands',
            '--candidates',
            type=str,
            default='inline',
            choices=['batch', 'inline', 'fixed', 'batch-all-cands'],
            help='The source of candidates during training '
            '(see TorchRankerAgent._build_candidates() for details).',
        )
        agent.add_argument(
            '-ecands',
            '--eval-candidates',
            type=str,
            default='inline',
            choices=['batch', 'inline', 'fixed', 'vocab', 'batch-all-cands'],
            help='The source of candidates during evaluation (defaults to the same'
            'value as --candidates if no flag is given)',
        )
        agent.add_argument(
            '--repeat-blocking-heuristic',
            type='bool',
            default=True,
            help='Block repeating previous utterances. '
            'Helpful for many models that score repeats highly, so switched '
            'on by default.',
        )
        agent.add_argument(
            '-fcp',
            '--fixed-candidates-path',
            type=str,
            help='A text file of fixed candidates to use for all examples, one '
            'candidate per line',
        )
        agent.add_argument(
            '--fixed-candidate-vecs',
            type=str,
            default='reuse',
            help='One of "reuse", "replace", or a path to a file with vectors '
            'corresponding to the candidates at --fixed-candidates-path. '
            'The default path is a /path/to/model-file.<cands_name>, where '
            '<cands_name> is the name of the file (not the full path) passed by '
            'the flag --fixed-candidates-path. By default, this file is created '
            'once and reused. To replace it, use the "replace" option.',
        )
        agent.add_argument(
            '--encode-candidate-vecs',
            type='bool',
            default=True,
            help='Cache and save the encoding of the candidate vecs. This '
            'might be used when interacting with the model in real time '
            'or evaluating on fixed candidate set when the encoding of '
            'the candidates is independent of the input.',
        )
        agent.add_argument(
            '--init-model',
            type=str,
            default=None,
            help='Initialize model with weights from this file.',
        )
        agent.add_argument(
            '--train-predict',
            type='bool',
            default=False,
            help='Get predictions and calculate mean rank during the train '
            'step. Turning this on may slow down training.',
        )
        agent.add_argument(
            '--cap-num-predictions',
            type=int,
            default=100,
            help='Limit to the number of predictions in output.text_candidates',
        )
        agent.add_argument(
            '--ignore-bad-candidates',
            type='bool',
            default=False,
            help='Ignore examples for which the label is not present in the '
            'label candidates. Default behavior results in RuntimeError. ',
        )
        agent.add_argument(
            '--rank-top-k',
            type=int,
            default=-1,
            help='Ranking returns the top k results of k > 0, otherwise sorts every '
            'single candidate according to the ranking.',
        )
        agent.add_argument(
            '--inference',
            choices={'max', 'topk'},
            default='max',
            help='Final response output algorithm',
        )
        agent.add_argument(
            '--topk',
            type=int,
            default=5,
            help='K used in Top K sampling inference, when selected',
        )

    def __init__(self, opt, shared=None):
        # Must call _get_init_model() first so that paths are updated if necessary
        # (e.g., a .dict file)
        init_model, is_finetune = self._get_init_model(opt, shared)
        opt['rank_candidates'] = True
        super().__init__(opt, shared)

        if shared:
            states = None
        else:
            # Note: we cannot change the type of metrics ahead of time, so you
            # should correctly initialize to floats or ints here
            self.metrics['loss'] = 0.0
            self.metrics['examples'] = 0
            self.metrics['rank'] = 0.0
            self.metrics['mrr'] = 0.0
            self.metrics['train_accuracy'] = 0.0

            self.criterion = self.build_criterion()
            self.model = self.build_model()
            if self.model is None or self.criterion is None:
                raise AttributeError(
                    'build_model() and build_criterion() need to return the model or criterion'
                )
            if self.use_cuda:
                self.model.cuda()
                self.criterion.cuda()

            print("Total parameters: {}".format(self._total_parameters()))
            print("Trainable parameters:  {}".format(self._trainable_parameters()))

            if self.fp16:
                self.model = self.model.half()
            if init_model:
                print('Loading existing model parameters from ' + init_model)
                states = self.load(init_model)
            else:
                states = {}

        self.rank_top_k = opt.get('rank_top_k', -1)

        # Vectorize and save fixed/vocab candidates once upfront if applicable
        self.set_fixed_candidates(shared)
        self.set_vocab_candidates(shared)

        if shared:
            # We don't use get here because hasattr is used on optimizer later.
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        else:
            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            self.init_optim(
                optim_params, states.get('optimizer'), states.get('optimizer_type')
            )
            self.build_lr_scheduler(states, hard_reset=is_finetune)

        if shared is None and is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.opt['gpu']], broadcast_buffers=False
            )

    def build_criterion(self):
        """
        Construct and return the loss function.

        By default torch.nn.CrossEntropyLoss.
        """
        return torch.nn.CrossEntropyLoss(reduction='sum')

    def set_interactive_mode(self, mode, shared=False):
        super().set_interactive_mode(mode, shared)
        self.candidates = self.opt['candidates']
        self.encode_candidate_vecs = self.opt['encode_candidate_vecs']
        if mode:
            self.eval_candidates = 'fixed'
            self.ignore_bad_candidates = True
            self.fixed_candidates_path = self.opt['fixed_candidates_path']
            if self.fixed_candidates_path is None or self.fixed_candidates_path == '':
                # Attempt to get a standard candidate set for the given task
                path = self.get_task_candidates_path()
                if path:
                    if not shared:
                        print("[setting fixed_candidates path to: " + path + " ]")
                    self.fixed_candidates_path = path
        else:
            self.eval_candidates = self.opt['eval_candidates']
            self.ignore_bad_candidates = self.opt.get('ignore_bad_candidates', False)
            self.fixed_candidates_path = self.opt['fixed_candidates_path']

    def get_task_candidates_path(self):
        path = self.opt['model_file'] + '.cands-' + self.opt['task'] + '.cands'
        if os.path.isfile(path) and self.opt['fixed_candidate_vecs'] == 'reuse':
            return path
        print("[ *** building candidates file as they do not exist: " + path + ' *** ]')
        from parlai.scripts.build_candidates import build_cands
        from copy import deepcopy

        opt = deepcopy(self.opt)
        opt['outfile'] = path
        opt['datatype'] = 'train:evalmode'
        opt['interactive_task'] = False
        opt['batchsize'] = 1
        build_cands(opt)
        return path

    @abstractmethod
    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        Given a batch and candidate set, return scores (for ranking).

        :param Batch batch:
            a Batch object (defined in torch_agent.py)
        :param LongTensor cand_vecs:
            padded and tokenized candidates
        :param FloatTensor cand_encs:
            encoded candidates, if these are passed into the function (in cases
            where we cache the candidate encodings), you do not need to call
            self.model on cand_vecs
        """
        pass

    def _get_batch_train_metrics(self, scores):
        """
        Get fast metrics calculations if we train with batch candidates.

        Specifically, calculate accuracy ('train_accuracy'), average rank,
        and mean reciprocal rank.
        """
        batchsize = scores.size(0)
        # get accuracy
        targets = scores.new_empty(batchsize).long()
        targets = torch.arange(batchsize, out=targets)
        nb_ok = (scores.max(dim=1)[1] == targets).float().sum().item()
        self.metrics['train_accuracy'] += nb_ok
        # calculate mean_rank
        above_dot_prods = scores - scores.diag().view(-1, 1)
        ranks = (above_dot_prods > 0).float().sum(dim=1) + 1
        mrr = 1.0 / (ranks + 0.00001)
        self.metrics['rank'] += torch.sum(ranks).item()
        self.metrics['mrr'] += torch.sum(mrr).item()

    def _get_train_preds(self, scores, label_inds, cands, cand_vecs):
        """Return predictions from training."""
        # TODO: speed these calculations up
        batchsize = scores.size(0)
        if self.rank_top_k > 0:
            _, ranks = scores.topk(
                min(self.rank_top_k, scores.size(1)), 1, largest=True
            )
        else:
            _, ranks = scores.sort(1, descending=True)
        for b in range(batchsize):
            rank = (ranks[b] == label_inds[b]).nonzero()
            rank = rank.item() if len(rank) == 1 else scores.size(1)
            self.metrics['rank'] += 1 + rank
            self.metrics['mrr'] += 1.0 / (1 + rank)

        ranks = ranks.cpu()
        # Here we get the top prediction for each example, but do not
        # return the full ranked list for the sake of training speed
        preds = []
        for i, ordering in enumerate(ranks):
            if cand_vecs.dim() == 2:  # num cands x max cand length
                cand_list = cands
            elif cand_vecs.dim() == 3:  # batchsize x num cands x max cand length
                cand_list = cands[i]
            if len(ordering) != len(cand_list):
                # We may have added padded cands to fill out the batch;
                # Here we break after finding the first non-pad cand in the
                # ranked list
                for x in ordering:
                    if x < len(cand_list):
                        preds.append(cand_list[x])
                        break
            else:
                preds.append(cand_list[ordering[0]])

        return Output(preds)

    def is_valid(self, obs):
        """
        Override from TorchAgent.

        Check to see if label candidates contain the label.
        """
        if not self.ignore_bad_candidates:
            return super().is_valid(obs)

        if not super().is_valid(obs):
            return False

        # skip examples for which the set of label candidates do not
        # contain the label
        if 'labels_vec' in obs and 'label_candidates_vecs' in obs:
            cand_vecs = obs['label_candidates_vecs']
            label_vec = obs['labels_vec']
            matches = [x for x in cand_vecs if torch.equal(x, label_vec)]
            if len(matches) == 0:
                warn_once(
                    'At least one example has a set of label candidates that '
                    'does not contain the label.'
                )
                return False

        return True

    def train_step(self, batch):
        """Train on a single batch of examples."""
        if batch.text_vec is None and batch.image is None:
            return
        batchsize = (
            batch.text_vec.size(0)
            if batch.text_vec is not None
            else batch.image.size(0)
        )
        self.model.train()
        self.zero_grad()

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.candidates, mode='train'
        )
        try:
            scores = self.score_candidates(batch, cand_vecs)
            loss = self.criterion(scores, label_inds)
            self.backward(loss)
            self.update_params()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print(
                    '| WARNING: ran out of memory, skipping batch. '
                    'if this happens frequently, decrease batchsize or '
                    'truncate the inputs to the model.'
                )
                return Output()
            else:
                raise e

        # Update loss
        self.metrics['loss'] += loss.item()
        self.metrics['examples'] += batchsize

        # Get train predictions
        if self.candidates == 'batch':
            self._get_batch_train_metrics(scores)
            return Output()
        if not self.opt.get('train_predict', False):
            warn_once(
                "Some training metrics are omitted for speed. Set the flag "
                "`--train-predict` to calculate train metrics."
            )
            return Output()
        return self._get_train_preds(scores, label_inds, cands, cand_vecs)

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None and batch.image is None:
            return
        batchsize = (
            batch.text_vec.size(0)
            if batch.text_vec is not None
            else batch.image.size(0)
        )
        self.model.eval()

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.eval_candidates, mode='eval'
        )

        cand_encs = None
        if self.encode_candidate_vecs and self.eval_candidates in ['fixed', 'vocab']:
            # if we cached candidate encodings for a fixed list of candidates,
            # pass those into the score_candidates function
            if self.eval_candidates == 'fixed':
                cand_encs = self.fixed_candidate_encs
            elif self.eval_candidates == 'vocab':
                cand_encs = self.vocab_candidate_encs

        scores = self.score_candidates(batch, cand_vecs, cand_encs=cand_encs)
        if self.rank_top_k > 0:
            _, ranks = scores.topk(
                min(self.rank_top_k, scores.size(1)), 1, largest=True
            )
        else:
            _, ranks = scores.sort(1, descending=True)

        # Update metrics
        if label_inds is not None:
            loss = self.criterion(scores, label_inds)
            self.metrics['loss'] += loss.item()
            self.metrics['examples'] += batchsize
            for b in range(batchsize):
                rank = (ranks[b] == label_inds[b]).nonzero()
                rank = rank.item() if len(rank) == 1 else scores.size(1)
                self.metrics['rank'] += 1 + rank
                self.metrics['mrr'] += 1.0 / (1 + rank)

        ranks = ranks.cpu()
        max_preds = self.opt['cap_num_predictions']
        cand_preds = []
        for i, ordering in enumerate(ranks):
            if cand_vecs.dim() == 2:
                cand_list = cands
            elif cand_vecs.dim() == 3:
                cand_list = cands[i]
            # using a generator instead of a list comprehension allows
            # to cap the number of elements.
            cand_preds_generator = (
                cand_list[rank] for rank in ordering if rank < len(cand_list)
            )
            cand_preds.append(list(islice(cand_preds_generator, max_preds)))

        if (
            self.opt.get('repeat_blocking_heuristic', True)
            and self.eval_candidates == 'fixed'
        ):
            cand_preds = self.block_repeats(cand_preds)

        if self.opt.get('inference', 'max') == 'max':
            preds = [cand_preds[i][0] for i in range(batchsize)]
        else:
            # Top-k inference.
            preds = []
            for i in range(batchsize):
                preds.append(random.choice(cand_preds[i][0 : self.opt['topk']]))

        return Output(preds, cand_preds)

    def block_repeats(self, cand_preds):
        """Heuristic to block a model repeating a line from the history."""
        history_strings = []
        for h in self.history.history_raw_strings:
            # Heuristic: Block any given line in the history, splitting by '\n'.
            history_strings.extend(h.split('\n'))

        new_preds = []
        for cp in cand_preds:
            np = []
            for c in cp:
                if c not in history_strings:
                    np.append(c)
            new_preds.append(np)
        return new_preds

    def _set_label_cands_vec(self, *args, **kwargs):
        """
        Set the 'label_candidates_vec' field in the observation.

        Useful to override to change vectorization behavior.
        """
        obs = args[0]
        if 'labels' in obs:
            cands_key = 'candidates'
        else:
            cands_key = 'eval_candidates'
        if self.opt[cands_key] not in ['inline', 'batch-all-cands']:
            # vectorize label candidates if and only if we are using inline
            # candidates
            return obs
        return super()._set_label_cands_vec(*args, **kwargs)

    def _build_candidates(self, batch, source, mode):
        """
        Build a candidate set for this batch.

        :param batch:
            a Batch object (defined in torch_agent.py)
        :param source:
            the source from which candidates should be built, one of
            ['batch', 'batch-all-cands', 'inline', 'fixed']
        :param mode:
            'train' or 'eval'

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
                example's label being treated as negatives).
                Note: with this setting, the candidate set is identical for all
                examples in a batch. This option may be undesirable if it is possible
                for duplicate labels to occur in a batch, since the second instance of
                the correct label will be treated as a negative.
            * batch-all-cands: the set of all candidates in this batch
                Use all candidates in the batch as candidate set.
                Note 1: This can result in a very large number of candidates.
                Note 2: In this case we will deduplicate candidates.
                Note 3: just like with 'batch' the candidate set is identical
                for all examples in a batch.
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
        batchsize = (
            batch.text_vec.size(0)
            if batch.text_vec is not None
            else batch.image.size(0)
        )

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
                    "None."
                )

            cands = batch.labels
            cand_vecs = label_vecs
            label_inds = label_vecs.new_tensor(range(batchsize))

        elif source == 'batch-all-cands':
            warn_once(
                '[ Executing {} mode with all candidates provided in the batch ]'
                ''.format(mode)
            )
            if batch.candidate_vecs is None:
                raise ValueError(
                    "If using candidate source 'batch-all-cands', then batch."
                    "candidate_vecs cannot be None. If your task does not have "
                    "inline candidates, consider using one of "
                    "--{m}={{'batch','fixed','vocab'}}."
                    "".format(m='candidates' if mode == 'train' else 'eval-candidates')
                )
            # initialize the list of cands with the labels
            cands = []
            all_cands_vecs = []
            # dictionary used for deduplication
            cands_to_id = {}
            for i, cands_for_sample in enumerate(batch.candidates):
                for j, cand in enumerate(cands_for_sample):
                    if cand not in cands_to_id:
                        cands.append(cand)
                        cands_to_id[cand] = len(cands_to_id)
                        all_cands_vecs.append(batch.candidate_vecs[i][j])
            cand_vecs, _ = padded_tensor(
                all_cands_vecs,
                self.NULL_IDX,
                use_cuda=self.use_cuda,
                fp16friendly=self.fp16,
            )
            label_inds = label_vecs.new_tensor(
                [cands_to_id[label] for label in batch.labels]
            )

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
                    "".format(m='candidates' if mode == 'train' else 'eval-candidates')
                )

            cands = batch.candidates
            cand_vecs = padded_3d(
                batch.candidate_vecs,
                self.NULL_IDX,
                use_cuda=self.use_cuda,
                fp16friendly=self.fp16,
            )
            if label_vecs is not None:
                label_inds = label_vecs.new_empty((batchsize))
                bad_batch = False
                for i, label_vec in enumerate(label_vecs):
                    label_vec_pad = label_vec.new_zeros(cand_vecs[i].size(1)).fill_(
                        self.NULL_IDX
                    )
                    if cand_vecs[i].size(1) < len(label_vec):
                        label_vec = label_vec[0 : cand_vecs[i].size(1)]
                    label_vec_pad[0 : label_vec.size(0)] = label_vec
                    label_inds[i] = self._find_match(cand_vecs[i], label_vec_pad)
                    if label_inds[i] == -1:
                        bad_batch = True
                if bad_batch:
                    if self.ignore_bad_candidates and not self.is_training:
                        label_inds = None
                    else:
                        raise RuntimeError(
                            'At least one of your examples has a set of label candidates '
                            'that does not contain the label. To ignore this error '
                            'set `--ignore-bad-candidates True`.'
                        )

        elif source == 'fixed':
            if self.fixed_candidates is None:
                raise ValueError(
                    "If using candidate source 'fixed', then you must provide the path "
                    "to a file of candidates with the flag --fixed-candidates-path or "
                    "the name of a task with --fixed-candidates-task."
                )
            warn_once(
                "[ Executing {} mode with a common set of fixed candidates "
                "(n = {}). ]".format(mode, len(self.fixed_candidates))
            )

            cands = self.fixed_candidates
            cand_vecs = self.fixed_candidate_vecs

            if label_vecs is not None:
                label_inds = label_vecs.new_empty((batchsize))
                bad_batch = False
                for batch_idx, label_vec in enumerate(label_vecs):
                    max_c_len = cand_vecs.size(1)
                    label_vec_pad = label_vec.new_zeros(max_c_len).fill_(self.NULL_IDX)
                    if max_c_len < len(label_vec):
                        label_vec = label_vec[0:max_c_len]
                    label_vec_pad[0 : label_vec.size(0)] = label_vec
                    label_inds[batch_idx] = self._find_match(cand_vecs, label_vec_pad)
                    if label_inds[batch_idx] == -1:
                        bad_batch = True
                if bad_batch:
                    if self.ignore_bad_candidates and not self.is_training:
                        label_inds = None
                    else:
                        raise RuntimeError(
                            'At least one of your examples has a set of label candidates '
                            'that does not contain the label. To ignore this error '
                            'set `--ignore-bad-candidates True`.'
                        )

        elif source == 'vocab':
            warn_once(
                '[ Executing {} mode with tokens from vocabulary as candidates. ]'
                ''.format(mode)
            )
            cands = self.vocab_candidates
            cand_vecs = self.vocab_candidate_vecs
            # NOTE: label_inds is None here, as we will not find the label in
            # the set of vocab candidates
        else:
            raise Exception("Unrecognized source: %s" % source)

        return (cands, cand_vecs, label_inds)

    @staticmethod
    def _find_match(cand_vecs, label_vec):
        matches = ((cand_vecs == label_vec).sum(1) == cand_vecs.size(1)).nonzero()
        if len(matches) > 0:
            return matches[0]
        return -1

    def share(self):
        """Share model parameters."""
        shared = super().share()
        shared['fixed_candidates'] = self.fixed_candidates
        shared['fixed_candidate_vecs'] = self.fixed_candidate_vecs
        shared['fixed_candidate_encs'] = self.fixed_candidate_encs
        shared['vocab_candidates'] = self.vocab_candidates
        shared['vocab_candidate_vecs'] = self.vocab_candidate_vecs
        shared['vocab_candidate_encs'] = self.vocab_candidate_encs
        shared['optimizer'] = self.optimizer
        return shared

    def reset_metrics(self):
        """Reset metrics."""
        super().reset_metrics()
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here
        self.metrics['examples'] = 0
        self.metrics['loss'] = 0.0
        self.metrics['rank'] = 0.0
        self.metrics['mrr'] = 0.0
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
            batch_train = self.candidates == 'batch' and self.is_training
            if not self.is_training or self.opt.get('train_predict') or batch_train:
                m['mean_rank'] = self.metrics['rank'] / examples
                m['mrr'] = self.metrics['mrr'] / examples
            if batch_train:
                m['train_accuracy'] = self.metrics['train_accuracy'] / examples
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def set_vocab_candidates(self, shared):
        """
        Load the tokens from the vocab as candidates.

        self.vocab_candidates will contain a [num_cands] list of strings
        self.vocab_candidate_vecs will contain a [num_cands, 1] LongTensor
        """
        if shared:
            self.vocab_candidates = shared['vocab_candidates']
            self.vocab_candidate_vecs = shared['vocab_candidate_vecs']
            self.vocab_candidate_encs = shared['vocab_candidate_encs']
        else:
            if 'vocab' in (self.opt['candidates'], self.opt['eval_candidates']):
                cands = []
                vecs = []
                for ind in range(1, len(self.dict)):
                    cands.append(self.dict.ind2tok[ind])
                    vecs.append(ind)
                self.vocab_candidates = cands
                self.vocab_candidate_vecs = torch.LongTensor(vecs).unsqueeze(1)
                print(
                    "[ Loaded fixed candidate set (n = {}) from vocabulary ]"
                    "".format(len(self.vocab_candidates))
                )
                if self.use_cuda:
                    self.vocab_candidate_vecs = self.vocab_candidate_vecs.cuda()

                if self.encode_candidate_vecs:
                    # encode vocab candidate vecs
                    self.vocab_candidate_encs = self._make_candidate_encs(
                        self.vocab_candidate_vecs
                    )
                    if self.use_cuda:
                        self.vocab_candidate_encs = self.vocab_candidate_encs.cuda()
                    if self.fp16:
                        self.vocab_candidate_encs = self.vocab_candidate_encs.half()
                    else:
                        self.vocab_candidate_encs = self.vocab_candidate_encs.float()
                else:
                    self.vocab_candidate_encs = None
            else:
                self.vocab_candidates = None
                self.vocab_candidate_vecs = None
                self.vocab_candidate_encs = None

    def set_fixed_candidates(self, shared):
        """
        Load a set of fixed candidates and their vectors (or vectorize them here).

        self.fixed_candidates will contain a [num_cands] list of strings
        self.fixed_candidate_vecs will contain a [num_cands, seq_len] LongTensor

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
            self.fixed_candidate_encs = shared['fixed_candidate_encs']
        else:
            opt = self.opt
            cand_path = self.fixed_candidates_path
            if 'fixed' in (self.candidates, self.eval_candidates):
                if not cand_path:
                    # Attempt to get a standard candidate set for the given task
                    path = self.get_task_candidates_path()
                    if path:
                        print("[setting fixed_candidates path to: " + path + " ]")
                        self.fixed_candidates_path = path
                        cand_path = self.fixed_candidates_path
                # Load candidates
                print("[ Loading fixed candidate set from {} ]".format(cand_path))
                with open(cand_path, 'r', encoding='utf-8') as f:
                    cands = [line.strip() for line in f.readlines()]
                # Load or create candidate vectors
                if os.path.isfile(self.opt['fixed_candidate_vecs']):
                    vecs_path = opt['fixed_candidate_vecs']
                    vecs = self.load_candidates(vecs_path)
                else:
                    setting = self.opt['fixed_candidate_vecs']
                    model_dir, model_file = os.path.split(self.opt['model_file'])
                    model_name = os.path.splitext(model_file)[0]
                    cands_name = os.path.splitext(os.path.basename(cand_path))[0]
                    vecs_path = os.path.join(
                        model_dir, '.'.join([model_name, cands_name, 'vecs'])
                    )
                    if setting == 'reuse' and os.path.isfile(vecs_path):
                        vecs = self.load_candidates(vecs_path)
                    else:  # setting == 'replace' OR generating for the first time
                        vecs = self._make_candidate_vecs(cands)
                        self._save_candidates(vecs, vecs_path)

                self.fixed_candidates = cands
                self.fixed_candidate_vecs = vecs
                if self.use_cuda:
                    self.fixed_candidate_vecs = self.fixed_candidate_vecs.cuda()

                if self.encode_candidate_vecs:
                    # candidate encodings are fixed so set them up now
                    enc_path = os.path.join(
                        model_dir, '.'.join([model_name, cands_name, 'encs'])
                    )
                    if setting == 'reuse' and os.path.isfile(enc_path):
                        encs = self.load_candidates(enc_path, cand_type='encodings')
                    else:
                        encs = self._make_candidate_encs(self.fixed_candidate_vecs)
                        self._save_candidates(
                            encs, path=enc_path, cand_type='encodings'
                        )
                    self.fixed_candidate_encs = encs
                    if self.use_cuda:
                        self.fixed_candidate_encs = self.fixed_candidate_encs.cuda()
                    if self.fp16:
                        self.fixed_candidate_encs = self.fixed_candidate_encs.half()
                    else:
                        self.fixed_candidate_encs = self.fixed_candidate_encs.float()
                else:
                    self.fixed_candidate_encs = None

            else:
                self.fixed_candidates = None
                self.fixed_candidate_vecs = None
                self.fixed_candidate_encs = None

    def load_candidates(self, path, cand_type='vectors'):
        """Load fixed candidates from a path."""
        print("[ Loading fixed candidate set {} from {} ]".format(cand_type, path))
        return torch.load(path, map_location=lambda cpu, _: cpu)

    def _make_candidate_vecs(self, cands):
        """Prebuild cached vectors for fixed candidates."""
        cand_batches = [cands[i : i + 512] for i in range(0, len(cands), 512)]
        print(
            "[ Vectorizing fixed candidate set ({} batch(es) of up to 512) ]"
            "".format(len(cand_batches))
        )
        cand_vecs = []
        for batch in tqdm(cand_batches):
            cand_vecs.extend(self.vectorize_fixed_candidates(batch))
        return padded_3d([cand_vecs], dtype=cand_vecs[0].dtype).squeeze(0)

    def _save_candidates(self, vecs, path, cand_type='vectors'):
        """Save cached vectors."""
        print("[ Saving fixed candidate set {} to {} ]".format(cand_type, path))
        with open(path, 'wb') as f:
            torch.save(vecs, f)

    def encode_candidates(self, padded_cands):
        """
        Convert the given candidates to vectors.

        This is an abstract method that must be implemented by the user.

        :param padded_cands:
            The padded candidates.
        """
        raise NotImplementedError(
            'Abstract method: user must implement encode_candidates(). '
            'If your agent encodes candidates independently '
            'from context, you can get performance gains with fixed cands by '
            'implementing this function and running with the flag '
            '--encode-candidate-vecs True.'
        )

    def _make_candidate_encs(self, vecs):
        """
        Encode candidates from candidate vectors.

        Requires encode_candidates() to be implemented.
        """

        cand_encs = []
        vec_batches = [vecs[i : i + 256] for i in range(0, len(vecs), 256)]
        print(
            "[ Encoding fixed candidates set from ({} batch(es) of up to 256) ]"
            "".format(len(vec_batches))
        )
        # Put model into eval mode when encoding candidates
        self.model.eval()
        with torch.no_grad():
            for vec_batch in tqdm(vec_batches):
                cand_encs.append(self.encode_candidates(vec_batch).cpu())
        return torch.cat(cand_encs, 0).to(vec_batch.device)

    def vectorize_fixed_candidates(self, cands_batch, add_start=False, add_end=False):
        """
        Convert a batch of candidates from text to vectors.

        :param cands_batch:
            a [batchsize] list of candidates (strings)
        :returns:
            a [num_cands] list of candidate vectors

        By default, candidates are simply vectorized (tokens replaced by token ids).
        A child class may choose to overwrite this method to perform vectorization as
        well as encoding if so desired.
        """
        return [
            self._vectorize_text(
                cand,
                truncate=self.label_truncate,
                truncate_left=False,
                add_start=add_start,
                add_end=add_end,
            )
            for cand in cands_batch
        ]
