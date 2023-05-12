#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Generic PyTorch-based Generator agent.

Implements quite a bit of boilerplate, including forced-decoding loss and a tree search.

Contains the following utilities:

* `ref:TorchGeneratorAgent` class, which serves as a useful parent for generative torch
  agents.
* Beam class which provides some generic beam functionality for classes to use
"""

from typing_extensions import TypedDict
from parlai.core.params import ParlaiParser
from abc import ABC, abstractmethod
from typing import TypeVar, List, Dict, Optional, Tuple, Set, Iterable
import math
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.core.torch_agent import TorchAgent, Batch, Output, DictionaryAgent
from parlai.utils.misc import warn_once
from parlai.utils.io import PathManager
import parlai.utils.logging as logging
from parlai.core.metrics import SumMetric, AverageMetric, FairseqBleuMetric
from parlai.utils.fp16 import FP16SafeCrossEntropy
import parlai.utils.fsdp as fsdp_utils
from parlai.utils.torch import (
    neginf,
    total_parameters,
    trainable_parameters,
    PipelineHelper,
)

if torch.cuda.is_available():
    from parlai.ops.ngram_repeat_block import NGramRepeatBlock


class SearchBlocklist(object):
    """
    Search block list facilitates blocking ngrams from being generated.
    """

    def __init__(self, dict_agent: DictionaryAgent) -> None:
        self.dict = dict_agent
        self._phrases: Set[str] = set()
        self._phrase_ngrams: Dict[int, List[List[int]]] = {}

    def __bool__(self):
        return bool(self._phrases)

    def clear(self) -> None:
        self._phrases = set()
        self._phrase_ngrams = {}

    def _add_literal(self, phrase_literal: str):
        if phrase_literal in self._phrases:
            return
        ngram = self.dict.txt2vec(phrase_literal)
        self._phrases.add(phrase_literal)
        logging.debug(f"Adding '{phrase_literal}' to the beam block_list {ngram}")
        l = len(ngram)
        if l not in self._phrase_ngrams:
            self._phrase_ngrams[l] = []
        self._phrase_ngrams[l].append(ngram)

    def add(self, phrase: str):
        phrase = phrase.strip()
        if not phrase:
            return
        self._add_literal(phrase)
        self._add_literal(phrase + "s")
        self._add_literal(phrase.lower())
        self._add_literal(phrase.lower() + "s")
        self._add_literal(phrase.upper())
        self._add_literal(phrase.upper() + "S")
        self._add_literal(phrase.title())
        self._add_literal(phrase.title() + "S")
        self._add_literal(phrase[0].upper() + phrase[1:])
        self._add_literal(phrase[0].upper() + phrase[1:] + "s")
        self._add_literal(phrase[0].upper() + phrase[1:].lower())
        self._add_literal(phrase[0].upper() + phrase[1:].lower() + "s")

    def items(self) -> Iterable[Tuple[int, List[List[int]]]]:
        return self._phrase_ngrams.items()


TSType = TypeVar('TSType', bound='TreeSearch')


class TorchGeneratorModel(nn.Module, ABC):
    """
    Abstract TorchGeneratorModel.

    This interface expects you to implement model with the following reqs:

    :attribute model.encoder:
        takes input returns tuple (enc_out, enc_hidden, attn_mask)

    :attribute model.decoder:
        takes decoder params and returns decoder outputs after attn

    :attribute model.output:
        takes decoder outputs and returns distr over dictionary
    """

    def __init__(
        self,
        padding_idx=0,
        start_idx=1,
        end_idx=2,
        unknown_idx=3,
        input_dropout=0,
        longest_label=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.START_IDX = start_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

    def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
        """
        Return initial input to the decoder.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """
        return torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)

    def decode_forced(self, encoder_states, ys):
        """
        Decode with a fixed, true sequence, computing loss.

        Useful for training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(bsz, inputs)
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    @abstractmethod
    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        pass

    @abstractmethod
    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        pass

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        # TODO: get rid of longest_label
        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that during prediction
        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)

        # use teacher forcing
        scores, preds = self.decode_forced(encoder_states, ys)
        return scores, preds, encoder_states


class PPLMetric(AverageMetric):
    def value(self):
        return math.exp(super().value())


class TorchGeneratorAgent(TorchAgent, ABC):
    """
    Abstract Generator agent; only meant to be extended.

    TorchGeneratorAgent aims to handle much of the bookkeeping and infrastructure work
    for any generative models, like seq2seq or transformer. It implements the train_step
    and eval_step. The only requirement is that your model *must* be implemented with
    the TorchGeneratorModel interface.
    """

    @classmethod
    def upgrade_opt(cls, opt_from_disk: Opt):
        # call the parent upgrades
        opt_from_disk = super(TorchGeneratorAgent, cls).upgrade_opt(opt_from_disk)

        # 2019-08-18: Adding support for generation other than beam search
        # Previously, selecting --beam-size > 1 enabled beam search and == 1 was
        # greedy. New behavior is --inference greedy or --inference beam.
        if 'inference' not in opt_from_disk:
            assert 'beam_size' in opt_from_disk
            if opt_from_disk['beam_size'] == 1:
                method = 'greedy'
            else:
                method = 'beam'
            opt_from_disk['inference'] = method
            warn_once(f'Old model inference method inferred as {method}')

        # 2020-06-03: Changing "blacklist" --> "blocklist"
        if 'beam_blacklist_filename' in opt_from_disk:
            if opt_from_disk['beam_blacklist_filename'] is not None:
                opt_from_disk['beam_block_list_filename'] = opt_from_disk[
                    'beam_blacklist_filename'
                ]
            del opt_from_disk['beam_blacklist_filename']

        # 2020-08-04: Introduce full context beam blocking
        # Previous, specifying --beam-context-block-ngram > 1 would block
        # from generating ngrams from model's context, which is limited
        # by truncation parameters. Now, we block on full dialogue history.
        if 'beam_block_full_context' not in opt_from_disk:
            warn_once('Loading model with `--beam-block-full-context false`')
            opt_from_disk['beam_block_full_context'] = False

        return opt_from_disk

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command line arguments.
        """
        agent = parser.add_argument_group('Torch Generator Agent')
        agent.add_argument(
            '--beam-size',
            type=int,
            default=1,
            help='Beam size, if 1 then greedy search',
        )
        agent.add_argument(
            '--beam-min-length',
            type=int,
            default=1,
            help='Minimum length of prediction to be generated by the beam search',
        )
        agent.add_argument(
            '--beam-context-block-ngram',
            type=int,
            default=-1,
            help=(
                'Size n-grams to block in beam search from the context. val <= 0 '
                'implies no blocking'
            ),
        )
        agent.add_argument(
            '--beam-block-ngram',
            type=int,
            default=-1,
            help='Size n-grams to block in beam search. val <= 0 implies no blocking',
        )
        agent.add_argument(
            '--beam-block-full-context',
            type='bool',
            default=True,
            help='Block n-grams from the *full* history context. Specify False to block '
            'up to m tokens in the past, where m is truncation parameter for agent',
        )
        agent.add_argument(
            '--beam-length-penalty',
            type=float,
            default=0.65,
            help='Applies a length penalty. Set to 0 for no penalty.',
        )
        agent.add_argument(
            '--skip-generation',
            type='bool',
            default=False,
            hidden=True,
            help='Skip beam search. Useful for speeding up training, '
            'if perplexity is the validation metric.',
        )
        agent.add_argument(
            '--inference',
            choices={
                'beam',
                'greedy',
                'topk',
                'nucleus',
                'delayedbeam',
                'delayednucleusbeam',
                'factual_nucleus',
            },
            default='greedy',
            help='Generation algorithm',
        )
        agent.add_argument(
            '--topk', type=int, default=10, help='K used in Top K sampling'
        )
        agent.add_argument(
            '--topp', type=float, default=0.9, help='p used in nucleus sampling'
        )
        agent.add_argument(
            '--beam-delay', type=int, default=30, help='used in delayedbeam search'
        )
        agent.add_argument(
            '--lambda-decay',
            type=float,
            default=0.9,
            help='decay factor in factual nucleus sampling',
        )
        agent.add_argument(
            '--omega-bound',
            type=float,
            default=0.3,
            help='lower bound in factual nucleus sampling',
        )
        agent.add_argument(
            '--p-reset',
            type='bool',
            default=True,
            help='Whether to reset p value in factual nucleus at full stops',
        )
        agent.add_argument(
            '--beam-block-list-filename',
            type=str,
            default=None,
            help='Load a text file of hard blocks for beam search to never say.',
        )
        agent.add_argument(
            '--temperature',
            type=float,
            default=1.0,
            help='temperature to add during decoding',
        )
        agent.add_argument(
            '--compute-tokenized-bleu',
            type='bool',
            default=False,
            help='if true, compute tokenized bleu scores',
        )
        parser.add_argument(
            '--gpu-beam-blocking',
            type='bool',
            help='Set to use CUDA kernel for beam search ngram blocking',
            default=False,
        )
        parser.add_argument(
            '--verbose-topk',
            type=int,
            help='Return the topk logits in the act message, if verbose mode is set.',
            default=-1,
        )

        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return agent

    def __init__(self, opt: Opt, shared=None):
        init_model, is_finetune = self._get_init_model(opt, shared)
        super().__init__(opt, shared)

        self.beam_size = opt.get('beam_size', 1)
        self.beam_min_length = opt.get('beam_min_length', 1)
        self.beam_block_ngram = opt.get('beam_block_ngram', -1)
        self.beam_context_block_ngram = opt.get('beam_context_block_ngram', -1)
        self.beam_block_full_context = opt.get('beam_block_full_context', False)
        self.temperature = opt.get('temperature', 1.0)
        assert self.temperature > 0, '--temperature must be greater than 0'
        self.show_token_details = opt.get(
            'verbose', False
        ) or 'token_losses' in opt.get('display_add_fields', '')
        self.compute_tokenized_bleu = opt.get('compute_tokenized_bleu', False)
        self.beam_block_list: Optional[SearchBlocklist] = None

        if shared:
            # set up shared properties
            states = shared.get('states', {})
            self.beam_block_list = shared.get('beam_block_list')
        else:
            # this is not a shared instance of this class, so do full init
            self.criterion = self.build_criterion()

            self.model = self.build_model()
            with fsdp_utils.maybe_fsdp_wrap(opt):
                self.model = fsdp_utils.fsdp_wrap(self.model)
                if self.fp16 and not fsdp_utils.delay_halving(opt):
                    self.model = self.model.half()

            # load the block_list for beam search
            self.beam_block_list = self._load_beam_block_list()

            if self.model is None or self.criterion is None:
                raise AttributeError(
                    'build_model() and build_criterion() need to return the model or criterion'
                )
            if self.use_cuda:
                if self.model_parallel:
                    ph = PipelineHelper()
                    ph.check_compatibility(self.opt)
                    self.model = ph.make_parallel(self.model)
                else:
                    self.model.cuda()
                self.criterion.cuda()

            if not fsdp_utils.is_fsdp(self.model):
                sync_parameters(self.model)

            train_params = trainable_parameters(self.model)
            total_params = total_parameters(self.model)
            logging.info(
                f"Total parameters: {total_params:,d} ({train_params:,d} trainable)"
            )

            if init_model is not None:
                # load model parameters if available
                logging.info(f'Loading existing model params from {init_model}')
                states = self.load(init_model)
            else:
                states = {}

        if shared is not None:
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        elif self._should_initialize_optimizer():
            # do this regardless of share state, but don't
            was_reset = self.init_optim(
                [p for p in self.model.parameters() if p.requires_grad],
                optim_states=states.get('optimizer'),
                saved_optim_type=states.get('optimizer_type'),
                is_finetune=is_finetune,
            )
            if was_reset:
                logging.warning("Optimizer was reset. Also resetting LR scheduler.")
            self.build_lr_scheduler(states, hard_reset=is_finetune or was_reset)

        if (
            shared is None
            and is_distributed()
            and opt.get('ddp_backend', fsdp_utils.DEFAULT_DDP_BACKEND) == 'ddp'
        ):
            device_ids = None if self.model_parallel else [self.opt['gpu']]
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=device_ids, broadcast_buffers=False
            )

        self.reset()

    def build_criterion(self):
        """
        Construct and return the loss function.

        By default torch.nn.CrossEntropyLoss.

        If overridden, this model should produce a sum that can be used for a per-token loss.
        """
        if not self.fp16:
            return torch.nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, reduction='none'
            )
        else:
            # FP16 safe cross entropy (softmax done in FP32)
            return FP16SafeCrossEntropy(ignore_index=self.NULL_IDX, reduction='none')

    def _v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def set_interactive_mode(self, mode, shared=False):
        """
        Turn on interactive mode.
        """
        super().set_interactive_mode(mode, shared)
        if mode:
            self.skip_generation = False
        else:
            self.skip_generation = self.opt.get('skip_generation', False)

    def _cache_dummy_batch(self, batch: Batch):
        """
        Cache a batch to be used as a dummy during _fake_forward_backward_pass.
        """
        if not hasattr(self, '_dummy_batch'):
            self._dummy_batch = batch

    def _fake_forward_backward_pass(self):
        """
        Force a worker to synchronize with others in case of distributed mode.

        Necessary during recovery of OOMs to prevent hangs during the all-reduce of
        gradients.
        """
        try:
            self._control_local_metrics(disabled=True)
            loss = 0 * self.compute_loss(self._dummy_batch)
            self._control_local_metrics(enabled=True)
            self.backward(loss)
            self.buffer_initialized = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                m = (
                    'CUDA OOM: Lower batch size (-bs) from {} or lower '
                    ' max sequence length (-tr) from {}'
                    ''.format(self.opt['batchsize'], self.opt['truncate'])
                )
                raise RuntimeError(m)
            else:
                raise e

    def reset_metrics(self):
        """
        Reset metrics for reporting loss and perplexity.
        """
        super().reset_metrics()

    def share(self):
        """
        Share internal states between parent and child instances.
        """
        shared = super().share()
        shared['beam_block_list'] = self.beam_block_list
        if hasattr(self, 'optimizer'):
            shared['optimizer'] = self.optimizer
        return shared

    def vectorize(self, *args, **kwargs):
        """
        Override vectorize for generative models.
        """
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        return super().vectorize(*args, **kwargs)

    def batchify(self, obs_batch, sort=False):
        batch = super().batchify(obs_batch, sort=sort)
        if (
            self.beam_block_full_context
            and obs_batch
            and any('full_text_vec' in o for o in obs_batch)
        ):
            batch['full_text_vec'], _ = self._pad_tensor(
                [obs_batch[i].get('full_text_vec', []) for i in batch.valid_indices]
            )
        return batch

    def _model_input(self, batch):
        """
        Create the input (x) value for the model.

        Must return a tuple.  This will be passed directly into the model via
        `*args`, i.e.,

        >>> model(*_model_input(batch))

        This is intentionally overridable so that richer models can pass the
        additional inputs.
        """
        return (batch.text_vec,)

    def _encoder_input(self, batch):
        """
        Create the input (x) value for the encoder.

        Must return a tuple.  This will be passed directly into the encoder via
        `*args`, i.e.,

        >>> model.encoder(*_encoder_input(batch))

        This is intentionally overridable so that richer models can pass the
        additional inputs directly to the encoder.
        """
        return self._model_input(batch)

    def record_per_token_metrics(self, batch, loss_per_token):
        """
        Override this method for custom loss values that require loss_per_token.
        """
        pass

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.reshape(-1, scores.size(-1))
        loss_flattened = self.criterion(score_view, batch.label_vec.view(-1))
        loss_per_token = loss_flattened.view(scores.shape[:-1])
        notnull = batch.label_vec.ne(self.NULL_IDX)

        # save loss to metrics
        # cross entropy loss
        self.record_local_metric(
            'loss', AverageMetric.from_mask(loss_per_token, notnull)
        )
        # perplexity
        self.record_local_metric('ppl', PPLMetric.from_mask(loss_per_token, notnull))
        # token-wise accuracy
        self.record_local_metric(
            'token_acc', AverageMetric.from_mask(batch.label_vec == preds, notnull)
        )
        # utterance-wise exact match
        num_target_tokens = notnull.long().sum(dim=-1)
        num_tokens_correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)
        self.record_local_metric(
            'token_em', AverageMetric.many(num_tokens_correct == num_target_tokens)
        )
        self.record_per_token_metrics(batch, loss_per_token)

        # actually do backwards loss
        loss = loss_per_token.sum(dim=1)
        loss = loss.sum()
        loss /= num_target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        # cache a dummy batch in case we OOM and need to catch up
        self._cache_dummy_batch(batch)

        self.model.train()
        self.zero_grad()

        try:
            loss = self.compute_loss(batch)
            self.backward(loss)
            self.update_params()
            oom_sync = False
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                oom_sync = True
                logging.error(
                    'Ran out of memory, skipping batch. '
                    'if this happens frequently, decrease batchsize or '
                    'truncate the inputs to the model.'
                )
                self.global_metrics.add('skipped_batches', SumMetric(1))
            else:
                raise e

        if oom_sync:
            # moved outside of the try-except because the raised exception in scope
            # actually prevents from the data being freed, which can sometimes cause
            # us to OOM during our OOM handling.
            # https://github.com/pytorch/pytorch/issues/18853#issuecomment-583779161

            # gradients are synced on backward, now this model is going to be
            # out of sync! catch up with the other workers
            self._fake_forward_backward_pass()

    def _construct_label_token_losses(self, labels, model_output):
        # Get non-aggregated losses
        scores, _, _ = model_output
        score_view = scores.reshape(-1, scores.size(-1))
        losses = self.criterion(score_view, labels.view(-1)).view(len(labels), -1)

        # Zip decoded tokens with losses
        token_losses = []
        for i, label in enumerate(labels):
            token_losses.append(
                list(
                    zip(
                        [self.dict[token] for token in label.tolist()],
                        losses[i].tolist(),
                    )
                )
            )
        return token_losses

    def _construct_generated_token_details(self, tokens, tokens_metadata):
        tokens_as_txt = [self.dict[int(token)] for token in tokens]
        return list(zip(tokens_as_txt, tokens_metadata))

    def _compute_fairseq_bleu(self, batch: Batch, preds):
        """
        Compute BLEU score between text and label, using the FAIRSeq BLEU Scorer.

        :param batch:
            Batch of observations
        :param texts:
            list of string predictions
        """
        all_results = []
        label_vec = batch.label_vec
        assert label_vec is not None, "label_vec must exist for fairseq bleu"
        for i, t in enumerate(preds):
            result = FairseqBleuMetric.compute_many(
                t,
                label_vec[i].unsqueeze(0),
                pad_idx=self.NULL_IDX,
                end_idx=self.END_IDX,
                unk_idx=self.dict[self.dict.unk_token],
            )
            if result is None:
                return
            all_results.append(result)

        bleu_scores = list(zip(*all_results))
        for k in range(4):
            self.record_local_metric(f'fairseq_bleu{k + 1}', bleu_scores[k])

    def _add_generation_metrics(self, batch, preds):
        """
        Can be overridden to allow for some metrics on the generations calculated at
        eval.
        """
        self.record_local_metric(
            'gen_n_toks',
            AverageMetric.many([p.size(0) for p in preds], [1] * len(preds)),
        )

    def rank_eval_label_candidates(self, batch, batchsize):
        """
        Rank label_candidates during eval_step.

        Can be overridden to allow for different ways of ranking candidates. Must have
        `--rank-candidates` set to True. By default, we roughly compute PPL to rank the
        candidates.
        """
        # compute roughly ppl to rank candidates
        cand_choices = []
        cand_choices_scores = []
        encoder_states = self.model.encoder(*self._encoder_input(batch))
        for i in range(batchsize):
            num_cands = len(batch.candidate_vecs[i])
            enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
            cands, _ = self._pad_tensor(batch.candidate_vecs[i], is_label=True)
            cands = cands.to(batch.text_vec.device)
            scores, _ = self.model.decode_forced(enc, cands)
            score_view = scores.reshape(num_cands * cands.size(1), -1)
            cand_losses = F.cross_entropy(
                score_view, cands.view(-1), reduction='none'
            ).view(num_cands, cands.size(1))
            # now cand_losses is cands x seqlen size, but we still need to
            # check padding and such
            mask = (cands != self.NULL_IDX).float()
            cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            sorted_scores, ordering = cand_scores.sort()
            cand_choices.append([batch.candidates[i][o] for o in ordering])
            cand_choices_scores.append(sorted_scores.tolist())

        return cand_choices, cand_choices_scores

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None
        logits = None
        logit_inds = None
        text_token_info = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.show_token_details:
                token_losses = self._construct_label_token_losses(
                    batch.label_vec, model_output
                )
                logits = model_output[0]
                k = self.opt['verbose_topk']
                if k != -1:
                    tk = torch.topk(logits, k, dim=2)
                    logits = tk.values
                    logit_inds = tk.indices
                    if isinstance(logit_inds, torch.Tensor):
                        logit_inds = logit_inds.cpu().numpy().tolist()
                if isinstance(logits, torch.Tensor):
                    logits = logits.cpu().numpy().tolist()

        beam_preds_scores = None
        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            prefix_tokens = self.get_prefix_tokens(batch)
            beam_preds_scores, beams = self._generate(
                batch, self.beam_size, maxlen, prefix_tokens=prefix_tokens
            )
            preds, _, _ = zip(*beam_preds_scores)
            self._add_generation_metrics(batch, preds)

            # bsz x beamsize
            beam_texts: List[List[Tuple[str, float]]] = []
            beam_texts_token_info: List[List[List[Tuple]]] = []
            for beam in beams:
                beam_texts.append([])
                if self.show_token_details:
                    beam_texts_token_info.append([])

                for tokens, score, token_metadata in beam.get_rescored_finished():
                    try:
                        if self.show_token_details:
                            beam_texts_token_info[-1].append(
                                self._construct_generated_token_details(
                                    tokens, token_metadata
                                )
                            )
                        beam_texts[-1].append((self._v2t(tokens), score.item()))
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

        cand_choices = None
        cand_scores = None
        if self.rank_candidates:
            cand_choices, cand_scores = self.rank_eval_label_candidates(batch, bsz)

        text = (
            [self._v2t(pred_data[0]) for pred_data in beam_preds_scores]
            if beam_preds_scores is not None
            else None
        )

        if self.show_token_details and beam_preds_scores is not None:
            text_token_info = []
            for beam_text_token_info in beam_texts_token_info:
                text_token_info.append(beam_text_token_info[0])

        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
        retval = Output(
            text,
            cand_choices,
            token_losses=token_losses,
            cand_scores=cand_scores,
            logits=logits,
            logit_inds=logit_inds,
        )

        if not self.skip_generation:
            retval.beam_texts = beam_texts
            retval.beam_texts_token_info = beam_texts_token_info
            retval.text_token_info = text_token_info
        return retval

    def _treesearch_factory(self, device, verbose=False):
        method = self.opt.get('inference', 'greedy')
        beam_size = self.opt.get('beam_size', 1)
        if method == 'greedy':
            return GreedySearch(
                beam_size,
                min_length=0,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                gpu_beam_blocking=self.opt.get('gpu_beam_blocking', False),
            )
        elif method == 'beam':
            return BeamSearch(
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                gpu_beam_blocking=self.opt.get('gpu_beam_blocking', False),
            )
        elif method == 'delayedbeam':
            return DelayedBeamSearch(
                self.opt['topk'],
                self.opt['beam_delay'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                gpu_beam_blocking=self.opt.get('gpu_beam_blocking', False),
            )
        elif method == 'delayednucleusbeam':
            return DelayedNucleusBeamSearch(
                self.opt['topp'],
                self.opt['beam_delay'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                gpu_beam_blocking=self.opt.get('gpu_beam_blocking', False),
            )
        elif method == 'topk':
            return TopKSampling(
                self.opt['topk'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                gpu_beam_blocking=self.opt.get('gpu_beam_blocking', False),
            )
        elif method == 'nucleus':
            return NucleusSampling(
                self.opt['topp'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                gpu_beam_blocking=self.opt.get('gpu_beam_blocking', False),
            )
        elif method == 'factual_nucleus':
            return FactualNucleusSampling(
                self.opt['topp'],
                self.opt['lambda_decay'],
                self.opt['omega_bound'],
                self.opt['p_reset'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                gpu_beam_blocking=self.opt.get('gpu_beam_blocking', False),
                dict=self.dict,
            )
        else:
            raise ValueError(f"Can't use inference method {method}")

    def _get_batch_context(self, batch):
        """
        Version of TGA._get_context() that operates on full batches for speed.
        """
        if hasattr(self, '_get_context'):
            # Warn users that have subclassed with '_get_gontext
            warn_once(
                "WARNING: TGA._get_context() has been removed, use TGA.get_batch_context() instead"
            )

        if self.beam_context_block_ngram <= 0:
            # We aren't context blocking, return empty tensor of the correct size
            return torch.zeros(batch.batchsize, 0, dtype=torch.long)

        ctxt = batch.text_vec
        if self.beam_block_full_context:
            ctxt = batch.full_text_vec
        return ctxt

    def _get_initial_decoder_input(
        self, bsz: int, beam_size: int, dev: torch.device
    ) -> torch.LongTensor:
        """
        Return initial input to the decoder.

        :param bsz:
            batchsize
        :param beam_size:
            beam size
        :param dev:
            device to send input to.

        :return initial_input:
            initial input for the decoder
        """
        return (
            torch.LongTensor([self.START_IDX])  # type: ignore
            .expand(bsz * beam_size, 1)
            .to(dev)
        )

    def _get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
        incr_state_inds: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Return next decoder input.

        :param prev_input:
            previous input to decoder
        :param selection:
            token selections for current timestep
        :param inds:
            incremental state indices

        :return decoder input:
            return decoder input for next timestep
        """
        prev_input = torch.index_select(prev_input, 0, incr_state_inds)
        decoder_input = torch.cat([prev_input, selection], dim=-1)
        return decoder_input

    def get_prefix_tokens(self, batch: Batch) -> Optional[torch.LongTensor]:
        """
        Set prefix tokens to seed decoding at generation time.

        By default, we do not utilize prefix tokens, but this is
        left overridable by child classes.

        Returned tensor should be of dimension bsz x len(prefix)
        """
        return None

    def _generation_activation(self, score: torch.Tensor) -> torch.float32:
        return F.log_softmax(score, dim=-1, dtype=torch.float32)

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence
        :param prefix_tokens:
            if given, a tensor of tokens that must begin the decoded sequence.

        :return:
            tuple (beam_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score, token_metadata) tuples for each sample in
              Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        if batch.text_vec is not None:
            batchsize = batch.batchsize
            batch_context_list = self._get_batch_context(batch)
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                .set_batch_context(
                    batch_context_list,
                    batch_idx,
                    self.opt.get('gpu_beam_blocking', False),
                )
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                for _ in range(bsz)
            ]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = self._generation_activation(score)  # type: ignore
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts]
                prefix_mask = torch.ones_like(score, dtype=torch.bool)
                prefix_mask[
                    :, :, prefix_toks
                ] = False  # everything except prefix toks should be neginf
                score[prefix_mask] = neginf(score.dtype)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i], _ts)
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams

    def _load_beam_block_list(self) -> SearchBlocklist:
        """
        Load the beam block_list.

        :return: a dict mapping ngram length to different ngrams
        """
        block_list = SearchBlocklist(self.dict)
        if not self.opt.get('beam_block_list_filename'):
            return block_list

        block_list_fn = self.opt['beam_block_list_filename']
        try:
            with PathManager.open(block_list_fn) as f:
                for line in f:
                    block_list.add(line.strip())
        except IOError:
            logging.error(
                f"Could not load beam block_list {block_list_fn}, using empty block_list."
            )
        return block_list


class _HypothesisTail(object):
    """
    Hold some bookkeeping about a hypothesis.
    """

    # use slots because we don't want dynamic attributes here
    __slots__ = ['timestep', 'hypid', 'score', 'tokenid', 'token_details']

    def __init__(self, timestep, hypid, score, tokenid, token_details):
        self.timestep = timestep
        self.hypid = hypid
        self.score = score
        self.tokenid = tokenid
        self.token_details = token_details


class _PathSelectionTokenDetails(TypedDict, total=False):
    token_logprob: float  # conditional log-probability of token (normalized)
    token_rank: int  # rank of token in conditional distribution


class _PathSelection(object):
    """
    Output of TreeSearch:select_paths.

    Represents output of path selection process.
    """

    __slots__ = ['hypothesis_ids', 'token_ids', 'scores', 'token_details']

    def __init__(
        self,
        hypothesis_ids,
        token_ids,
        scores,
        token_details: Optional[List[_PathSelectionTokenDetails]] = None,
    ):
        self.hypothesis_ids = hypothesis_ids
        self.token_ids = token_ids
        self.scores = scores
        self.token_details = token_details  # length equal to beam size


class TreeSearch(object):
    """
    Abstract Tree Search class.

    It keeps information about beam_size concurrent, developing hypotheses. Concrete
    implementations make choices about which token to explore next at each point in the
    tree. Different choices result in different generation algorithms.
    """

    def __init__(
        self,
        beam_size,
        block_ngram=-1,
        context_block_ngram=-1,
        padding_token=0,
        bos_token=1,
        eos_token=2,
        min_length=3,
        device='cpu',
        length_penalty=0.65,
        verbose=False,
        gpu_beam_blocking=False,
        dict=None,
    ):
        """
        Instantiate Beam object.

        :param beam_size:
            number of hypothesis in the beam
        :param block_ngram:
            size of ngrams to block.
        :param context_block_ngram:
            size of context ngrams to block
        :param padding_token:
            padding token ID
        :param bos_token:
            beginning of sentence token ID
        :param eos_token:
            end of sentence token ID
        :param min_length:
            minimum length of the predicted sequence
        :param device:
            What device to use for computations
        :param dict:
            dictionary, if necessary
        """
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.block_ngram = block_ngram
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.context = None
        self.context_block_ngram = context_block_ngram
        self.block_list: Optional[SearchBlocklist] = None
        self.device = device
        # recent score for each hypo in the beam
        self.scores = None
        # self.scores values per each time step
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(self.bos).to(self.device)
        ]

        self.verbose = verbose
        # (beam size, sample length) list of lists containing token-level data for each token in each hypo in the beam
        self.token_details: Optional[List[List[_PathSelectionTokenDetails]]] = None
        if self.verbose:
            self.token_details = []
            for _ in range(self.beam_size):
                self.token_details.append([{"token_logprob": 0.0, "token_rank": 0}])

        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.gpu_beam_blocking = gpu_beam_blocking and torch.cuda.is_available()
        self.partial_hyps = torch.tensor([[self.bos] for i in range(beam_size)])
        if self.gpu_beam_blocking:
            self.partial_hyps = self.partial_hyps.cuda()
            self.no_repeat_ngram_op = NGramRepeatBlock()

    def set_context(self: TSType, context: torch.LongTensor) -> TSType:
        """
        Set the internal context representation and return self.

        :param context:
            a LongTensor representing the input context; used for context
            ngram blocking, if supplied
        """
        self.context = context.tolist()
        return self

    def set_batch_context(
        self: TSType,
        batch_context_list: torch.LongTensor,
        batch_idx: int,
        gpu_beam_blocking: bool,
    ) -> TSType:
        """
        Version of .set_context() that operates on a single element of a batch.

        Set the internal context representation and return self.

        :param batch_context_list:
            a list of lists, each one containing the context for one member of the batch
        :param batch_idx:
            index of the batch
        :param gpu_beam_blocking:
            whether we are using gpu kernel for beam blocking, if so return a tensor,
            else return a list.
        """
        context = batch_context_list[batch_idx]
        self.context = context if gpu_beam_blocking else context.tolist()
        return self

    def set_block_list(self: TSType, block_list: Optional[SearchBlocklist]) -> TSType:
        self.block_list = block_list
        return self

    def get_output_from_current_step(self):
        """
        Get the output at the current step.
        """
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """
        Get the backtrack at the current step.
        """
        return self.bookkeep[-1]

    @abstractmethod
    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        """
        Select the next vocabulary item in these beams.

        :param logprobs:
            a (beamsize x vocab) tensor of log probabilities. If this is the first
            turn in the dialogue, it will be a (1 x vocab) tensor.
        :param prior_scores:
            a (beamsize) tensor of weights with the cumulative running
            log-probability of each beam. If the first turn, it will be a (1) tensor.
        :param current_length:
            the current length in tokens
        :return:
            a {hypothesis_ids, token_ids, scores, token_details} , where:

            - hypothesis_ids is a LongTensor of hypotheses we're extending. May have
              repeats, but should always be (beamsize) long.
            - token_ids is a (beamsize) LongTensor of next-token choices for
              each of the hypotheses.
            - scores is a (beamsize) Tensor with the updated cumulative log-probs
              of each beam.
            - token_details is a (beamsize) list of objects with with metadata about each generated token.
        """
        pass

    def _block_ngrams(
        self,
        ngram_size: int,
        logprobs: torch.Tensor,
        step: int = 0,
        if_context_blocking=False,
    ):
        """
        Hard block ngrams from the logprobs.

        :param ngram_size:
            The length of ngrams to block. Must be > 0.
        :param logprobs:
            Float or HalfTensor, representing the log-probabilities. This is
            modified in place.
        :param step:
            current step on generating utterances
        :param if_context_blocking:
            whether we are doing context blocking
        """
        # gpu beam blocking
        if self.gpu_beam_blocking:
            context = self.context if if_context_blocking else None
            logprobs = self.no_repeat_ngram_op(
                hypothesis=self.partial_hyps,
                context=context,
                lprobs=logprobs,
                bsz=1,
                step=step,
                beam_size=self.beam_size,
                no_repeat_ngram_size=ngram_size,
                if_context_blocking=if_context_blocking,
            )
            return logprobs

        # cpu beam blocking
        for beam_id, hyp in enumerate(self.partial_hyps.tolist()):
            if len(hyp) < ngram_size - 1:
                continue
            source = hyp if if_context_blocking is False else self.context
            prefix = hyp[-(ngram_size - 1) :]
            for i in range(len(source) - ngram_size + 1):
                ngram = source[i : i + ngram_size]
                if ngram_size == 1 or prefix == ngram[:-1]:
                    logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
        return logprobs

    def _block_block_list(self, logprobs: torch.Tensor) -> torch.Tensor:
        if self.block_list is None:
            return logprobs

        for beam_id, hyp in enumerate(self.partial_hyps.tolist()):
            for ngram_size, bad_ngrams in self.block_list.items():
                prefix = hyp[-(ngram_size - 1) :]
                for ngram in bad_ngrams:
                    if (ngram_size == 1) or prefix == ngram[:-1]:
                        logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
        return logprobs

    def advance(self, logprobs, step):
        """
        Advance the beam one step.
        """
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(logprobs.size(0)):
                logprobs[hyp_id][self.eos] = neginf(logprobs.dtype)

        if self.scores is None:
            self.scores = torch.zeros(1).type_as(logprobs).to(logprobs.device)

        # penalize hypotheses ending in EOS on the prior scores (self.scores) level
        # this is related to search which uses prior scores (self.scores) (e.g. beam)
        for hyp_id, token in enumerate(self.outputs[-1]):
            if token == self.eos:
                self.scores[hyp_id] = neginf(self.scores.dtype)

        # beam blocking
        if self.block_ngram > 0:
            # self blocking
            logprobs = self._block_ngrams(
                ngram_size=self.block_ngram,
                logprobs=logprobs,
                step=step,
                if_context_blocking=False,
            )

        logprobs = self._block_block_list(logprobs)

        if self.context_block_ngram > 0:
            if self.context is None:
                raise ValueError(
                    "Must use TreeSearch.set_context to use context blocking."
                )
            # context blocking
            logprobs = self._block_ngrams(
                ngram_size=self.context_block_ngram,
                logprobs=logprobs,
                step=step,
                if_context_blocking=True,
            )

        path_selection = self.select_paths(logprobs, self.scores, current_length)
        self.scores = path_selection.scores
        # use clone() here to ensure that self.all_scores will not be changed
        # later due to any penalties to self.scores
        self.all_scores.append(self.scores.clone())

        self.outputs.append(path_selection.token_ids)
        self.bookkeep.append(path_selection.hypothesis_ids)

        # this checking for device seems suboptimal
        # might need to change later
        if self.partial_hyps.get_device() == -1:
            hyp_device = 'cpu'
        else:
            hyp_device = self.partial_hyps.get_device()
        self.partial_hyps = torch.cat(
            (
                self.partial_hyps[path_selection.hypothesis_ids.long().to(hyp_device)],
                path_selection.token_ids.view(path_selection.token_ids.shape[0], -1).to(
                    hyp_device
                ),
            ),
            1,
        )

        if self.verbose:
            assert path_selection.token_details
            assert self.token_details
            for i in range(self.beam_size):
                self.token_details[i].append(path_selection.token_details[i])

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                if self.scores[hypid] <= neginf(self.scores.dtype):
                    continue
                #  this is finished hypo, adding to finished

                eostail = _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.all_scores[-1][hypid],
                    tokenid=self.eos,
                    token_details=self.token_details[hypid][-1]
                    if self.token_details is not None
                    else None,
                )
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def is_done(self):
        """
        Return whether beam search is complete.
        """
        return self.eos_top and self.n_best_counter >= self.beam_size

    def _find_ngrams(self, input_list, n):
        """
        Find ngrams of size n in input list.
        """
        return list(zip(*[input_list[i:] for i in range(n)]))

    def _get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep:
            timestep with range up to len(self.outputs) - 1

        :param hyp_id:
            id with range up to beam_size - 1

        :return:
            hypothesis sequence
        """
        hyp_idx = []
        endback = hypothesis_tail.hypid

        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(
                _HypothesisTail(
                    timestep=i,
                    hypid=endback,
                    score=self.all_scores[i][endback],
                    tokenid=self.outputs[i][endback],
                    token_details=self.token_details[endback][i]
                    if self.token_details is not None
                    else None,
                )
            )
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    def _get_pretty_hypothesis(self, list_of_hypotails):
        """
        Return hypothesis as a tensor of token ids.
        """
        return torch.stack([ht.tokenid for ht in reversed(list_of_hypotails)])

    def get_rescored_finished(self, n_best=None):
        """
        Return finished hypotheses according to adjusted scores.

        Score adjustment is done according to the Google NMT paper, which
        penalizes long utterances.

        :param n_best:
            number of finalized hypotheses to return

        :return:
            list of (tokens, score, token_metadata) 3-tuples, in sorted order, where:
              - tokens is a tensor of token ids
              - score is the adjusted log probability of the entire utterance
              - token_metadata dictionary:
                    token_logprobs -> a tensor of conditional log probabilities of tokens
                    token_ranks -> a tensor of ranks of tokens in vocabulator, by probability, when sampled
        """
        # if we never actually finished, force one
        if not self.finished:
            self.outputs[-1][0] = self.eos
            self.finished.append(
                _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=0,
                    score=self.all_scores[-1][0],
                    tokenid=self.outputs[-1][0],
                    token_details=self.token_details[0][-1]
                    if self.token_details is not None
                    else None,
                )
            )

        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, self.length_penalty)
            rescored_finished.append(
                _HypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid,
                    token_details=finished_item.token_details,
                )
            )

        # Note: beam size is almost always pretty small, so sorting is cheap enough
        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        n_best_list = []
        for hyp in srted:
            hyp_data = self._get_hyp_from_finished(hyp)
            token_ids = self._get_pretty_hypothesis(hyp_data)
            token_metadata = (
                [tok.token_details for tok in reversed(hyp_data)]
                if self.verbose
                else None
            )
            n_best_list.append((token_ids, hyp.score, token_metadata))

        # check that there is at least one finished candidate
        # and assert that each of them contains only one EOS
        assert (
            len(n_best_list) >= 1
        ), f'TreeSearch returned {len(n_best_list)} candidates, must be >= 1'
        for (pred, score, _) in n_best_list:
            assert (pred == self.eos).sum() == 1, (
                f'TreeSearch returned a finalized hypo with multiple end tokens '
                f'with score {score.item():.2f}'
            )

        return n_best_list


class GreedySearch(TreeSearch):
    """
    Greedy search.

    Picks the highest probability utterance at each step.  Only works with
    --beam-size 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.beam_size != 1:
            raise ValueError('Greedy search can only be run with beam size 1.')

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        tok_scores, tok_ids = logprobs.max(1)
        best_scores = tok_scores + prior_scores
        hyp_ids = torch.arange(logprobs.size(0), device=logprobs.device)

        token_details: Optional[List[_PathSelectionTokenDetails]] = None
        if self.verbose:
            tok_logprob = torch.softmax(logprobs.view(-1), dim=-1)[tok_ids].log().item()
            tok_rank = 0
            token_details = [{"token_logprob": tok_logprob, "token_rank": tok_rank}]

        return _PathSelection(
            hypothesis_ids=hyp_ids,
            token_ids=tok_ids,
            scores=best_scores,
            token_details=token_details,
        )


class BeamSearch(TreeSearch):
    """
    Beam search.
    """

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        """
        Select the next vocabulary item in these beams.
        """
        # if numel is 1, then this is the first time step, only one hyp is expanded
        if prior_scores.numel() == 1:
            logprobs = logprobs[0:1]

        # beam search actually looks over all hypotheses together so we flatten
        beam_scores = logprobs + prior_scores.unsqueeze(1).expand_as(logprobs)
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_idxs = torch.topk(flat_beam_scores, self.beam_size, dim=-1)
        voc_size = logprobs.size(-1)

        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = torch.div(best_idxs, voc_size, rounding_mode='trunc')
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        token_details: Optional[List[_PathSelectionTokenDetails]] = None
        if self.verbose:
            probs = torch.softmax(logprobs, dim=-1)
            tok_probs = (
                torch.index_select(probs, 0, hyp_ids)
                .gather(1, tok_ids.unsqueeze(1))
                .view(-1)
            )
            tok_ranks = (
                probs.argsort(1, descending=True)
                .argsort(1)
                .view(-1)
                .gather(0, best_idxs)
            )

            token_details = []

            for tok_logprob, tok_rank in zip(
                tok_probs.log().cpu().numpy(), tok_ranks.cpu().numpy()
            ):
                token_details.append(
                    {
                        "token_logprob": tok_logprob.item(),
                        "token_rank": int(tok_rank.item()),
                    }
                )

        return _PathSelection(
            hypothesis_ids=hyp_ids,
            token_ids=tok_ids,
            scores=best_scores,
            token_details=token_details,
        )


class DelayedBeamSearch(TreeSearch):
    """
    DelayedBeam: Top-K sampling followed by beam search (Massarelli et al., 2019).

    Samples from a truncated distribution where only the most probable K words
    are considered at each time for the first N tokens, then switches to beam
    after N steps.

    See https://arxiv.org/abs/1911.03587 for details.
    """

    def __init__(self, k, delay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.delay = delay

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        if current_length < self.delay:
            return TopKSampling.select_paths(
                self, logprobs, prior_scores, current_length
            )
        else:
            return BeamSearch.select_paths(self, logprobs, prior_scores, current_length)


class DelayedNucleusBeamSearch(TreeSearch):
    def __init__(self, p, delay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.delay = delay

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        if current_length < self.delay:
            return NucleusSampling.select_paths(
                self, logprobs, prior_scores, current_length
            )
        else:
            return BeamSearch.select_paths(self, logprobs, prior_scores, current_length)


class TopKSampling(TreeSearch):
    """
    Top-K sampling (Fan et al., 2018).

    Samples from a truncated distribution where only the most probable K words
    are considered at each time.

    Typical values of k are 2, 10, 50.

    See https://arxiv.org/abs/1805.04833 for details.
    """

    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        values, indices = logprobs.topk(self.k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        choices = torch.multinomial(probs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = indices[hyp_ids, choices]
        scores = values[hyp_ids, choices]
        best_scores = prior_scores.expand_as(scores) + scores

        token_details: Optional[List[_PathSelectionTokenDetails]] = None
        if self.verbose:
            tok_logprobs = probs[hyp_ids, choices].log().view(-1).cpu().numpy()
            tok_ranks = choices.view(-1).cpu().numpy()
            token_details = []

            for tok_logprob, tok_rank in zip(tok_logprobs, tok_ranks):
                token_details.append(
                    {"token_logprob": tok_logprob, "token_rank": int(tok_rank)}
                )

        return _PathSelection(
            hypothesis_ids=hyp_ids,
            token_ids=tok_ids,
            scores=best_scores,
            token_details=token_details,
        )


class NucleusSampling(TreeSearch):
    """
    Nucelus, aka top-p sampling (Holtzman et al., 2019).

    Samples from a truncated distribution which covers a fixed CDF proportion
    of the original distribution.

    Typical values of p are 0.3 and 0.9.

    See https://arxiv.org/abs/1904.09751 for details.
    """

    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def update_p(self, tokens: torch.Tensor):
        pass

    def get_mask(self, sorted_probs: torch.Tensor) -> torch.Tensor:
        """
        Get probability mask.

        :param sorted_probs:
            sorted probabilities

        :return mask:
            mask out tokens below the p value when sampling.
        """
        return (sorted_probs.cumsum(dim=-1) - sorted_probs) >= self.p

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        # Unlike the other treesearch methods, we have to switch to linspace
        # for the probabilities in order to compute the CDF.
        probs = torch.softmax(logprobs, dim=-1)
        sprobs, sinds = probs.sort(dim=-1, descending=True)
        # The subtraction here is to get the exclusive prefix sum,
        # to guarantee the first element is not masked
        mask = self.get_mask(sprobs)
        trunc_sprobs = sprobs.detach().clone()
        trunc_sprobs[mask] = 0
        trunc_sprobs.div_(trunc_sprobs.sum(dim=-1).unsqueeze(1))
        choices = torch.multinomial(trunc_sprobs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = sinds[hyp_ids, choices]
        self.update_p(tok_ids)
        # Convert back to logspace.
        scores = trunc_sprobs[hyp_ids, choices].log()
        best_scores = prior_scores.expand_as(scores) + scores

        token_details: Optional[List[_PathSelectionTokenDetails]] = None
        if self.verbose:
            tok_logprobs = sprobs[hyp_ids, choices].log().view(-1).cpu().numpy()
            tok_ranks = choices.view(-1).cpu().numpy()
            token_details = []

            for tok_logprob, tok_rank in zip(tok_logprobs, tok_ranks):
                token_details.append(
                    {"token_logprob": tok_logprob, "token_rank": int(tok_rank)}
                )

        return _PathSelection(
            hypothesis_ids=hyp_ids,
            token_ids=tok_ids,
            scores=best_scores,
            token_details=token_details,
        )


class FactualNucleusSampling(NucleusSampling):
    """
    Factual Nucleus Sampling.

    See https://arxiv.org/pdf/2206.04624.pdf for more information
    """

    def __init__(
        self, p, lambda_decay, omega_bound, p_reset, beam_size, *args, **kwargs
    ):
        super().__init__(p, beam_size, *args, **kwargs)
        assert 'dict' in kwargs
        buffer = torch.zeros(beam_size)
        self.p = buffer.clone().fill_(p).unsqueeze(1)
        self.init_p = self.p.clone()
        self.lambda_decay = lambda_decay
        self.omega_bound = torch.tensor(omega_bound)
        self.toks_since_reset = buffer.clone()
        self.full_stop_list = torch.tensor(
            [kwargs['dict'].txt2vec(w) for w in ['.', '?', '!']]
        )
        self.p_reset = p_reset

    def update_p(self, tokens: torch.Tensor):
        """
        Updates sampling P value according to tokens generated.

        When tokens are *not* punctuation, p is decayed by lambda_decay factor.

        Otherwise, we reset the p value.

        :param tokens:
            sampled tokens.
        """
        for i, t in enumerate(tokens):
            if self.full_stop_list.to(tokens.device).eq(t).sum() > 0:
                self.toks_since_reset[i] = 0
            else:
                self.toks_since_reset[i] += 1
            decay_factor = max(0, self.toks_since_reset[i] - 1)
            self.p[i] = torch.max(
                self.omega_bound, self.init_p[i] * (self.lambda_decay ** (decay_factor))
            )

    def get_mask(self, sorted_probs: torch.Tensor) -> torch.Tensor:
        return (sorted_probs.cumsum(dim=-1) - sorted_probs) >= self.p.expand(
            sorted_probs.size()
        ).to(sorted_probs.device)
