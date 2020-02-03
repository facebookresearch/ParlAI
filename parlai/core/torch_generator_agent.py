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

from abc import ABC, abstractmethod
from typing import TypeVar, List
import math
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.core.torch_agent import TorchAgent, Batch, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.utils.torch import neginf, FP16SafeCrossEntropy


try:
    from nltk.translate import bleu_score as nltkbleu

except ImportError:
    nltkbleu = None

try:
    from fairseq import bleu as fairseq_bleu

except ImportError:
    fairseq_bleu = None


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
    ):
        super().__init__()
        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

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
        inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)
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


class TorchGeneratorAgent(TorchAgent, ABC):
    """
    Abstract Generator agent; only meant to be extended.

    TorchGeneratorAgent aims to handle much of the bookkeeping and infrastructure work
    for any generative models, like seq2seq or transformer. It implements the train_step
    and eval_step. The only requirement is that your model *must* implemented the
    interface TorchGeneratorModel interface.
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
        return opt_from_disk

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command line arguments.
        """
        agent = argparser.add_argument_group('Torch Generator Agent')
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
            choices={'beam', 'greedy', 'topk', 'nucleus'},
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
            '--compute-tokenized-bleu',
            type='bool',
            default=False,
            help='if true, compute tokenized bleu scores',
        )

        super(TorchGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt: Opt, shared=None):
        init_model, is_finetune = self._get_init_model(opt, shared)
        super().__init__(opt, shared)

        self.beam_size = opt.get('beam_size', 1)
        self.beam_min_length = opt.get('beam_min_length', 1)
        self.beam_block_ngram = opt.get('beam_block_ngram', -1)
        self.beam_context_block_ngram = opt.get('beam_context_block_ngram', -1)
        self.output_token_losses = opt.get('verbose', False)
        self.compute_tokenized_bleu = opt.get('compute_tokenized_bleu', False)

        if shared:
            # set up shared properties
            states = shared.get('states', {})
            self.fairseq_bleu_scorer = shared['fairseq_bleu_scorer']
            self.ntlk_bleu_scores = shared['nltk_bleu_scores']
        else:
            # Note: we cannot change the type of metrics ahead of time, so you
            # should correctly initialize to floats or ints here
            self.metrics['nll_loss'] = 0.0
            self.metrics['loss'] = 0.0
            self.metrics['correct_tokens'] = 0
            self.metrics['total_skipped_batches'] = 0

            # this is not a shared instance of this class, so do full init
            self.criterion = self.build_criterion()
            self._init_and_reset_bleu_scorers()
            # ensure all distributed copies will always be in sync
            self.model = self.build_model()

            if self.model is None or self.criterion is None:
                raise AttributeError(
                    'build_model() and build_criterion() need to return the model or criterion'
                )
            if self.use_cuda:
                self.model.cuda()
                self.criterion.cuda()

            sync_parameters(self.model)
            print("Total parameters: {}".format(self._total_parameters()))
            print("Trainable parameters:  {}".format(self._trainable_parameters()))

            if self.fp16:
                self.model = self.model.half()

            if init_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]' ''.format(init_model))
                states = self.load(init_model)
            else:
                states = {}

        if (
            # only build an optimizer if we're training
            'train' in opt.get('datatype', '')
            # and this is the main model, or on every fork if doing hogwild
            and (shared is None or self.opt.get('numthreads', 1) > 1)
        ):
            # do this regardless of share state, but don't
            self.init_optim(
                [p for p in self.model.parameters() if p.requires_grad],
                optim_states=states.get('optimizer'),
                saved_optim_type=states.get('optimizer_type'),
            )
            self.build_lr_scheduler(states, hard_reset=is_finetune)

        if shared is None and is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.opt['gpu']], broadcast_buffers=False
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

    def _dummy_batch(self, batchsize, maxlen):
        """
        Create a dummy batch.

        This is used to preinitialize the cuda buffer, or otherwise force a
        null backward pass after an OOM.

        If your model uses additional inputs beyond text_vec and label_vec,
        you will need to override it to add additional fields.
        """
        return Batch(
            text_vec=torch.ones(batchsize, maxlen).long().cuda(),
            label_vec=torch.ones(batchsize, 2).long().cuda(),
            text_lengths=[maxlen] * batchsize,
        )

    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        """
        Pre-initialize CUDA buffer by doing fake forward pass.
        """
        if self.use_cuda and (force or not hasattr(self, 'buffer_initialized')):
            try:
                loss = self.compute_loss(self._dummy_batch(batchsize, maxlen))
                self.backward(loss)
                self.buffer_initialized = True
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    m = (
                        'CUDA OOM: Lower batch size (-bs) from {} or lower '
                        ' max sequence length (-tr) from {}'
                        ''.format(batchsize, maxlen)
                    )
                    raise RuntimeError(m)
                else:
                    raise e

    def _init_and_reset_bleu_scorers(self):
        if not hasattr(self, 'fairseq_bleu_scorer'):
            if fairseq_bleu is None:
                self.fairseq_bleu_scorer = None
            else:
                self.fairseq_bleu_scorer = fairseq_bleu.Scorer(
                    self.NULL_IDX, self.END_IDX, self.dict[self.dict.unk_token]
                )
        self.nltk_bleu_scores = {
            f'bleu-{i}': {'score': 0, 'cnt': 0} for i in range(1, 5)
        }

    def reset_metrics(self):
        """
        Reset metrics for reporting loss and perplexity.
        """
        super().reset_metrics()
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here
        self.metrics['loss'] = 0.0
        self.metrics['nll_loss'] = 0.0
        self.metrics['num_tokens'] = 0
        self.metrics['correct_tokens'] = 0
        self._init_and_reset_bleu_scorers()

    def share(self):
        """
        Share internal states between parent and child instances.
        """
        shared = super().share()
        if self.opt.get('numthreads', 1) > 1:
            shared['states'] = {  # don't share optimizer states
                'optimizer_type': self.opt['optimizer']
            }
        shared['fairseq_bleu_scorer'] = self.fairseq_bleu_scorer
        shared['nltk_bleu_scores'] = self.nltk_bleu_scores
        return shared

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may differ
        from a truly independent measurement.

        Additionally report tokenized bleu scores, if desired.
        """
        base = super().report()
        m = {}
        num_tok = self.metrics['num_tokens']
        if num_tok > 0:
            m['loss'] = self.metrics['loss']
            if self.metrics['correct_tokens'] > 0:
                m['token_acc'] = self.metrics['correct_tokens'] / num_tok
            m['nll_loss'] = self.metrics['nll_loss'] / num_tok
            try:
                m['ppl'] = math.exp(m['nll_loss'])
            except OverflowError:
                m['ppl'] = float('inf')
        if self.metrics['total_skipped_batches'] > 0:
            m['total_skipped_batches'] = self.metrics['total_skipped_batches']
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        if not self.skip_generation and self.compute_tokenized_bleu:
            base.update({'fairseq_bleu': 'N/A', 'nltk_bleu_unnormalized': 'N/A'})
            if fairseq_bleu is not None:
                try:
                    fairseq_bleu_scores = {
                        k: self.fairseq_bleu_scorer.result_string(order=k)
                        for k in range(1, 5)
                    }
                except ZeroDivisionError:
                    # some preds are REAL bad
                    fairseq_bleu_scores = {k: '= 0,' for k in range(1, 5)}

                base['fairseq_bleu'] = {
                    k: float(v[v.index('= ') + 2 : v.index(',')])
                    for k, v in fairseq_bleu_scores.items()
                }
            if nltkbleu is not None:
                base['nltk_bleu_unnormalized'] = {
                    k: round_sigfigs(v['score'] / v['cnt'], 4)
                    for k, v in self.nltk_bleu_scores.items()
                }
        return base

    def vectorize(self, *args, **kwargs):
        """
        Override vectorize for generative models.
        """
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        return super().vectorize(*args, **kwargs)

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
        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1)).sum()
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum().item()
        correct = ((batch.label_vec == preds) * notnull).sum().item()
        self.metrics['correct_tokens'] += correct
        self.metrics['nll_loss'] += loss.item()
        self.metrics['num_tokens'] += target_tokens
        loss /= target_tokens  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        if batch.text_vec is not None:
            batchsize = batch.text_vec.size(0)
        elif batch.image is not None:
            batchsize = len(batch.image)
        # helps with memory usage
        self._init_cuda_buffer(batchsize, self.truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            loss = self.compute_loss(batch)
            self.metrics['loss'] += loss.item()
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
                self.metrics['total_skipped_batches'] += 1
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e

    def _construct_token_losses(self, labels, model_output):
        # Get non-aggregated losses
        scores, _, _ = model_output
        score_view = scores.view(-1, scores.size(-1))
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

    def _compute_fairseq_bleu(self, batch: Batch, texts: List[str]):
        """
        Compute BLEU score between text and label, using the FAIRSeq BLEU Scorer.

        :param batch:
            Batch of observations
        :param texts:
            list of string predictions
        """
        if fairseq_bleu is None:
            return 0
        aa = torch.IntTensor(1)
        for i, t in enumerate(texts):
            self.fairseq_bleu_scorer.add(
                batch.label_vec[i].type_as(aa),
                self._vectorize_text(t, True, True, self.label_truncate, False).type_as(
                    aa
                ),
            )

    def _compute_nltk_bleu(self, batch: Batch, texts: List[str]):
        """
        Compute BLEU score between text and label(s), using the NLTK BLEU Scorer.

        Note this differs from BLEU in ParlAI metrics in that the answers
        are unnormalized (no removal of stop words, etc.)

        :param batch:
            Batch of observations
        :param texts:
            list of string predictions
        """

        def _bleu(guess: str, answers: List[str], weights: List[float]):
            """
            Compute approximate BLEU score between guess and a set of answers.

            This function does not process guess or answers
            """
            if nltkbleu is None:
                return 0
            return nltkbleu.sentence_bleu(
                [a.split(" ") for a in answers],
                guess.split(" "),
                smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
                weights=weights,
            )

        for i, p in enumerate(texts):
            obs = batch.observations[i]
            references = []
            for lbl in obs['eval_labels']:
                references.append(
                    self._v2t(
                        self._vectorize_text(
                            lbl, True, True, self.label_truncate, False
                        )
                    )
                )
            for i in range(4):
                weights = [1 / (i + 1) for _ in range(i + 1)]
                self.nltk_bleu_scores[f'bleu-{i + 1}']['score'] += _bleu(
                    p, references, weights
                )
                self.nltk_bleu_scores[f'bleu-{i + 1}']['cnt'] += 1

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

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            self.metrics['loss'] += loss.item()
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning,
            )
        else:
            maxlen = self.label_truncate or 256
            beam_preds_scores, _ = self._generate(batch, self.beam_size, maxlen)
            preds, scores = zip(*beam_preds_scores)

        cand_choices = None
        # TODO: abstract out the scoring here
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(*self._encoder_input(batch))
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = self._pad_tensor(batch.candidate_vecs[i])
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
        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, text)
            self._compute_nltk_bleu(batch, text)
        return Output(text, cand_choices, token_losses=token_losses)

    def _treesearch_factory(self, device):
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
            )
        else:
            raise ValueError(f"Can't use inference method {method}")

    def _generate(self, batch, beam_size, max_ts):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence

        :return:
            tuple (beam_pred_scores, n_best_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - n_best_preds_scores: list of n_best list of tuples (prediction, score)
              for each sample from Batch
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
            dev = batch.label_vec.device

        bsz = (
            len(batch.text_lengths)
            if batch.text_lengths is not None
            else len(batch.image)
        )
        if batch.text_vec is not None:
            beams = [
                self._treesearch_factory(dev).set_context(ctx) for ctx in batch.text_vec
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = (
            torch.LongTensor([self.START_IDX]).expand(bsz * beam_size, 1).to(dev)
        )

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
            score = F.log_softmax(score, dim=-1)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = torch.cat([decoder_input, selection], dim=-1)

        # get all finilized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams


class _HypothesisTail(object):
    """
    Hold some bookkeeping about a hypothesis.
    """

    # use slots because we don't want dynamic attributes here
    __slots__ = ['timestep', 'hypid', 'score', 'tokenid']

    def __init__(self, timestep, hypid, score, tokenid):
        self.timestep = timestep
        self.hypid = hypid
        self.score = score
        self.tokenid = tokenid


TSType = TypeVar('TSType', bound='TreeSearch')


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
        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.partial_hyps = [[self.bos] for i in range(beam_size)]

    def set_context(self: TSType, context: torch.LongTensor) -> TSType:
        """
        Set the internal context representation and return self.

        :param context:
            a LongTensor representing the input context; used for context
            ngram blocking, if supplied
        """
        self.context = context.tolist()
        return self

    def get_output_from_current_step(self):
        """
        Get the outputput at the current step.
        """
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """
        Get the backtrack at the current step.
        """
        return self.bookkeep[-1]

    @abstractmethod
    def select_paths(self, logprobs, prior_scores):
        """
        Select the next vocabulary item in these beams.

        :param logprobs:
            a (beamsize x vocab) tensor of log probabilities. If this is the first
            turn in the dialogue, it will be a (1 x vocab) tensor.
        :param prior_scores:
            a (beamsize) tensor of weights with the cumulative running
            log-probability of each beam. If the first turn, it will be a (1) tensor.

        :return:
            a (hypothesis_ids, token_id, scores) tuple, where:

            - hypothesis_ids is a LongTensor of hypotheses we're extending. May have
              repeats, but should always be (beamsize) long.
            - token_ids is a (beamsize) LongTensor of next-token choices for
              each of the hypotheses.
            - scores is a (beamsize) Tensor with the updated cumulative log-probs
              of each beam.
        """
        pass

    def _block_ngrams(
        self, ngram_size: int, logprobs: torch.Tensor, source: torch.LongTensor = None
    ):
        """
        Hard block ngrams from the logprobs, based on the source.

        :param ngram_size:
            The length of ngrams to block. Must be > 0.
        :param logprobs:
            Float or HalfTensor, representing the log-probabilities. This is
            modified in place.
        :param source:
            Source text to grab ngrams from. If None, it uses the current
            hypothesis (i.e. self-blocking).
        """
        for beam_id, hyp in enumerate(self.partial_hyps):
            if len(hyp) < ngram_size - 1:
                continue
            source_ = hyp if source is None else source
            ngrams = self._find_ngrams(source_, ngram_size)
            prefix = hyp[-(ngram_size - 1) :]
            for ngram in ngrams:
                if ngram_size == 1 or prefix == list(ngram[:-1]):
                    logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
        return logprobs

    def advance(self, logprobs):
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
            logprobs = self._block_ngrams(self.block_ngram, logprobs, None)

        if self.context_block_ngram > 0:
            if self.context is None:
                raise ValueError(
                    "Must use TreeSearch.set_context to use context blocking."
                )
            logprobs = self._block_ngrams(
                self.context_block_ngram, logprobs, self.context
            )

        hyp_ids, tok_ids, self.scores = self.select_paths(logprobs, self.scores)
        # use clone() here to ensure that self.all_scores will not be changed
        # later due to any penalties to self.scores
        self.all_scores.append(self.scores.clone())

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [
            self.partial_hyps[hyp_ids[i]] + [tok_ids[i].item()]
            for i in range(self.beam_size)
        ]

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                if self.scores[hypid] == neginf(self.scores.dtype):
                    continue
                #  this is finished hypo, adding to finished
                eostail = _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.all_scores[-1][hypid],
                    tokenid=self.eos,
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
            list of (tokens, score) pairs, in sorted order, where:
              - tokens is a tensor of token ids
              - score is the adjusted log probability of the entire utterance
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
                )
            )

        # Note: beam size is almost always pretty small, so sorting is cheap enough
        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        n_best_list = [
            (self._get_pretty_hypothesis(self._get_hyp_from_finished(hyp)), hyp.score)
            for hyp in srted
        ]

        # check that there is at least one finished candidate
        # and assert that each of them contains only one EOS
        assert (
            len(n_best_list) >= 1
        ), f'TreeSearch returned {len(n_best_list)} candidates, must be >= 1'
        for (pred, score) in n_best_list:
            assert (
                pred == self.eos
            ).sum() == 1, f'TreeSearch returned a finalized hypo with multiple end tokens \
            with score {score.item():.2f}'

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

    def select_paths(self, logprobs, prior_scores):
        tok_scores, tok_ids = logprobs.max(1)
        best_scores = tok_scores + prior_scores
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        return (hyp_ids, tok_ids, best_scores)


class BeamSearch(TreeSearch):
    """
    Beam search.
    """

    def select_paths(self, logprobs, prior_scores):
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
        hyp_ids = best_idxs / voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        return (hyp_ids, tok_ids, best_scores)


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

    def select_paths(self, logprobs, prior_scores):
        values, indices = logprobs.topk(self.k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        choices = torch.multinomial(probs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = indices[hyp_ids, choices]
        scores = values[hyp_ids, choices]
        best_scores = prior_scores.expand_as(scores) + scores
        return (hyp_ids, tok_ids, best_scores)


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

    def select_paths(self, logprobs, prior_scores):
        # Unlike the other treesearch methods, we have to switch to linspace
        # for the probabilities in order to compute the CDF.
        probs = torch.softmax(logprobs, dim=-1)
        sprobs, sinds = probs.sort(dim=-1, descending=True)
        # The subtraction here is so that we always include the first word to
        # go over p. For example, if the most probable token has a prob of 0.5, and
        # p = 0.3, then we need still need to include that first token.
        mask = (sprobs.cumsum(dim=-1) - sprobs[:, :1]) >= self.p
        sprobs[mask] = 0
        sprobs.div_(sprobs.sum(dim=-1).unsqueeze(1))
        choices = torch.multinomial(sprobs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = sinds[hyp_ids, choices]
        # Convert back to logspace.
        scores = sprobs[hyp_ids, choices].log()
        best_scores = prior_scores.expand_as(scores) + scores
        return (hyp_ids, tok_ids, best_scores)
