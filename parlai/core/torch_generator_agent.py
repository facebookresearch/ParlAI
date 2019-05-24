#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Generic PyTorch-based Generator agent. Implements quite a bit of boilerplate,
including forced-decoding loss and beam search.

Contains the following utilities:

* TorchGeneratorAgent class, which serves as a useful parent for generative torch
  agents.
* Beam class which provides some generic beam functionality for classes to use
"""

import os
import math
import tempfile
from collections import defaultdict, Counter, namedtuple
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.torch_agent import TorchAgent, Batch, Output
from parlai.core.utils import padded_tensor, round_sigfigs, warn_once, neginf
from parlai.core.thread_utils import SharedTable
from parlai.core.distributed_utils import is_distributed


class TorchGeneratorModel(nn.Module):
    """
    This Interface expects you to implement model with the following reqs:

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

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def decode_greedy(self, encoder_states, bsz, maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for _i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, incr_state)
            scores = scores[:, -1:, :]
            scores = self.output(scores)
            _, preds = scores.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

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
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

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
        raise NotImplementedError(
            "reorder_encoder_states must be implemented by the model"
        )

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
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
        raise NotImplementedError(
            "reorder_decoder_incremental_state must be implemented by model"
        )

    def forward(self, *xs, ys=None, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
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
        if ys is not None:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)

        if ys is not None:
            # use teacher forcing
            scores, preds = self.decode_forced(encoder_states, ys)
        else:
            scores, preds = self.decode_greedy(
                encoder_states,
                bsz,
                maxlen or self.longest_label
            )

        return scores, preds, encoder_states


class TorchGeneratorAgent(TorchAgent):
    """
    Abstract Generator agent. Only meant to be extended.

    TorchGeneratorAgent aims to handle much of the bookkeeping and
    infrastructure work for any generative models, like seq2seq or transformer.
    It implements the train_step and eval_step. The only requirement is that
    your model *must* implemented the interface TorchGeneratorModel interface.
    """
    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('Torch Generator Agent')
        agent.add_argument('--beam-size', type=int, default=1,
                           help='Beam size, if 1 then greedy search')
        agent.add_argument('--beam-dot-log', type='bool', default=False, hidden=True,
                           help='Dump beam trees as png dot images into /tmp folder')
        agent.add_argument('--beam-min-n-best', type=int, default=3,
                           help='Minimum number of nbest candidates to achieve '
                                'during the beam search')
        agent.add_argument('--beam-min-length', type=int, default=1,
                           help='Minimum length of prediction to be generated by '
                                'the beam search')
        agent.add_argument('--beam-block-ngram', type=int, default=0, hidden=True,
                           help='Block all repeating ngrams up to history length n-1')
        agent.add_argument('--skip-generation', type='bool', default=False, hidden=True,
                           help='Skip beam search. Useful for speeding up training, '
                                'if perplexity is the validation metric.')

        super(TorchGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        init_model, is_finetune = self._get_init_model(opt, shared)
        super().__init__(opt, shared)

        self.beam_dot_log = opt.get('beam_dot_log', False)
        self.beam_size = opt.get('beam_size', 1)
        self.beam_min_n_best = opt.get('beam_min_n_best', 3)
        self.beam_min_length = opt.get('beam_min_length', 3)
        self.beam_block_ngram = opt.get('beam_block_ngram', 0)
        self.skip_generation = opt.get('skip_generation', False)

        if shared:
            # set up shared properties
            self.model = shared['model']
            self.criterion = shared['criterion']
            self.metrics = shared['metrics']
            states = shared.get('states', {})
        else:
            # Note: we cannot change the type of metrics ahead of time, so you
            # should correctly initialize to floats or ints here
            self.metrics = {
                'nll_loss': 0.0,
                'loss': 0.0,
                'num_tokens': 0,
                'correct_tokens': 0,
                'total_skipped_batches': 0
            }
            # this is not a shared instance of this class, so do full init
            if self.beam_dot_log:
                self.beam_dot_dir = tempfile.mkdtemp(
                    prefix='{}-beamdot-beamsize-{}-'.format(
                        os.path.basename(
                            opt.get('model_file')),
                        self.beam_size))
                print(
                    '[ Saving dot beam logs in {} ]'.format(
                        self.beam_dot_dir))

            self.build_criterion()
            self.build_model()
            if self.fp16:
                self.model = self.model.half()

            if init_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]'
                      ''.format(init_model))
                states = self.load(init_model)
            else:
                states = {}

        if (
            # only build an optimizer if we're training
            'train' in opt.get('datatype', '') and
            # and this is the main model, or on every fork if doing hogwild
            (shared is None or self.opt.get('numthreads', 1) > 1)
        ):
            # do this regardless of share state, but don't
            self.init_optim(
                [p for p in self.model.parameters() if p.requires_grad],
                optim_states=states.get('optimizer'),
                saved_optim_type=states.get('optimizer_type')
            )
            self.build_lr_scheduler(states, hard_reset=is_finetune)

        if shared is None and is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.opt['gpu']],
                broadcast_buffers=False,
            )

        self.reset()

    def _v2t(self, vec):
        """Convert token indices to string of tokens."""
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def build_model(self):
        """
        Construct the model.

        The model should be set to self.model, and support
        the TorchGeneratorModel interface.
        """
        raise NotImplementedError(
            "AbstractClass: build_model must be implemented by the user."
        )

    def build_criterion(self):
        """
        Constructs the loss function. By default torch.nn.CrossEntropyLoss.
        The criterion function should be set to self.criterion.

        If overridden, this model should (1) handle calling cuda and (2)
        produce a sum that can be used for a per-token loss.
        """
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.NULL_IDX, reduction='sum'
        )
        if self.use_cuda:
            self.criterion.cuda()

    def _dummy_batch(self, batchsize, maxlen):
        """
        Creates a dummy batch. This is used to preinitialize the cuda buffer,
        or otherwise force a null backward pass after an OOM.
        """
        return Batch(
            text_vec=torch.ones(batchsize, maxlen).long().cuda(),
            label_vec=torch.ones(batchsize, 2).long().cuda(),
        )

    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        """Pre-initialize CUDA buffer by doing fake forward pass."""
        if self.use_cuda and (force or not hasattr(self, 'buffer_initialized')):
            try:
                loss = self.compute_loss(self._dummy_batch(batchsize, maxlen))
                self.backward(loss)
                self.buffer_initialized = True
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    m = ('CUDA OOM: Lower batch size (-bs) from {} or lower '
                         ' max sequence length (-tr) from {}'
                         ''.format(batchsize, maxlen))
                    raise RuntimeError(m)
                else:
                    raise e

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        super().reset_metrics()
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here
        self.metrics['loss'] = 0.0
        self.metrics['nll_loss'] = 0.0
        self.metrics['num_tokens'] = 0
        self.metrics['correct_tokens'] = 0

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['model'] = self.model
        shared['criterion'] = self.criterion
        if self.opt.get('numthreads', 1) > 1:
            # we're doing hogwild so share the model too
            if isinstance(self.metrics, dict):
                # move metrics and model to shared memory
                self.metrics = SharedTable(self.metrics)
                self.model.share_memory()
            shared['states'] = {  # don't share optimizer states
                'optimizer_type': self.opt['optimizer'],
            }
        shared['metrics'] = self.metrics  # do after numthreads check
        if self.beam_dot_log is True:
            shared['beam_dot_dir'] = self.beam_dot_dir
        return shared

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
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
        return base

    def vectorize(self, *args, **kwargs):
        """Override vectorize for generative models."""
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        return super().vectorize(*args, **kwargs)

    def _model_input(self, batch):
        """
        Creates the input (x) value for the model. Must return a tuple.
        This will be passed directly into the model via *args, i.e.,

        >>> model(*_model_input(batch))

        This is intentionally overridable so that richer models can pass the
        additional inputs.
        """
        return (batch.text_vec, )

    def compute_loss(self, batch, return_output=False):
        """
        Computes and returns the loss for the given batch. Easily overridable for
        customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
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
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
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
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e

    def _write_beam_dots(self, text_vecs, beams):
        """Write the beam dot files to disk."""
        for i, b in enumerate(beams):
            dot_graph = b.get_beam_dot(dictionary=self.dict, n_best=3)
            image_name = self._v2t(text_vecs[i, -20:])
            image_name = image_name.replace(' ', '-').replace('__null__', '')
            dot_graph.write_png(
                os.path.join(self.beam_dot_dir, "{}.png".format(image_name))
            )

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
            maxlen = self.label_truncate or 256
            _, preds, *_ = self.model(
                *self._model_input(batch), bsz=bsz, maxlen=maxlen
            )
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

        cand_choices = None
        # TODO: abstract out the scoring here
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
        return Output(text, cand_choices)

    def beam_search(self, model, batch, beam_size, start=1, end=2,
                    pad=0, min_length=3, min_n_best=5, max_ts=40, block_ngram=0):
        """
        Beam search given the model and Batch

        This function expects to be given a TorchGeneratorModel. Please refer to
        that interface for information.

        :param TorchGeneratorModel model:
            Implements the above interface
        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int start:
            start of sequence token
        :param int end:
            end of sequence token
        :param int pad:
            padding token
        :param int min_length:
            minimum length of the decoded sequence
        :param int min_n_best:
            minimum number of completed hypothesis generated from each beam
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
        encoder_states = model.encoder(*self._model_input(batch))
        dev = batch.text_vec.device

        bsz = len(batch.text_lengths)
        beams = [
            Beam(beam_size, min_length=min_length, padding_token=pad,
                 bos_token=start, eos_token=end, min_n_best=min_n_best,
                 cuda=dev, block_ngram=block_ngram)
            for i in range(bsz)
        ]

        # repeat encoder outputs and decoder inputs
        decoder_input = torch.LongTensor([start]).expand(bsz * beam_size, 1).to(dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            # exit early if needed
            if all((b.done() for b in beams)):
                break

            score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            for i, b in enumerate(beams):
                if not b.done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [beam_size * i +
                    b.get_backtrack_from_current_step() for i, b in enumerate(beams)])
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
            decoder_input = torch.cat([decoder_input, selection], dim=-1)

        for b in beams:
            b.check_finished()

        beam_preds_scores = [list(b.get_top_hyp()) for b in beams]
        for pair in beam_preds_scores:
            pair[0] = Beam.get_pretty_hypothesis(pair[0])

        n_best_beams = [b.get_rescored_finished(n_best=min_n_best) for b in beams]
        n_best_beam_preds_scores = []
        for i, beamhyp in enumerate(n_best_beams):
            this_beam = []
            for hyp in beamhyp:
                pred = beams[i].get_pretty_hypothesis(
                    beams[i].get_hyp_from_finished(hyp))
                score = hyp.score
                this_beam.append((pred, score))
            n_best_beam_preds_scores.append(this_beam)

        return beam_preds_scores, n_best_beam_preds_scores, beams


class _mydefaultdict(defaultdict):
    """
    Get function also uses default_factory for this defaultdict.

    This makes dict.get() behave like dict[] if a default is not provided.
    """

    def get(self, key, default=None):
        """
        Return value at key or default if key is not in dict.

        If a default is not provided, return the default factory value.
        """
        # override default from "get" (like "__getitem__" already is)
        return super().get(key, default or self.default_factory())


class PerplexityEvaluatorAgent(TorchGeneratorAgent):
    """
    Subclass for doing standardized perplexity evaluation.

    This is designed to be used in conjunction with the PerplexityWorld at
    parlai/scripts/eval_ppl.py. It uses the `next_word_probability` function
    to calculate the probability of tokens one token at a time.
    """

    def __init__(self, opt, shared=None):
        """Initialize evaluator."""
        if opt.get('multigpu'):
            print('| WARNING: Multi-GPU is not supported for the Perplexity ' +
                  'Evaluator Agent. Setting this option to False.')
            opt['multigpu'] = False
        super().__init__(opt, shared)
        self.prev_enc = None
        self.last_xs = None

    def next_word_probability(self, partial_out):
        """
        Return probability distribution over next words.

        This probability is based on both nn input and partial true output.
        This is used to calculate the per-word perplexity.

        :param observation:
            input observation dict

        :param partial_out:
            list of previous "true" words

        :return:
            a dict, where each key is a word and each value is a probability
            score for that word.  Unset keys will use a probability of 1e-7.

            e.g. {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
        """
        obs = self.observation
        xs = obs['text_vec'].unsqueeze(0)
        ys = self._vectorize_text(
            ' '.join(partial_out), False, True, self.truncate
        ).unsqueeze(0)
        if self.prev_enc is not None and self.last_xs is not None and (
                xs.shape[1] != self.last_xs.shape[1] or
                (xs == self.last_xs).sum().item() != xs.shape[1]):
            # reset prev_enc, this is a new input
            self.prev_enc = None
        self.last_xs = xs

        self.model.eval()
        out = self.model(
            xs,
            ys=(ys if len(partial_out) > 0 else None),
            prev_enc=self.prev_enc,
            maxlen=1)
        scores, self.prev_enc = out
        # scores is bsz x seqlen x num_words, so select probs of current index
        probs = F.softmax(scores.select(1, -1), dim=1).squeeze()
        dist = _mydefaultdict(lambda: 1e-7)  # default probability for any token
        for i in range(len(probs)):
            dist[self.dict[i]] = probs[i].item()
        return dist


class Beam(object):
    """Generic beam class. It keeps information about beam_size hypothesis."""

    def __init__(self, beam_size, min_length=3, padding_token=0, bos_token=1,
                 eos_token=2, min_n_best=3, cuda='cpu', block_ngram=0):
        """
        Instantiate Beam object.

        :param beam_size:
            number of hypothesis in the beam
        :param min_length:
            minimum length of the predicted sequence
        :param padding_token:
            Set to 0 as usual in ParlAI
        :param bos_token:
            Set to 1 as usual in ParlAI
        :param eos_token:
            Set to 2 as usual in ParlAI
        :param min_n_best:
            Beam will not be done unless this amount of finished hypothesis
            (with EOS) is done
        :param cuda:
            What device to use for computations
        """
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = cuda
        # recent score for each hypo in the beam
        self.scores = torch.Tensor(self.beam_size).float().zero_().to(
            self.device)
        # self.scores values per each time step
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [torch.Tensor(self.beam_size).long()
                        .fill_(self.bos).to(self.device)]
        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.HypothesisTail = namedtuple(
            'HypothesisTail', ['timestep', 'hypid', 'score', 'tokenid'])
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best
        self.block_ngram = block_ngram
        self.partial_hyps = [[self.bos] for i in range(beam_size)]

    @staticmethod
    def find_ngrams(input_list, n):
        """Get list of ngrams with context length n-1"""
        return list(zip(*[input_list[i:] for i in range(n)]))

    def get_output_from_current_step(self):
        """Get the outputput at the current step."""
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """Get the backtrack at the current step."""
        return self.bookkeep[-1]

    def advance(self, softmax_probs):
        """Advance the beam one step."""
        voc_size = softmax_probs.size(-1)
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(softmax_probs.size(0)):
                softmax_probs[hyp_id][self.eos] = neginf(softmax_probs.dtype)
        if len(self.bookkeep) == 0:
            # the first step we take only the first hypo into account since all
            # hypos are the same initially
            beam_scores = softmax_probs[0]
        else:
            # we need to sum up hypo scores and curr softmax scores before topk
            # [beam_size, voc_size]
            beam_scores = (softmax_probs +
                           self.scores.unsqueeze(1).expand_as(softmax_probs))
            for i in range(self.outputs[-1].size(0)):
                if self.block_ngram > 0:
                    current_hypo = self.partial_hyps[i][1:]
                    current_ngrams = []
                    for ng in range(self.block_ngram):
                        ngrams = Beam.find_ngrams(current_hypo, ng)
                        if len(ngrams) > 0:
                            current_ngrams.extend(ngrams)
                    counted_ngrams = Counter(current_ngrams)
                    if any(v > 1 for k, v in counted_ngrams.items()):
                        # block this hypothesis hard
                        beam_scores[i] = neginf(softmax_probs.dtype)

                #  if previous output hypo token had eos
                # we penalize those word probs to never be chosen
                if self.outputs[-1][i] == self.eos:
                    # beam_scores[i] is voc_size array for i-th hypo
                    beam_scores[i] = neginf(softmax_probs.dtype)

        flatten_beam_scores = beam_scores.view(-1)  # [beam_size * voc_size]
        with torch.no_grad():
            best_scores, best_idxs = torch.topk(
                flatten_beam_scores, self.beam_size, dim=-1)

        self.scores = best_scores
        self.all_scores.append(self.scores)
        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs / voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [self.partial_hyps[hyp_ids[i]] +
                             [tok_ids[i].item()] for i in range(self.beam_size)]

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                #  this is finished hypo, adding to finished
                eostail = self.HypothesisTail(timestep=len(self.outputs) - 1,
                                              hypid=hypid,
                                              score=self.scores[hypid],
                                              tokenid=self.eos)
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        """Return whether beam search is complete."""
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """
        Get single best hypothesis.

        :return: hypothesis sequence and the final score
        """
        top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
        return (self.get_hyp_from_finished(top_hypothesis_tail),
                top_hypothesis_tail.score)

    def get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep:
            timestep with range up to len(self.outputs)-1

        :param hyp_id:
            id with range up to beam_size-1

        :return:
            hypothesis sequence
        """
        assert (self.outputs[hypothesis_tail.timestep]
                [hypothesis_tail.hypid] == self.eos)
        assert hypothesis_tail.tokenid == self.eos
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(self.HypothesisTail(
                timestep=i, hypid=endback, score=self.all_scores[i][endback],
                tokenid=self.outputs[i][endback]))
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    @staticmethod
    def get_pretty_hypothesis(list_of_hypotails):
        """Return prettier version of the hypotheses."""
        hypothesis = []
        for i in list_of_hypotails:
            hypothesis.append(i.tokenid)

        hypothesis = torch.stack(list(reversed(hypothesis)))

        return hypothesis

    def get_rescored_finished(self, n_best=None):
        """
        Return finished hypotheses in rescored order.

        :param n_best:
            how many n best hypothesis to return

        :return:
            list with hypothesis
        """
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, 0.65)
            rescored_finished.append(self.HypothesisTail(
                timestep=finished_item.timestep, hypid=finished_item.hypid,
                score=finished_item.score / length_penalty,
                tokenid=finished_item.tokenid))

        srted = sorted(rescored_finished, key=attrgetter('score'),
                       reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return srted

    def check_finished(self):
        """
        Check if self.finished is empty and add hyptail in that case.

        This will be suboptimal hypothesis since the model did not get any EOS
        """
        if len(self.finished) == 0:
            # we change output because we want outputs to have eos
            # to pass assert in L102, it is ok since empty self.finished
            # means junk prediction anyway
            self.outputs[-1][0] = self.eos
            hyptail = self.HypothesisTail(timestep=len(self.outputs) - 1,
                                          hypid=0,
                                          score=self.all_scores[-1][0],
                                          tokenid=self.outputs[-1][0])

            self.finished.append(hyptail)

    def get_beam_dot(self, dictionary=None, n_best=None):
        """
        Create pydot graph representation of the beam.

        :param outputs:
            self.outputs from the beam

        :param dictionary:
            tok 2 word dict to save words in the tree nodes

        :returns:
            pydot graph
        """
        try:
            import pydot
        except ImportError:
            print("Please install pydot package to dump beam visualization")

        graph = pydot.Dot(graph_type='digraph')
        outputs = [i.tolist() for i in self.outputs]
        bookkeep = [i.tolist() for i in self.bookkeep]
        all_scores = [i.tolist() for i in self.all_scores]
        if n_best is None:
            n_best = int(self.beam_size / 2)

        # get top nbest hyp
        top_hyp_idx_n_best = []
        n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue',
                         'green2', 'tan']
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for hyptail in sorted_finished:
            # do not include EOS since it has rescored score not from original
            # self.all_scores, we color EOS with black
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(
                hyptail))

        # create nodes
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                if tstep == 0:
                    hypid = 0  # collapse all __NULL__ nodes
                node_tail = self.HypothesisTail(timestep=tstep, hypid=hypid,
                                                score=all_scores[tstep][hypid],
                                                tokenid=token)
                color = 'white'
                rank = None
                for i, hypseq in enumerate(top_hyp_idx_n_best):
                    if node_tail in hypseq:
                        if n_best <= 5:  # color nodes only if <=5
                            color = n_best_colors[i]
                        rank = i
                        break
                label = (
                    "<{}".format(dictionary.vec2txt([token])
                                 if dictionary is not None else token) +
                    " : " +
                    "{:.{prec}f}>".format(all_scores[tstep][hypid], prec=3))

                graph.add_node(pydot.Node(
                    node_tail.__repr__(), label=label, fillcolor=color,
                    style='filled',
                    xlabel='{}'.format(rank) if rank is not None else ''))

        # create edges
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i, prev_id in enumerate(lis):
                from_node = graph.get_node(
                    '"{}"'.format(self.HypothesisTail(
                        timestep=revtstep, hypid=prev_id,
                        score=all_scores[revtstep][prev_id],
                        tokenid=outputs[revtstep][prev_id]).__repr__()))[0]
                to_node = graph.get_node(
                    '"{}"'.format(self.HypothesisTail(
                        timestep=revtstep + 1, hypid=i,
                        score=all_scores[revtstep + 1][i],
                        tokenid=outputs[revtstep + 1][i]).__repr__()))[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)

        return graph
