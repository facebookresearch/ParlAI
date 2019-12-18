#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .torch_agent_v1 import TorchAgent, Output, Beam
from .utils_v1 import padded_tensor
from .utils_v0 import SharedTable, round_sigfigs
from .modules_v1 import Seq2seq, opt_to_kwargs

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

import os
import math
import json
import tempfile


class Seq2seqAgent(TorchAgent):
    """
    Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    several flavors of RNN. It then uses a linear layer (whose weights can
    be shared with the embedding layer) to convert RNN output states into
    output tokens. This model supports greedy decoding, selecting the
    highest probability token at each time step, as well as beam
    search.

    For more information, see the following papers:
    - Neural Machine Translation by Jointly Learning to Align and Translate
      `(Bahdanau et al. 2014) <arxiv.org/abs/1409.0473>`_
    - Sequence to Sequence Learning with Neural Networks
      `(Sutskever et al. 2014) <arxiv.org/abs/1409.3215>`_
    - Effective Approaches to Attention-based Neural Machine Translation
      `(Luong et al. 2015) <arxiv.org/abs/1508.04025>`_
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument(
            '--init-model',
            type=str,
            default=None,
            help='load dict/model/opts from this path',
        )
        agent.add_argument(
            '-hs',
            '--hiddensize',
            type=int,
            default=128,
            help='size of the hidden layers',
        )
        agent.add_argument(
            '-esz',
            '--embeddingsize',
            type=int,
            default=128,
            help='size of the token embeddings',
        )
        agent.add_argument(
            '-nl', '--numlayers', type=int, default=2, help='number of hidden layers'
        )
        agent.add_argument(
            '-dr', '--dropout', type=float, default=0.1, help='dropout rate'
        )
        agent.add_argument(
            '-bi',
            '--bidirectional',
            type='bool',
            default=False,
            help='whether to encode the context with a ' 'bidirectional rnn',
        )
        agent.add_argument(
            '-att',
            '--attention',
            default='none',
            choices=['none', 'concat', 'general', 'dot', 'local'],
            help='Choices: none, concat, general, local. '
            'If set local, also set attention-length. '
            '(see arxiv.org/abs/1508.04025)',
        )
        agent.add_argument(
            '-attl',
            '--attention-length',
            default=48,
            type=int,
            help='Length of local attention.',
        )
        agent.add_argument(
            '--attention-time',
            default='post',
            choices=['pre', 'post'],
            help='Whether to apply attention before or after ' 'decoding.',
        )
        agent.add_argument(
            '-rnn',
            '--rnn-class',
            default='lstm',
            choices=Seq2seq.RNN_OPTS.keys(),
            help='Choose between different types of RNNs.',
        )
        agent.add_argument(
            '-dec',
            '--decoder',
            default='same',
            choices=['same', 'shared'],
            help='Choose between different decoder modules. '
            'Default "same" uses same class as encoder, '
            'while "shared" also uses the same weights. '
            'Note that shared disabled some encoder '
            'options--in particular, bidirectionality.',
        )
        agent.add_argument(
            '-lt',
            '--lookuptable',
            default='unique',
            choices=['unique', 'enc_dec', 'dec_out', 'all'],
            help='The encoder, decoder, and output modules can '
            'share weights, or not. '
            'Unique has independent embeddings for each. '
            'Enc_dec shares the embedding for the encoder '
            'and decoder. '
            'Dec_out shares decoder embedding and output '
            'weights. '
            'All shares all three weights.',
        )
        agent.add_argument(
            '-soft',
            '--numsoftmax',
            default=1,
            type=int,
            help='default 1, if greater then uses mixture of '
            'softmax (see arxiv.org/abs/1711.03953).',
        )
        agent.add_argument(
            '--beam-size',
            type=int,
            default=1,
            help='Beam size, if 1 then greedy search',
        )
        agent.add_argument(
            '--beam-dot-log',
            type='bool',
            default=False,
            help='Dump beam trees as png dot images into /tmp folder',
        )
        agent.add_argument(
            '--beam-min-n-best',
            type=int,
            default=3,
            help='Minimum number of nbest candidates to achieve '
            'during the beam search',
        )
        agent.add_argument(
            '--beam-min-length',
            type=int,
            default=3,
            help='Minimum length of prediction to be generated by ' 'the beam search',
        )
        agent.add_argument(
            '-idr',
            '--input-dropout',
            type=float,
            default=0.0,
            help='Each token from the input will be masked with'
            ' __unk__ token with this probability.',
        )
        agent.add_argument(
            '--beam-block-ngram',
            type=int,
            default=0,
            help='Block all repeating ngrams up to history length n-1',
        )
        TorchAgent.add_cmdline_args(argparser)
        Seq2seqAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        """
        Return current version of this model, counting up from 0.

        Models may not be backwards-compatible with older versions. Version 1 split from
        version 0 on Aug 29, 2018. To use version 0, use --model legacy:seq2seq:0
        (legacy agent code is located in parlai/agents/legacy_agents).
        """
        return 1

    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        init_model = None
        if not shared:  # only do this on first setup
            # first check load path in case we need to override paths
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                # check first for 'init_model' for loading model from file
                init_model = opt['init_model']
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # next check for 'model_file', this would override init_model
                init_model = opt['model_file']

            if init_model is not None:
                # if we are loading a model, should load its dict too
                if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                    opt['dict_file'] = init_model + '.dict'
        super().__init__(opt, shared)
        opt = self.opt

        # all instances may need some params
        self.id = 'Seq2Seq'
        self.multigpu = (
            opt.get('multigpu') and self.use_cuda and (opt.get('batchsize') > 1)
        )
        states = {}

        self.beam_dot_log = opt.get('beam_dot_log', False)
        self.beam_size = opt.get('beam_size', 1)
        self.beam_min_n_best = opt.get('beam_min_n_best', 3)
        self.beam_min_length = opt.get('beam_min_length', 3)
        self.beam_block_ngram = opt.get('beam_block_ngram', 0)

        if shared:
            # set up shared properties
            self.model = shared['model']
            self.metrics = shared['metrics']
            states = shared.get('states', {})
        else:
            self.metrics = {
                'loss': 0.0,
                'num_tokens': 0,
                'correct_tokens': 0,
                'total_skipped_batches': 0,
            }
            # this is not a shared instance of this class, so do full init
            if self.beam_dot_log:
                self.beam_dot_dir = tempfile.mkdtemp(
                    prefix='{}-beamdot-beamsize-{}-'.format(
                        os.path.basename(opt.get('model_file')), self.beam_size
                    )
                )
                print('[ Saving dot beam logs in {} ]'.format(self.beam_dot_dir))
            if init_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]' ''.format(init_model))
                states = self.load(init_model)

            self._init_model(states=states)

        # set up criteria
        if opt.get('numsoftmax', 1) > 1:
            self.criterion = nn.NLLLoss(ignore_index=self.NULL_IDX, size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, size_average=False
            )

        if self.use_cuda:
            self.criterion.cuda()

        if 'train' in opt.get('datatype', ''):
            self.init_optim(
                [p for p in self.model.parameters() if p.requires_grad],
                optim_states=states.get('optimizer'),
                saved_optim_type=states.get('optimizer_type'),
            )

        self.reset()

    def _init_model(self, states=None):
        """
        Initialize model, override to change model setup.
        """
        opt = self.opt

        kwargs = opt_to_kwargs(opt)
        self.model = Seq2seq(
            len(self.dict),
            opt['embeddingsize'],
            opt['hiddensize'],
            padding_idx=self.NULL_IDX,
            start_idx=self.START_IDX,
            unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            **kwargs,
        )

        if opt.get('dict_tokenizer') == 'bpe' and opt['embedding_type'] != 'random':
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(self.model.decoder.lt.weight, opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(
                    self.model.encoder.lt.weight, opt['embedding_type'], log=False
                )

        if states:
            # set loaded states if applicable
            self.model.load_state_dict(states['model'])

        if opt['embedding_type'].endswith('fixed'):
            print('Seq2seq: fixing embedding weights.')
            self.model.decoder.lt.weight.requires_grad = False
            self.model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                self.model.decoder.e2s.weight.requires_grad = False

        if self.use_cuda:
            self.model.cuda()
            if self.multigpu:
                self.model = torch.nn.DataParallel(self.model)
                self.model.encoder = self.model.module.encoder
                self.model.decoder = self.model.module.decoder
                self.model.longest_label = self.model.module.longest_label
                self.model.output = self.model.module.output

        return self.model

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

    def zero_grad(self):
        """
        Zero out optimizer.
        """
        self.optimizer.zero_grad()

    def update_params(self):
        """
        Do one optimization step.
        """
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

    def reset_metrics(self):
        """
        Reset metrics for reporting loss and perplexity.
        """
        super().reset_metrics()
        self.metrics['loss'] = 0.0
        self.metrics['num_tokens'] = 0
        self.metrics['correct_tokens'] = 0

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may differ
        from a truly independent measurement.
        """
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
        return m

    def share(self):
        """
        Share internal states between parent and child instances.
        """
        shared = super().share()
        shared['model'] = self.model
        if self.opt.get('numthreads', 1) > 1:
            # we're doing hogwild so share the model too
            if isinstance(self.metrics, dict):
                # move metrics and model to shared memory
                self.metrics = SharedTable(self.metrics)
                self.model.share_memory()
            shared['states'] = {  # don't share optimizer states
                'optimizer_type': self.opt['optimizer']
            }
        shared['metrics'] = self.metrics  # do after numthreads check
        if self.beam_dot_log is True:
            shared['beam_dot_dir'] = self.beam_dot_dir
        return shared

    def vectorize(self, *args, **kwargs):
        """
        Override vectorize for seq2seq.
        """
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        return super().vectorize(*args, **kwargs)

    def batchify(self, *args, **kwargs):
        """
        Override batchify options for seq2seq.
        """
        kwargs['sort'] = True  # need sorted for pack_padded
        return super().batchify(*args, **kwargs)

    def _init_cuda_buffer(self, model, criterion, batchsize, maxlen):
        """
        Pre-initialize CUDA buffer by doing fake forward pass.
        """
        if self.use_cuda and not hasattr(self, 'buffer_initialized'):
            try:
                print('preinitializing pytorch cuda buffer')
                dummy = torch.ones(batchsize, maxlen).long().cuda()
                out = model(dummy, dummy)
                sc = out[0]  # scores
                loss = criterion(sc.view(-1, sc.size(-1)), dummy.view(-1))
                loss.backward()
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

    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        batchsize = batch.text_vec.size(0)
        if self.multigpu and batchsize % 2 != 0:
            # throw out one training example
            batch = self.truncate_input(batch)
        # helps with memory usage
        self._init_cuda_buffer(
            self.model, self.criterion, batchsize, self.truncate or 180
        )
        self.model.train()
        self.zero_grad()

        try:
            seq_len = None if not self.multigpu else batch.text_vec.size(1)
            out = self.model(batch.text_vec, batch.label_vec, seq_len=seq_len)

            # generated response
            scores = out[0]
            _, preds = scores.max(2)

            score_view = scores.view(-1, scores.size(-1))
            loss = self.criterion(score_view, batch.label_vec.view(-1))
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens
            loss /= target_tokens  # average loss per token
            loss.backward()
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
            else:
                raise e

    def _build_cands(self, batch):
        if not batch.candidates:
            return None, None
        cand_inds = [i for i in range(len(batch.candidates)) if batch.candidates[i]]
        cands = [batch.candidate_vecs[i] for i in cand_inds]
        max_cands_len = max(
            [max([cand.size(0) for cand in cands_i]) for cands_i in cands]
        )
        for i, c in enumerate(cands):
            cands[i] = padded_tensor(c, use_cuda=self.use_cuda, max_len=max_cands_len)[
                0
            ].unsqueeze(0)
        cands = torch.cat(cands, 0)
        return cands, cand_inds

    def _pick_cands(self, cand_preds, cand_inds, cands):
        cand_replies = [None] * len(cands)
        for idx, order in enumerate(cand_preds):
            batch_idx = cand_inds[idx]
            cand_replies[batch_idx] = [cands[batch_idx][i] for i in order]
        return cand_replies

    def greedy_search(self, batch):
        cand_params = self._build_cands(batch)
        seq_len = None if not self.multigpu else batch.text_vec.size(1)
        out = self.model(batch.text_vec, ys=None, cands=cand_params[0], seq_len=seq_len)
        return out, cand_params

    @staticmethod
    def beam_search(
        model,
        batch,
        beam_size,
        start=1,
        end=2,
        pad=0,
        min_length=3,
        min_n_best=5,
        max_ts=40,
        block_ngram=0,
    ):
        """
        Beam search given the model and Batch This function uses model with the
        following reqs:

        - model.encoder takes input returns tuple (enc_out, enc_hidden, attn_mask)
        - model.decoder takes decoder params and returns decoder outputs after attn
        - model.output takes decoder outputs and returns distr over dictionary

        Function arguments:
        model : nn.Module, here defined in modules.py
        batch : Batch structure with input and labels
        beam_size : Size of each beam during the search
        start : start of sequence token
        end : end of sequence token
        pad : padding token
        min_length : minimum length of the decoded sequence
        min_n_best : minimum number of completed hypothesis generated from each beam
        max_ts: the maximum length of the decoded sequence

        Return:
        beam_preds_scores : list of tuples (prediction, score) for each sample in Batch
        n_best_preds_scores : list of n_best list of tuples (prediction, score) for
                              each sample from Batch
        beams : list of Beam instances defined in Beam class, can be used for any
                following postprocessing, e.g. dot logging.
        """
        encoder_states = model.encoder(batch.text_vec)
        enc_out = encoder_states[0]
        enc_hidden = encoder_states[1]
        attn_mask = encoder_states[2]
        current_device = encoder_states[0][0].device

        batch_size = len(batch.text_lengths)
        beams = [
            Beam(
                beam_size,
                min_length=min_length,
                padding_token=pad,
                bos_token=start,
                eos_token=end,
                min_n_best=min_n_best,
                cuda=current_device,
                block_ngram=block_ngram,
            )
            for i in range(batch_size)
        ]
        decoder_input = (
            torch.Tensor([start])
            .detach()
            .expand(batch_size, 1)
            .long()
            .to(current_device)
        )
        # repeat encoder_outputs, hiddens, attn_mask
        decoder_input = decoder_input.repeat(1, beam_size).view(
            beam_size * batch_size, -1
        )
        enc_out = (
            enc_out.unsqueeze(1)
            .repeat(1, beam_size, 1, 1)
            .view(batch_size * beam_size, -1, enc_out.size(-1))
        )
        attn_mask = (
            encoder_states[2]
            .repeat(1, beam_size)
            .view(attn_mask.size(0) * beam_size, -1)
        )
        repeated_hiddens = []
        if isinstance(enc_hidden, tuple):  # LSTM
            for i in range(len(enc_hidden)):
                repeated_hiddens.append(
                    enc_hidden[i].unsqueeze(2).repeat(1, 1, beam_size, 1)
                )
            num_layers = enc_hidden[0].size(0)
            hidden_size = enc_hidden[0].size(-1)
            enc_hidden = tuple(
                [
                    repeated_hiddens[i].view(
                        num_layers, batch_size * beam_size, hidden_size
                    )
                    for i in range(len(repeated_hiddens))
                ]
            )
        else:  # GRU
            num_layers = enc_hidden.size(0)
            hidden_size = enc_hidden.size(-1)
            enc_hidden = (
                enc_hidden.unsqueeze(2)
                .repeat(1, 1, beam_size, 1)
                .view(num_layers, batch_size * beam_size, hidden_size)
            )

        hidden = enc_hidden
        for ts in range(max_ts):
            if all((b.done() for b in beams)):
                break
            output, hidden = model.decoder(decoder_input, hidden, (enc_out, attn_mask))
            score = model.output(output)
            # score contains softmax scores for batch_size * beam_size samples
            score = score.view(batch_size, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            for i, b in enumerate(beams):
                if not b.done():
                    b.advance(score[i])
            decoder_input = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            permute_hidden_idx = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            # permute decoder hiddens with respect to chosen hypothesis now
            if isinstance(hidden, tuple):  # LSTM
                for i in range(len(hidden)):
                    hidden[i].data.copy_(
                        hidden[i].data.index_select(dim=1, index=permute_hidden_idx)
                    )
            else:  # GRU
                hidden.data.copy_(
                    hidden.data.index_select(dim=1, index=permute_hidden_idx)
                )
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
                    beams[i].get_hyp_from_finished(hyp)
                )
                score = hyp.score
                this_beam.append((pred, score))
            n_best_beam_preds_scores.append(this_beam)

        return beam_preds_scores, n_best_beam_preds_scores, beams

    def extend_input(self, batch):
        # add pad tensor to text vec
        pad_tensor = torch.zeros(1, batch.text_vec.size(1)).long().cuda()
        text_vec = torch.cat([batch.text_vec, pad_tensor], 0)
        batch = batch._replace(text_vec=text_vec)
        if batch.label_vec is not None:
            # add pad tensor to label vec
            pad_tensor = torch.zeros(1, batch.label_vec.size(1)).long().cuda()
            label_vec = torch.cat([batch.label_vec, pad_tensor], 0)
            batch = batch._replace(label_vec=label_vec)
        if batch.candidates is not None:
            # add dummy candidates list
            dummy_list = [['None'] for _ in range(len(batch.candidates[0]))]
            batch = batch._replace(candidates=batch.candidates + [dummy_list])
            # add pad tensor to candidate_vecs
            new_vecs = batch.candidate_vecs + [
                [torch.zeros(1).long() for _ in range(len(batch.candidate_vecs[0]))]
            ]
            batch = batch._replace(candidate_vecs=new_vecs)
        return batch

    def truncate_input(self, batch):
        # truncate batch for multigpu
        text_vec = batch.text_vec[:-1]
        batch = batch._replace(text_vec=text_vec)
        if batch.label_vec is not None:
            label_vec = batch.label_vec[:-1]
            batch = batch._replace(label_vec=label_vec)
        return batch

    def truncate_output(self, out):
        new_out_0 = out[0][:-1]
        new_out_1 = None if out[1] is None else out[1][:-1]
        new_out_2 = [vec[:-1] for vec in out[2]]
        return tuple([new_out_0, new_out_1, new_out_2])

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None:
            return
        orig_batch = batch  # save for evaluation
        needs_truncation = self.multigpu and batch.text_vec.size(0) % 2 != 0
        if needs_truncation:
            # for multigpu, we need to split evenly across gpus
            batch = self.extend_input(batch)
        self.model.eval()
        cand_scores = None
        if self.beam_size == 1:
            out, cand_params = self.greedy_search(batch)
            if needs_truncation:
                out = self.truncate_output(out)
                if cand_params[0] is not None:
                    cand_params = (cand_params[0][:-1], cand_params[1][:-1])
            scores, cand_scores = out[0], out[1]
            _, preds = scores.max(2)
        elif self.beam_size > 1:
            out = Seq2seqAgent.beam_search(
                self.model,
                batch,
                self.beam_size,
                start=self.START_IDX,
                end=self.END_IDX,
                pad=self.NULL_IDX,
                min_length=self.beam_min_length,
                min_n_best=self.beam_min_n_best,
                block_ngram=self.beam_block_ngram,
            )
            if needs_truncation:
                out = self.truncate_output(out)
            beam_preds_scores, _, beams = out
            preds, scores = (
                [p[0] for p in beam_preds_scores],
                [p[1] for p in beam_preds_scores],
            )
            if self.beam_dot_log is True:
                for i, b in enumerate(beams):
                    dot_graph = b.get_beam_dot(dictionary=self.dict, n_best=3)
                    image_name = (
                        self._v2t(batch.text_vec[i, -20:])
                        .replace(' ', '-')
                        .replace('__null__', '')
                    )
                    dot_graph.write_png(
                        os.path.join(self.beam_dot_dir, "{}.png".format(image_name))
                    )

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            seq_len = None if not self.multigpu else batch.text_vec.size(1)
            out = self.model(batch.text_vec, batch.label_vec, seq_len=seq_len)
            if needs_truncation:
                out = self.truncate_output(out)
            f_scores = out[0]  # forced scores
            _, f_preds = f_scores.max(2)  # forced preds
            score_view = f_scores.view(-1, f_scores.size(-1))
            loss = self.criterion(score_view, orig_batch.label_vec.view(-1))
            # save loss to metrics
            notnull = orig_batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((orig_batch.label_vec == f_preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens

        cand_choices = None
        if cand_scores is not None:
            cand_preds = cand_scores.sort(1, descending=True)[1]
            # now select the text of the cands based on their scores
            cand_choices = self._pick_cands(
                cand_preds, cand_params[1], orig_batch.candidates
            )

        text = [self._v2t(p) for p in preds]
        return Output(text, cand_choices)

    def save(self, path=None):
        """
        Save model parameters if model_file is set.
        """
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            model['model'] = self.model.state_dict()
            model['longest_label'] = self.model.longest_label
            model['optimizer'] = self.optimizer.state_dict()
            model['optimizer_type'] = self.opt['optimizer']

            with open(path, 'wb') as write:
                torch.save(model, write)

            # save opt file
            with open(path + '.opt', 'w') as handle:
                # save version string
                self.opt['model_version'] = self.model_version()
                json.dump(self.opt, handle)

    def load(self, path):
        """
        Return opt and model states.
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)

        # check opt file for multigpu
        with open(path + ".opt", 'r') as handle:
            saved_opt = json.load(handle)
        if saved_opt.get('multigpu'):
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in states['model'].items():
                if k.startswith('module'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
            states['model'] = new_state_dict

        return states


class mydefaultdict(defaultdict):
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


class PerplexityEvaluatorAgent(Seq2seqAgent):
    """
    Subclass for doing standardized perplexity evaluation.

    This is designed to be used in conjunction with the PerplexityWorld at
    parlai/scripts/eval_ppl.py. It uses the `next_word_probability` function to
    calculate the probability of tokens one token at a time.
    """

    def __init__(self, opt, shared=None):
        """
        Initialize evaluator.
        """
        if opt.get('multigpu'):
            print(
                '| WARNING: Multi-GPU is not supported for the Perplexity '
                + 'Evaluator Agent. Setting this option to False.'
            )
            opt['multigpu'] = False
        super().__init__(opt, shared)
        self.prev_enc = None
        self.last_xs = None

    def next_word_probability(self, partial_out):
        """
        Return probability distribution over next words.

        This probability is based on both nn input and partial true output.
        This is used to calculate the per-word perplexity.

        Arguments:
        observation -- input observation dict
        partial_out -- list of previous "true" words

        Returns a dict, where each key is a word and each value is a
        probability score for that word.
        Unset keys will use a probability of 1e-7.

        e.g.
        {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
        """
        obs = self.observation
        xs = obs['text_vec'].unsqueeze(0)
        ys = self._vectorize_text(
            ' '.join(partial_out), False, True, self.truncate
        ).unsqueeze(0)
        if (
            self.prev_enc is not None
            and self.last_xs is not None
            and (
                xs.shape[1] != self.last_xs.shape[1]
                or (xs == self.last_xs).sum().item() != xs.shape[1]
            )
        ):
            # reset prev_enc, this is a new input
            self.prev_enc = None
        self.last_xs = xs

        self.model.eval()
        out = self.model(
            xs,
            ys=(ys if len(partial_out) > 0 else None),
            prev_enc=self.prev_enc,
            maxlen=1,
        )
        scores, self.prev_enc = out[0], out[2]
        # scores is bsz x seqlen x num_words, so select probs of current index
        probs = F.softmax(scores.select(1, -1), dim=1).squeeze()
        dist = mydefaultdict(lambda: 1e-7)  # default probability for any token
        for i in range(len(probs)):
            dist[self.dict[i]] = probs[i].item()
        return dist
