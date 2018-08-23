# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.build_data import modelzoo_path
from parlai.core.utils import argsort, padded_tensor, round_sigfigs
from parlai.core.thread_utils import SharedTable
from .modules import Seq2seq

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, defaultdict

import os
import math
import pickle


class Seq2seqAgent(TorchAgent):
    """Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    several flavors of RNN. It then uses a linear layer (whose weights can
    be shared with the embedding layer) to convert RNN output states into
    output tokens. This model currently uses greedy decoding, selecting the
    highest probability token at each time step.

    For more information, see the following papers:
    - Neural Machine Translation by Jointly Learning to Align and Translate
      `(Bahdanau et al. 2014) <arxiv.org/abs/1409.0473>`_
    - Sequence to Sequence Learning with Neural Networks
      `(Sutskever et al. 2014) <arxiv.org/abs/1409.3215>`_
    - Effective Approaches to Attention-based Neural Machine Translation
      `(Luong et al. 2015) <arxiv.org/abs/1508.04025>`_
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('--init-model', type=str, default=None,
                           help='load dict/features/weights/opts from this file')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=1,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-clip', '--gradient-clip', type=float, default=0.1,
                           help='gradient clipping using l2 norm')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'dot', 'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                '(see arxiv.org/abs/1508.04025)')
        agent.add_argument('-attl', '--attention-length', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('--attention-time', default='post',
                           choices=['pre', 'post'],
                           help='Whether to apply attention before or after '
                                'decoding.')
        agent.add_argument('-rnn', '--rnn-class', default='lstm',
                           choices=Seq2seq.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'],
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights. '
                                'Note that shared disabled some encoder '
                                'options--in particular, bidirectionality.')
        agent.add_argument('-lt', '--lookuptable', default='unique',
                           choices=['unique', 'enc_dec', 'dec_out', 'all'],
                           help='The encoder, decoder, and output modules can '
                                'share weights, or not. '
                                'Unique has independent embeddings for each. '
                                'Enc_dec shares the embedding for the encoder '
                                'and decoder. '
                                'Dec_out shares decoder embedding and output '
                                'weights. '
                                'All shares all three weights.')
        agent.add_argument('-mom', '--momentum', default=-1, type=float,
                           help='if applicable, momentum value for optimizer. '
                                'if > 0, sgd uses nesterov momentum.')
        agent.add_argument('-emb', '--embedding-type', default='random',
                           choices=['random', 'glove', 'glove-fixed',
                                    'fasttext', 'fasttext-fixed',
                                    'glove-twitter'],
                           help='Choose between different strategies '
                                'for word embeddings. Default is random, '
                                'but can also preinitialize from Glove or '
                                'Fasttext.'
                                'Preinitialized embeddings can also be fixed '
                                'so they are not updated during training.')
        agent.add_argument('-soft', '--numsoftmax', default=1, type=int,
                           help='default 1, if greater then uses mixture of '
                                'softmax (see arxiv.org/abs/1711.03953).')
        agent.add_argument('-rf', '--report-freq', type=float, default=0.001,
                           help='Report frequency of prediction during eval.')
        agent.add_argument('--beam-size', type=int, default=1,
                           help='Beam size, if 1 then greedy search')
        agent.add_argument('--beam-log-freq', type=float, default=0.0,
                           help='The portion of beams to dump from minibatch '
                                'into model_name.beam_dump folder')
        agent.add_argument('--topk', type=int, default=1,
                           help='Top k sampling from renormalized softmax in '
                                'test/valid time, default 1 means simple '
                                'greedy max output')
        agent.add_argument('--softmax-layer-bias', type='bool', default=False,
                           help='Put True if you want to include the bias in '
                                'decoder.e2s layer')
        TorchAgent.add_cmdline_args(argparser)
        Seq2seqAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Set up model."""
        init_model = None
        if not shared:  # only do this for original instance
            # first check load path in case we need to override paths

            # check first for 'init_model' for loading model from file
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                init_model = opt['init_model']
            # next check for 'model_file', this would override init_model
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                init_model = opt['model_file']

            if init_model is not None:
                if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                    opt['dict_file'] = init_model + '.dict'
        super().__init__(opt, shared)
        opt = self.opt

        # all instances may need some params
        self.id = 'Seq2Seq'
        self.metrics = {'loss': 0.0, 'num_tokens': 0, 'correct_tokens': 0,
                        'total_skipped_batches': 0}
        self.report_freq = opt.get('report_freq', 0.001)
        self.beam_size = opt.get('beam_size', 1)
        self.topk = opt.get('topk', 1)
        states = {}

        if shared:
            # set up shared properties
            self.model = shared['model']
            self.metrics = shared['metrics']
            states = shared.get('states', {})

        else:
            # this is not a shared instance of this class, so do full init
            if init_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]'.format(init_model))
                states = self.load(init_model)

            if not hasattr(self, 'model_class'):
                # this allows child classes to override this but inherit init
                self.model_class = Seq2seq
            self.model = self.model_class(
                opt, len(self.dict), padding_idx=self.NULL_IDX,
                start_idx=self.START_IDX, end_idx=self.END_IDX,
                longest_label=states.get('longest_label', 1))

            if opt.get('dict_tokenizer') == 'bpe' and opt['embedding_type'] != 'random':
                print('skipping preinitialization of embeddings for bpe')
            elif not states and opt['embedding_type'] != 'random':
                # set up preinitialized embeddings
                try:
                    import torchtext.vocab as vocab
                except ImportError as ex:
                    print('Please install torch text with `pip install torchtext`')
                    raise ex
                pretrained_dim = 300
                if opt['embedding_type'].startswith('glove'):
                    if 'twitter' in opt['embedding_type']:
                        init = 'glove-twitter'
                        name = 'twitter.27B'
                        pretrained_dim = 200
                    else:
                        init = 'glove'
                        name = '840B'
                    embs = vocab.GloVe(name=name, dim=pretrained_dim,
                        cache=modelzoo_path(self.opt.get('datapath'),
                                            'models:glove_vectors')
                    )
                elif opt['embedding_type'].startswith('fasttext'):
                    init = 'fasttext'
                    embs = vocab.FastText(language='en',
                        cache=modelzoo_path(self.opt.get('datapath'),
                                            'models:fasttext_vectors')
                    )
                else:
                    raise RuntimeError('embedding type not implemented')

                if opt['embeddingsize'] != pretrained_dim:
                    rp = torch.Tensor(pretrained_dim, opt['embeddingsize']).normal_()
                    t = lambda x: torch.mm(x.unsqueeze(0), rp)
                else:
                    t = lambda x: x
                cnt = 0
                for w, i in self.dict.tok2ind.items():
                    if w in embs.stoi:
                        vec = t(embs.vectors[embs.stoi[w]])
                        self.model.decoder.lt.weight.data[i] = vec
                        cnt += 1
                        if opt['lookuptable'] in ['unique', 'dec_out']:
                            # also set encoder lt, since it's not shared
                            self.model.encoder.lt.weight.data[i] = vec
                print('Seq2seq: initialized embeddings for {} tokens from {}.'
                      ''.format(cnt, init))

            if states:
                # set loaded states if applicable
                self.model.load_state_dict(states['model'])

            if self.use_cuda:
                self.model.cuda()

        # set up criteria
        if opt.get('numsoftmax', 1) > 1:
            self.criterion = nn.NLLLoss(
                ignore_index=self.NULL_IDX, size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, size_average=False)

        if self.use_cuda:
            self.criterion.cuda()

        if 'train' in opt.get('datatype', ''):
            # we only set up optimizers when training
            self.clip = opt.get('gradient_clip', -1)

            # set up optimizer
            lr = opt['learningrate']
            optim_class = Seq2seqAgent.OPTIM_OPTS[opt['optimizer']]
            kwargs = {'lr': lr}
            if opt.get('momentum') > 0 and opt['optimizer'] in ['sgd', 'rmsprop']:
                kwargs['momentum'] = opt['momentum']
                if opt['optimizer'] == 'sgd':
                    kwargs['nesterov'] = True
            if opt['optimizer'] == 'adam':
                # amsgrad paper: https://openreview.net/forum?id=ryQu7f-RZ
                kwargs['amsgrad'] = True

            if opt['embedding_type'].endswith('fixed'):
                print('Seq2seq: fixing embedding weights.')
                self.model.decoder.lt.weight.requires_grad = False
                self.model.encoder.lt.weight.requires_grad = False
                if opt['lookuptable'] in ['dec_out', 'all']:
                    self.model.decoder.e2s.weight.requires_grad = False
            self.optimizer = optim_class([p for p in self.model.parameters() if p.requires_grad], **kwargs)
            if states.get('optimizer'):
                if states['optimizer_type'] != opt['optimizer']:
                    print('WARNING: not loading optim state since optim class '
                          'changed.')
                else:
                    try:
                        self.optimizer.load_state_dict(states['optimizer'])
                    except ValueError:
                        print('WARNING: not loading optim state since model '
                              'params changed.')
                    if self.use_cuda:
                        for state in self.optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.cuda()
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', factor=0.5, patience=3, verbose=True)

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

    def zero_grad(self):
        """Zero out optimizer."""
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        super().reset_metrics()
        self.metrics['loss'] = 0.0
        self.metrics['num_tokens'] = 0
        self.metrics['correct_tokens'] = 0

    def report(self):
        """Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        m = {}
        num_tok = self.metrics['num_tokens']
        if num_tok > 0:
            if self.metrics['correct_tokens'] > 0:
                m['token_acc'] = self.metrics['correct_tokens'] / num_tok
                m['correct_tokens'] = self.metrics['correct_tokens']
            m['loss'] = self.metrics['loss'] / num_tok
            m['num_tokens'] = num_tok
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
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['model'] = self.model
        if self.opt.get('numthreads', 1) > 1:
            # we're doing hogwild so share the model too
            if type(self.metrics) == dict:
                # move metrics and model to shared memory
                self.metrics = SharedTable(self.metrics)
                self.model.share_memory()
            shared['states'] = {  # don't share optimizer states
                'optimizer_type': self.opt['optimizer'],
            }
        shared['metrics'] = self.metrics  # do after numthreads check
        return shared

    def vectorize(self, *args, **kwargs):
        """Override vectorize for seq2seq."""
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        return super().vectorize(*args, **kwargs)

    def batchify(self, *args, **kwargs):
        """Override batchify options for seq2seq."""
        kwargs['sort'] = True  # need sorted for pack_padded
        return super().batchify(*args, **kwargs)

    def _init_cuda_buffer(self, batchsize):
        if self.use_cuda and not hasattr(self, 'buffer_initialized'):
            try:
                print('preinitializing pytorch cuda buffer')
                bsz = self.opt.get('batchsize', batchsize)
                maxlen = self.truncate or 180
                dummy = torch.ones(bsz, maxlen).long().cuda()
                sc = self.model(dummy, dummy)[1]
                loss = self.criterion(sc.view(-1, sc.size(-1)), dummy.view(-1))
                loss.backward()
                self.buffer_initialized = True
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    m = ('CUDA OOM: Lower batch size (-bs) from {} or lower max'
                         ' sequence length (-tr) from {}'.format(bsz, maxlen))
                    raise RuntimeError(m)
                else:
                    raise e

    def train_step(self, batch):
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
        self._init_cuda_buffer(batchsize)  # helps with memory usage
        self.model.train()
        self.zero_grad()
        preds = None
        try:
            out = self.model(batch.text_vec, batch.label_vec)

            # generated response
            preds, scores = out[0], out[1]

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
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
            else:
                raise e

    def _build_cands(self, batch):
        if not batch.candidates:
            return None, None
        cand_inds = [i for i in range(len(batch.candidates))
                     if batch.candidates[i]]
        cands = [batch.candidate_vecs[i] for i in cand_inds]
        for i, c in enumerate(cands):
            cands[i] = padded_tensor(c, use_cuda=self.use_cuda)[0]
        return cands, cand_inds

    def _pick_cands(self, cand_preds, cand_inds, cands):
        cand_replies = [None] * len(cands)
        for idx, order in enumerate(cand_preds):
            batch_idx = cand_inds[idx]
            cand_replies[batch_idx] = [cands[batch_idx][i] for i in order]
        return cand_replies

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        self.model.eval()
        cands, cand_inds = self._build_cands(batch)
        out = self.model(batch.text_vec, ys=None,
                         cands=cands, cand_indices=cand_inds,
                         beam_size=self.beam_size, topk=self.topk)
        preds, cand_preds = out[0], out[2]

        if batch.label_vec is not None:
            # calculate loss on targets
            out = self.model(batch.text_vec, batch.label_vec)
            f_preds, scores = out[0], out[1]
            score_view = scores.view(-1, scores.size(-1))
            loss = self.criterion(score_view, batch.label_vec.view(-1))
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == f_preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens

        cand_choices = None
        if cand_preds is not None:
            # now select the text of the cands based on their scores
            cand_choices = self._pick_cands(cand_preds, cand_inds,
                                            batch.candidates)

        text = [self._v2t(p) for p in preds]
        return Output(text, cand_choices)

    def save(self, path=None):
        """Save model parameters if model_file is set."""
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
            with open(path + ".opt", 'wb') as handle:
                pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        return states

    def receive_metrics(self, metrics_dict):
        """Use the metrics to decide when to adjust LR schedule."""
        if 'loss' in metrics_dict:
            self.scheduler.step(metrics_dict['loss'])


class mydefaultdict(defaultdict):
    """Custom defaultdict which overrides defaults requested by the get
    function with the default factory.
    """
    def get(self, key, default=None):
        # override default from "get" (like "__getitem__" already is)
        return super().get(key, default or self.default_factory())


class PerplexityEvaluatorAgent(Seq2seqAgent):
    """Subclass for doing standardized perplexity evaluation.

    This is designed to be used in conjunction with the PerplexityWorld at
    parlai/scripts/eval_ppl.py. It uses the `next_word_probability` function
    to calculate the probability of tokens one token at a time.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.prev_enc = None
        self.last_xs = None

    def next_word_probability(self, partial_out):
        """Return probability distribution over next words given an input and
        partial true output. This is used to calculate the per-word perplexity.

        Arguments:
        observation -- input observation dict
        partial_out -- list of previous "true" words

        Returns a dict, where each key is a word and each value is a probability
        score for that word. Unset keys assume a probability of zero.

        e.g.
        {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
        """
        obs = self.observation
        xs = obs['text_vec'].unsqueeze(0)
        ys = self._vectorize_text(' '.join(partial_out), False, True, self.truncate).unsqueeze(0)
        if self.prev_enc is not None and self.last_xs is not None and (
                xs.shape[1] != self.last_xs.shape[1] or
                (xs == self.last_xs).sum().item() != xs.shape[1]):
            # reset prev_enc, this is a new input
            self.prev_enc = None
        self.last_xs = xs

        self.model.eval()
        # no need to predict farther ahead
        # if you pass in any ys, this will be ignored
        self.model.longest_label = 1
        out = self.model(
            xs,
            ys=(ys if len(partial_out) > 0 else None),
            prev_enc=self.prev_enc)
        scores, self.prev_enc = out[1], out[4]
        # scores is bsz x seqlen x num_words, so select probs of current index
        probs = F.softmax(scores.select(1, -1), dim=1).squeeze()
        dist = mydefaultdict(lambda: 1e-7)  # default probability for any token
        for i in range(len(probs)):
            dist[self.dict[i]] = probs[i].item()
        return dist
