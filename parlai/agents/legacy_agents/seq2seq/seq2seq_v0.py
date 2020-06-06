#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .utils_v0 import modelzoo_path
from .dict_v0 import DictionaryAgent
from .utils_v0 import maintain_dialog_history, PaddingUtils, round_sigfigs
from .utils_v0 import SharedTable
from .modules_v0 import Seq2seq

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, defaultdict

import copy
import os
import math
import pickle


class Agent(object):
    """
    Base class for all other agents.
    """

    def __init__(self, opt, shared=None):
        if not hasattr(self, 'id'):
            self.id = 'agent'
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        self.observation = None

    def observe(self, observation):
        """
        Receive an observation/action dict.
        """
        self.observation = observation
        return observation

    def act(self):
        """
        Return an observation/action dict based upon given observation.
        """
        if hasattr(self, 'observation') and self.observation is not None:
            print('agent received observation:')
            print(self.observation)

        t = {}
        t['text'] = 'hello, teacher!'
        print('agent sending message:')
        print(t)
        return t

    def getID(self):
        return self.id

    def epoch_done(self):
        return False

    def reset(self):
        self.observation = None

    def reset_metrics(self):
        pass

    def save(self, path=None):
        """
        If applicable, save any parameters needed to recreate this agent from loaded
        parameters.
        """
        pass

    def share(self):
        """
        If applicable, share any parameters needed to create a shared version of this
        agent.
        """
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        return shared

    def shutdown(self):
        """
        Perform any final cleanup if needed.
        """
        pass


class Seq2seqAgent(Agent):
    """
    Agent which takes an input sequence and produces an output sequence.

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

    OPTIM_OPTS = {
        'adadelta': optim.Adadelta,  # type: ignore
        'adagrad': optim.Adagrad,  # type: ignore
        'adam': optim.Adam,
        'adamax': optim.Adamax,  # type: ignore
        'asgd': optim.ASGD,  # type: ignore
        'lbfgs': optim.LBFGS,  # type: ignore
        'rmsprop': optim.RMSprop,  # type: ignore
        'rprop': optim.Rprop,  # type: ignore
        'sgd': optim.SGD,
    }

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument(
            '--init-model',
            type=str,
            default=None,
            help='load dict/features/weights/opts from this file',
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
            '-lr', '--learningrate', type=float, default=1, help='learning rate'
        )
        agent.add_argument(
            '-dr', '--dropout', type=float, default=0.1, help='dropout rate'
        )
        agent.add_argument(
            '-clip',
            '--gradient-clip',
            type=float,
            default=0.1,
            help='gradient clipping using l2 norm',
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
            '--no-cuda',
            action='store_true',
            default=False,
            help='disable GPUs even if available',
        )
        agent.add_argument(
            '-gpu', '--gpu', type=int, default=-1, help='which GPU device to use'
        )
        # ranking arguments
        agent.add_argument(
            '-rc',
            '--rank-candidates',
            type='bool',
            default=False,
            help='rank candidates if available. this is done by'
            ' computing the prob score per token for each '
            'candidate and selecting the highest scoring.',
        )
        agent.add_argument(
            '-tr',
            '--truncate',
            type=int,
            default=-1,
            help='truncate input & output lengths to speed up '
            'training (may reduce accuracy). This fixes all '
            'input and output to have a maximum length. This '
            'reduces the total amount '
            'of padding in the batches.',
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
            '-opt',
            '--optimizer',
            default='sgd',
            choices=Seq2seqAgent.OPTIM_OPTS.keys(),
            help='Choose between pytorch optimizers. '
            'Any member of torch.optim is valid and will '
            'be used with default params except learning '
            'rate (as specified by -lr).',
        )
        agent.add_argument(
            '-mom',
            '--momentum',
            default=-1,
            type=float,
            help='if applicable, momentum value for optimizer. '
            'if > 0, sgd uses nesterov momentum.',
        )
        agent.add_argument(
            '-emb',
            '--embedding-type',
            default='random',
            choices=['random', 'glove', 'glove-fixed', 'fasttext', 'fasttext-fixed'],
            help='Choose between different strategies '
            'for word embeddings. Default is random, '
            'but can also preinitialize from Glove or '
            'Fasttext.'
            'Preinitialized embeddings can also be fixed '
            'so they are not updated during training.',
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
            '-rf',
            '--report-freq',
            type=float,
            default=0.001,
            help='Report frequency of prediction during eval.',
        )
        agent.add_argument(
            '-histr',
            '--history-replies',
            default='label_else_model',
            type=str,
            choices=['none', 'model', 'label', 'label_else_model'],
            help='Keep replies in the history, or not.',
        )
        agent.add_argument(
            '-pt',
            '--person-tokens',
            type='bool',
            default=False,
            help='use special tokens before each speaker',
        )
        agent.add_argument(
            '--beam-size',
            type=int,
            default=1,
            help='Beam size, if 1 then greedy search',
        )
        agent.add_argument(
            '--beam-log-freq',
            type=float,
            default=0.0,
            help='The portion of beams to dump from minibatch into model_name.beam_dump folder',
        )
        agent.add_argument(
            '--topk',
            type=int,
            default=1,
            help='Top k sampling from renormalized softmax in test/valid time, default 1 means simple greedy max output',
        )
        agent.add_argument(
            '--softmax-layer-bias',
            type='bool',
            default=False,
            help='Put True if you want to include the bias in decoder.e2s layer',
        )
        Seq2seqAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        super().__init__(opt, shared)
        opt = self.opt  # there is a deepcopy in the init

        # all instances may need some params
        self.truncate = opt['truncate'] if opt['truncate'] > 0 else None
        self.metrics = {
            'loss': 0.0,
            'num_tokens': 0,
            'correct_tokens': 0,
            'total_skipped_batches': 0,
        }
        self.history = {}
        self.report_freq = opt.get('report_freq', 0.001)
        self.use_person_tokens = opt.get('person_tokens', False)
        self.batch_idx = shared and shared.get('batchindex') or 0
        self.rank = opt['rank_candidates']
        self.beam_size = opt.get('beam_size', 1)
        self.topk = opt.get('topk', 1)
        states = {}

        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        if opt.get('numthreads', 1) > 1:
            torch.set_num_threads(1)

        if shared:
            # set up shared properties
            self.opt = shared['opt']
            opt = self.opt
            self.dict = shared['dict']
            self.START_IDX = shared['START_IDX']
            self.END_IDX = shared['END_IDX']
            self.NULL_IDX = shared['NULL_IDX']
            # answers contains a batch_size list of the last answer produced
            self.answers = shared['answers']
            self.model = shared['model']
            self.metrics = shared['metrics']
            states = shared.get('states', {})

        else:
            # this is not a shared instance of this class, so do full init
            # answers contains a batch_size list of the last answer produced
            self.answers = [None] * opt['batchsize']

            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            init_model = None
            # check first for 'init_model' for loading model from file
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                init_model = opt['init_model']
            # next check for 'model_file', this would override init_model
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                init_model = opt['model_file']

            if init_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]'.format(init_model))
                states = self.load(init_model)

                if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                    opt['dict_file'] = init_model + '.dict'

            # load dictionary and basic tokens & vectors
            self.dict = DictionaryAgent(opt)
            self.id = 'Seq2Seq'
            # we use START markers to start our output
            self.START_IDX = self.dict[self.dict.start_token]
            # we use END markers to end our output
            self.END_IDX = self.dict[self.dict.end_token]
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict[self.dict.null_token]

            if not hasattr(self, 'model_class'):
                # this allows child classes to override this but inherit init
                self.model_class = Seq2seq
            self.model = self.model_class(
                opt,
                len(self.dict),
                padding_idx=self.NULL_IDX,
                start_idx=self.START_IDX,
                end_idx=self.END_IDX,
                longest_label=states.get('longest_label', 1),
            )

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
                    init = 'glove'
                    name = '840B'
                    embs = vocab.GloVe(
                        name=name,
                        dim=pretrained_dim,
                        cache=modelzoo_path(
                            self.opt.get('datapath'), 'models:glove_vectors'
                        ),
                    )
                elif opt['embedding_type'].startswith('fasttext'):
                    init = 'fasttext'
                    embs = vocab.FastText(
                        language='en',
                        cache=modelzoo_path(
                            self.opt.get('datapath'), 'models:fasttext_vectors'
                        ),
                    )
                else:
                    raise RuntimeError('embedding type not implemented')

                if opt['embeddingsize'] != pretrained_dim:
                    rp = torch.Tensor(pretrained_dim, opt['embeddingsize']).normal_()
                else:
                    rp = None
                cnt = 0
                for w, i in self.dict.tok2ind.items():
                    if w in embs.stoi:
                        vec = embs.vectors[embs.stoi[w]]
                        if rp is not None:
                            vec = torch.mm(vec.unsqueeze(0), rp)
                        self.model.decoder.lt.weight.data[i] = vec
                        cnt += 1
                        if opt['lookuptable'] in ['unique', 'dec_out']:
                            # also set encoder lt, since it's not shared
                            self.model.encoder.lt.weight.data[i] = vec
                print(
                    'Seq2seq: initialized embeddings for {} tokens from {}.'
                    ''.format(cnt, init)
                )

            if states:
                # set loaded states if applicable
                self.model.load_state_dict(states['model'])

            if self.use_cuda:
                self.model.cuda()

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
            # we only set up optimizers when training
            # we only set this up for the original instance or hogwild ones
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
                # https://openreview.net/forum?id=ryQu7f-RZ
                kwargs['amsgrad'] = True

            if opt['embedding_type'].endswith('fixed'):
                print('Seq2seq: fixing embedding weights.')
                self.model.decoder.lt.weight.requires_grad = False
                self.model.encoder.lt.weight.requires_grad = False
                if opt['lookuptable'] in ['dec_out', 'all']:
                    self.model.decoder.e2s.weight.requires_grad = False
            self.optimizer = optim_class(
                [p for p in self.model.parameters() if p.requires_grad], **kwargs
            )
            if states.get('optimizer'):
                if states['optimizer_type'] != opt['optimizer']:
                    print(
                        'WARNING: not loading optim state since optim class ' 'changed.'
                    )
                else:
                    try:
                        self.optimizer.load_state_dict(states['optimizer'])
                    except ValueError:
                        print(
                            'WARNING: not loading optim state since model '
                            'params changed.'
                        )
                    if self.use_cuda:
                        for state in self.optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.cuda()
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', factor=0.5, patience=3, verbose=True
            )

        self.reset()

    def override_opt(self, new_opt):
        """
        Set overridable opts from loaded opt file.

        Print out each added key and each overriden key. Only override args specific to
        the model.
        """
        model_args = {
            'hiddensize',
            'embeddingsize',
            'numlayers',
            'optimizer',
            'encoder',
            'decoder',
            'lookuptable',
            'attention',
            'attention_length',
            'rnn_class',
        }
        for k, v in new_opt.items():
            if k not in model_args:
                # skip non-model args
                continue
            if k not in self.opt:
                print('[ Adding new option: | {k}: {v} | ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print(
                    '[ Overriding option: | {k}: {old} => {v} | ]'.format(
                        k=k, old=self.opt[k], v=v
                    )
                )
            self.opt[k] = v
        if 'dict_file' in new_opt and not self.opt.get('dict_file'):
            print(
                '[ No dictionary path detected, trying to load previous '
                'path {} ]'.format(new_opt['dict_file'])
            )
            self.opt['dict_file'] = new_opt['dict_file']
        return self.opt

    def parse(self, text):
        """
        Convert string to token indices.
        """
        return self.dict.txt2vec(text)

    def v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
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

    def reset(self):
        """
        Reset observation and episode_done.
        """
        self.observation = None
        self.history.clear()
        for i in range(len(self.answers)):
            self.answers[i] = None
        self.reset_metrics()

    def reset_metrics(self):
        """
        Reset metrics for reporting loss and perplexity.
        """
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
        shared['opt'] = self.opt
        shared['answers'] = self.answers
        shared['dict'] = self.dict
        shared['START_IDX'] = self.START_IDX
        shared['END_IDX'] = self.END_IDX
        shared['NULL_IDX'] = self.NULL_IDX
        shared['model'] = self.model
        if self.opt.get('numthreads', 1) > 1:
            # we're doing hogwild so share the model too
            if type(self.metrics) == dict:
                # move metrics and model to shared memory
                self.metrics = SharedTable(self.metrics)
                self.model.share_memory()
            shared['states'] = {  # don't share optimizer states
                'optimizer_type': self.opt['optimizer']
            }
        shared['metrics'] = self.metrics  # do after numthreads check
        return shared

    def observe(self, observation):
        """
        Save observation for act.

        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()

        if not obs.get('preprocessed', False) or 'text2vec' not in obs:
            obs['text2vec'] = maintain_dialog_history(
                self.history,
                obs,
                reply=self.answers[self.batch_idx],
                historyLength=self.truncate,
                useReplies=self.opt.get('history_replies'),
                dict=self.dict,
                useStartEndIndices=self.use_person_tokens,
            )
        else:
            obs['text2vec'] = deque(obs['text2vec'], maxlen=self.truncate)
        self.observation = obs
        self.answers[self.batch_idx] = None
        return obs

    def predict(self, xs, ys=None, cands=None, valid_cands=None, is_training=False):
        """
        Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank candidates as
        well if they are available and param is set.
        """
        predictions, cand_preds = None, None
        if is_training:
            self.model.train()
            self.zero_grad()
            out = None
            try:
                out = self.model(xs, ys, rank_during_training=cands is not None)
                # generated response
                _preds, scores, cand_preds = out[0], out[1], out[2]

                score_view = scores.view(-1, scores.size(-1))
                loss = self.criterion(score_view, ys.view(-1))
                # save loss to metrics
                y_ne = ys.ne(self.NULL_IDX)
                target_tokens = y_ne.long().sum().item()
                correct = ((ys == _preds) * y_ne).sum().item()
                self.metrics['correct_tokens'] += correct
                self.metrics['loss'] += loss.item()
                self.metrics['num_tokens'] += target_tokens
                loss /= target_tokens  # average loss per token
                loss.backward()
            except RuntimeError as e:
                # catch out of memory exceptions during fwd/bck (skip batch)
                if 'out of memory' in str(e):
                    print(
                        '| WARNING: ran out of memory, skipping batch. '
                        'if this happens frequently, decrease batchsize or '
                        'truncate the inputs to the model.'
                    )
                    self.metrics['total_skipped_batches'] += 1
                    return predictions, cand_preds
                else:
                    raise e
            self.update_params()
        else:
            self.model.eval()
            out = self.model(
                xs,
                ys=None,
                cands=cands,
                valid_cands=valid_cands,
                beam_size=self.beam_size,
                topk=self.topk,
            )
            predictions, cand_preds = out[0], out[2]

            if ys is not None:
                # calculate loss on targets
                out = self.model(xs, ys)
                scores = out[1]
                score_view = scores.view(-1, scores.size(-1))
                loss = self.criterion(score_view, ys.view(-1))
                # save loss to metrics
                target_tokens = ys.ne(self.NULL_IDX).long().sum().item()
                self.metrics['loss'] += loss.item()
                self.metrics['num_tokens'] += target_tokens

        return predictions, cand_preds

    def vectorize(self, observations):
        """
        Convert a list of observations into input & target tensors.
        """
        is_training = any(['labels' in obs for obs in observations])
        xs, ys, labels, valid_inds, _, _ = PaddingUtils.pad_text(
            observations,
            self.dict,
            end_idx=self.END_IDX,
            null_idx=self.NULL_IDX,
            dq=True,
            eval_labels=True,
            truncate=self.truncate,
        )

        if xs is None:
            return None, None, None, None, None, None, None
        xs = torch.LongTensor(xs)
        if ys is not None:
            ys = torch.LongTensor(ys)
        if self.use_cuda:
            # copy to gpu
            xs = xs.cuda()
            if ys is not None:
                ys = ys.cuda()

        cands = None
        valid_cands = None
        if not is_training and self.rank:
            # set up candidates
            cands = []
            valid_cands = []
            for i, v in enumerate(valid_inds):
                if 'label_candidates' in observations[v]:
                    curr_lcs = list(observations[v]['label_candidates'])
                    curr_cands = [{'text': c} for c in curr_lcs]
                    cs, _, _, valid_c_inds, *_ = PaddingUtils.pad_text(
                        curr_cands,
                        self.dict,
                        null_idx=self.NULL_IDX,
                        dq=True,
                        truncate=self.truncate,
                    )
                    valid_cands.append((i, v, [curr_lcs[j] for j in valid_c_inds]))
                    cs = torch.LongTensor(cs)
                    if self.use_cuda:
                        cs = cs.cuda()
                    cands.append(cs)

        return xs, ys, labels, valid_inds, cands, valid_cands, is_training

    def init_cuda_buffer(self, batchsize):
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
                    m = (
                        'CUDA OOM: Lower batch size (-bs) from {} or lower max'
                        ' sequence length (-tr) from {}'.format(bsz, maxlen)
                    )
                    raise RuntimeError(m)
                else:
                    raise e

    def batch_act(self, observations):
        batchsize = len(observations)
        self.init_cuda_buffer(batchsize)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, labels, valid_inds, cands, valid_cands, is_training = self.vectorize(
            observations
        )

        if xs is None:
            # no valid examples, just return empty responses
            return batch_reply

        # produce predictions, train on targets if availables
        cand_inds = [i[0] for i in valid_cands] if valid_cands is not None else None
        predictions, cand_preds = self.predict(xs, ys, cands, cand_inds, is_training)

        if is_training:
            report_freq = 0
        else:
            report_freq = self.report_freq
        if predictions is not None:
            PaddingUtils.map_predictions(
                predictions,
                valid_inds,
                batch_reply,
                observations,
                self.dict,
                self.END_IDX,
                report_freq=report_freq,
                labels=labels,
                answers=self.answers,
                ys=ys.data if ys is not None else None,
            )

        if cand_preds is not None:
            if valid_cands is None:
                valid_cands = [(None, i, labels) for i in valid_inds]
            for i in range(len(valid_cands)):
                order = cand_preds[i]
                _, batch_idx, curr_cands = valid_cands[i]
                curr = batch_reply[batch_idx]
                curr['text_candidates'] = [
                    curr_cands[idx] for idx in order if idx < len(curr_cands)
                ]

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

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
            with open(path + ".opt", 'wb') as handle:
                pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def shutdown(self):
        """
        Save the state of the model when shutdown.
        """
        path = self.opt.get('model_file', None)
        if path is not None and hasattr(self, 'optimizer'):
            self.save(path + '.shutdown_state')
        super().shutdown()

    def load(self, path):
        """
        Return opt and model states.
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        return states

    def receive_metrics(self, metrics_dict):
        """
        Use the metrics to decide when to adjust LR schedule.
        """
        if 'loss' in metrics_dict:
            self.scheduler.step(metrics_dict['loss'])


class mydefaultdict(defaultdict):
    """
    Custom defaultdict which overrides defaults requested by the get function with the
    default factory.
    """

    def get(self, key, default=None):
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
        super().__init__(opt, shared)
        self.prev_enc = None
        self.last_xs = None

    def next_word_probability(self, partial_out):
        """
        Return probability distribution over next words given an input and partial true
        output. This is used to calculate the per-word perplexity.

        Arguments:
        observation -- input observation dict
        partial_out -- list of previous "true" words

        Returns a dict, where each key is a word and each value is a probability
        score for that word. Unset keys assume a probability of zero.

        e.g.
        {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
        """
        obs = self.observation
        obs['eval_labels'] = [' '.join(partial_out)]
        batch = self.vectorize([obs])

        xs, ys = batch[0], batch[1]
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
        # no need to predict farther ahead
        # if you pass in any ys, this will be ignored
        self.model.longest_label = 1
        out = self.model(
            xs, ys=(ys if len(partial_out) > 0 else None), prev_enc=self.prev_enc
        )
        scores, self.prev_enc = out[1], out[4]
        # scores is bsz x seqlen x num_words, so select probs of current index
        probs = F.softmax(scores.select(1, -1), dim=1).squeeze()
        dist = mydefaultdict(lambda: 1e-7)  # default probability for any token
        for i in range(len(probs)):
            dist[self.dict[i]] = probs[i].item()
        return dist
