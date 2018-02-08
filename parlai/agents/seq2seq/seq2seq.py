# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import maintain_dialog_history, PaddingUtils
from .modules import Seq2seq, RandomProjection

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

from collections import deque

import os
import math
import random


class Seq2seqAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    several flavors of RNN. It then uses a linear layer (whose weights can
    be shared with the embedding layer) to convert RNN output states into
    output tokens. This model currently uses greedy decoding, selecting the
    highest probability token at each time step.

    For more information, see Sequence to Sequence Learning with Neural
    Networks `(Sutskever et al. 2014) <https://arxiv.org/abs/1409.3215>`_.
    """

    OPTIM_OPTS = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'sgd': optim.SGD,
    }

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        Seq2seqAgent.dictionary_class().add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.005,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-clip', '--gradient-clip', type=float, default=0.2,
                           help='gradient clipping using l2 norm')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'dot', 'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                'For more details see: '
                                'https://arxiv.org/pdf/1508.04025.pdf')
        agent.add_argument('-attl', '--attention-length', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-rc', '--rank-candidates', type='bool',
                           default=False,
                           help='rank candidates if available. this is done by'
                                ' computing the mean score per token for each '
                                'candidate and selecting the highest scoring.')
        agent.add_argument('-tr', '--truncate', type=int, default=-1,
                           help='truncate input & output lengths to speed up '
                           'training (may reduce accuracy). This fixes all '
                           'input and output to have a maximum length. This '
                           'reduces the total amount '
                           'of padding in the batches.')
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
        agent.add_argument('-lt', '--lookuptable', default='all',
                           choices=['unique', 'enc_dec', 'dec_out', 'all'],
                           help='The encoder, decoder, and output modules can '
                                'share weights, or not. '
                                'Unique has independent embeddings for each. '
                                'Enc_dec shares the embedding for the encoder '
                                'and decoder. '
                                'Dec_out shares decoder embedding and output '
                                'weights. '
                                'All shares all three weights.')
        agent.add_argument('-opt', '--optimizer', default='adam',
                           choices=Seq2seqAgent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        agent.add_argument('-emb', '--embedding-type', default='random',
                           choices=['random', 'glove', 'glove-fixed',
                                    'fasttext', 'fasttext-fixed'],
                           help='Choose between different strategies '
                                'for word embeddings. Default is random, '
                                'but can also preinitialize from Glove or '
                                'Fasttext.'
                                'Preinitialized embeddings can also be fixed '
                                'so they are not updated during training.')
        agent.add_argument('-hist', '--history-length', default=100000, type=int,
                           help='Number of past tokens to remember. '
                                'Default remembers 100000 tokens.')
        agent.add_argument('-histr', '--history-replies',
                           default='none', type=str,
                           choices=['none', 'model', 'label'],
                           help='Keep replies in the history, or not.')

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = self.opt  # there is a deepcopy in the init

        # all instances may need some params
        self.truncate = opt['truncate'] if opt['truncate'] > 0 else None
        self.history = {}
        self.states = {}

        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()


        if shared:
            # set up shared properties
            self.dict = shared['dict']
            self.START_IDX = shared['START_IDX']
            self.END_IDX = shared['END_IDX']
            self.NULL_IDX = shared['NULL_IDX']
            # answers contains a batch_size list of the last answer produced
            self.answers = shared['answers']

            if 'model' in shared:
                # model is shared during hogwild
                self.model = shared['model']
                self.states = shared['states']
        else:
            # this is not a shared instance of this class, so do full init
            # answers contains a batch_size list of the last answer produced
            self.answers = [None] * opt['batchsize']

            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, self.states = self.load(opt['model_file'])
                # override model-specific options with stored ones
                opt = self.override_opt(new_opt)

            if opt['dict_file'] is None and opt.get('model_file'):
                # set default dict-file if not set
                opt['dict_file'] = opt['model_file'] + '.dict'

            # load dictionary and basic tokens & vectors
            self.dict = DictionaryAgent(opt)
            self.id = 'Seq2Seq'
            # we use START markers to start our output
            self.START_IDX = self.dict[self.dict.start_token]
            # we use END markers to end our output
            self.END_IDX = self.dict[self.dict.end_token]
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict[self.dict.null_token]

            self.model = Seq2seq(opt, len(self.dict),
                                 padding_idx=self.NULL_IDX,
                                 start_idx=self.START_IDX,
                                 end_idx=self.END_IDX,
                                 longest_label=self.states.get('longest_label', 1))

            if opt['embedding_type'] != 'random':
                # set up preinitialized embeddings
                try:
                    import torchtext.vocab as vocab
                except ModuleNotFoundError as ex:
                    print('Please install torch text with `pip install torchtext`')
                    raise ex
                if opt['embedding_type'].startswith('glove'):
                    init = 'glove'
                    embs = vocab.GloVe(name='840B', dim=300)
                elif opt['embedding_type'].startswith('fasttext'):
                    init = 'fasttext'
                    embs = vocab.FastText(language='en')
                else:
                    raise RuntimeError('embedding type not implemented')

                if opt['embeddingsize'] != 300:
                    rp = torch.Tensor(300, opt['embeddingsize']).normal_()
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

            if self.states:
                # set loaded states if applicable
                self.model.load_state_dict(self.states['model'])

            if self.use_cuda:
                self.model.cuda()

        if hasattr(self, 'model'):
            # if model was built, do more setup
            self.clip = opt.get('gradient_clip', 0.2)
            self.rank = opt['rank_candidates']

            # set up tensors once
            self.xs = torch.LongTensor(1, 1)
            self.ys = torch.LongTensor(1, 1)
            if self.rank:
                self.cands = torch.LongTensor(1, 1, 1)

            # set up criteria
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX)

            if self.use_cuda:
                # push to cuda
                self.xs = self.xs.cuda(async=True)
                self.ys = self.ys.cuda(async=True)
                if self.rank:
                    self.cands = self.cands.cuda(async=True)
                self.criterion.cuda()

            # set up optimizer
            lr = opt['learningrate']
            optim_class = Seq2seqAgent.OPTIM_OPTS[opt['optimizer']]
            kwargs = {'lr': lr}
            if opt['optimizer'] == 'sgd':
                kwargs['momentum'] = 0.95
                kwargs['nesterov'] = True

            if opt['embedding_type'].endswith('fixed'):
                print('Seq2seq: fixing embedding weights.')
                self.model.decoder.lt.weight.requires_grad = False
                self.model.encoder.lt.weight.requires_grad = False
                if opt['lookuptable'] in ['dec_out', 'all']:
                    self.model.decoder.e2s.weight.requires_grad = False
            self.optimizer = optim_class([p for p in self.model.parameters() if p.requires_grad], **kwargs)
            if self.states:
                if self.states['optimizer_type'] != opt['optimizer']:
                    print('WARNING: not loading optim state since optim class '
                          'changed.')
                else:
                    self.optimizer.load_state_dict(self.states['optimizer'])

        self.reset()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                      'encoder', 'decoder', 'lookuptable', 'attention',
                      'attention_length'}
        for k, v in new_opt.items():
            if k not in model_args:
                # skip non-model args
                continue
            if k not in self.opt:
                print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('Overriding option [ {k}: {old} => {v}]'.format(
                      k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        return self.opt

    def parse(self, text):
        """Convert string to token indices."""
        return self.dict.txt2vec(text)

    def v2t(self, vec):
        """Convert token indices to string of tokens."""
        if type(vec) == Variable:
            vec = vec.data
        new_vec = []
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
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['answers'] = self.answers
        shared['dict'] = self.dict
        shared['START_IDX'] = self.START_IDX
        shared['END_IDX'] = self.END_IDX
        shared['NULL_IDX'] = self.NULL_IDX
        if self.opt.get('numthreads', 1) > 1:
            shared['model'] = self.model
            self.model.share_memory()
            shared['states'] = self.states
        return shared

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        batch_idx = self.opt.get('batchindex', 0)
        if not obs.get('preprocessed', False):
            obs['text2vec'] = maintain_dialog_history(
                self.history, obs,
                reply=self.answers[batch_idx],
                historyLength=self.opt['history_length'],
                useReplies=self.opt['history_replies'],
                dict=self.dict,
                useStartEndIndices=False)
        else:
            obs['text2vec'] = deque(obs['text2vec'], self.opt['history_length'])
        self.observation = obs
        self.answers[batch_idx] = None
        return obs

    def predict(self, xs, ys=None, cands=None, valid_cands=None):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available and param is set.
        """
        is_training = ys is not None
        text_cand_inds, loss_dict = None, None
        if is_training:
            self.model.train()
            self.zero_grad()
            loss = 0
            predictions, scores, _ = self.model(xs, ys)
            loss += self.criterion(scores.view(-1, scores.size(-1)), ys.view(-1))
            loss.backward()
            self.update_params()
            loss_dict = {'loss': loss.mul(len(xs)).data[0]}
            loss_dict['ppl'] = (math.e**loss).mul(len(xs)).data[0]
        else:
            self.model.eval()
            predictions, scores, text_cand_inds = self.model(xs, ys, cands,
                                                             valid_cands)

        return predictions, text_cand_inds, loss_dict

    def vectorize(self, observations):
        """Convert a list of observations into input & target tensors."""
        ys = None
        xs, ys, labels, valid_inds, _, _ = PaddingUtils.pad_text(
            observations, self.dict, self.END_IDX, self.NULL_IDX, dq=True,
            eval_labels=False, truncate=self.truncate)
        if xs is None:
            return None, None, None, None, None, None
        if self.use_cuda:
            # copy to gpu
            self.xs.resize_(xs.size())
            self.xs.copy_(xs, async=True)
            xs = Variable(self.xs)
            if ys is not None:
                self.ys.resize_(ys.size())
                self.ys.copy_(ys, async=True)
                ys = Variable(self.ys)
        else:
            xs = Variable(xs)
            if ys is not None:
                ys = Variable(ys)

        # set up candidates
        cands = None
        valid_cands = None
        if ys is None and self.rank:
            # only do ranking when no targets available and ranking flag set
            parsed_cs = []
            valid_cands = []
            for i, v in enumerate(valid_inds):
                if 'label_candidates' in observations[v]:
                    # each candidate tuple is a pair of the parsed version and
                    # the original full string
                    cs = list(observations[v]['label_candidates'])
                    curr_dqs = [deque(maxlen=self.truncate) for _ in cs]
                    for dq, c in zip(curr_dqs, cs):
                        dq.extendleft(reversed(self.parse(c)))
                    parsed_cs.append(curr_dqs)
                    valid_cands.append((i, v, cs))
            if len(parsed_cs) > 0:
                # TODO: store lengths of cands separately, so don't have zero
                #       padding for varying number of cands per example
                # found cands, pack them into tensor
                max_c_len = max(max(len(c) for c in cs) for cs in parsed_cs)
                max_c_cnt = max(len(cs) for cs in parsed_cs)
                for cs in parsed_cs:
                    for c in cs:
                        c += [self.NULL_IDX] * (max_c_len - len(c))
                    cs += [self.NULL_IDX] * (max_c_cnt - len(cs))
                cands = torch.LongTensor(parsed_cs)
                if self.use_cuda:
                    # copy to gpu
                    self.cands.resize_(cands.size())
                    self.cands.copy_(cands, async=True)
                    cands = Variable(self.cands)
                else:
                    cands = Variable(cands)

        return xs, ys, labels, valid_inds, cands, valid_cands

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, labels, valid_inds, cands, valid_cands = self.vectorize(observations)

        if xs is None:
            # no valid examples, just return empty responses
            return batch_reply

        # produce predictions, train on targets if availables
        predictions, text_cand_inds, loss = self.predict(xs, ys, cands, valid_cands)
        if loss is not None:
            if 'metrics' in batch_reply[0]:
                for k, v in loss.items():
                    batch_reply[0]['metrics'][k] = v
            else:
                batch_reply[0]['metrics'] = loss


        if ys is not None:
            report_freq = 0
        else:
            report_freq = 0.1
        PaddingUtils.map_predictions(
            predictions, valid_inds, batch_reply, observations, self.dict,
            self.END_IDX, report_freq=report_freq, labels=labels,
            answers=self.answers, ys=ys)

        if text_cand_inds is not None:
            text_cand_inds = text_cand_inds.cpu().data
            for i in range(len(valid_cands)):
                order = text_cand_inds[i]
                _, batch_idx, curr_cands = valid_cands[i]
                curr = batch_reply[batch_idx]
                curr['text_candidates'] = [curr_cands[idx] for idx in order
                                           if idx < len(curr_cands)]

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            model['model'] = self.model.state_dict()
            model['longest_label'] = self.model.longest_label
            model['optimizer'] = self.optimizer.state_dict()
            model['optimizer_type'] = self.opt['optimizer']
            model['opt'] = self.opt

            with open(path, 'wb') as write:
                torch.save(model, write)

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()

    def load(self, path):
        """Return opt and model states."""
        with open(path, 'rb') as read:
            states = torch.load(read)

        return states['opt'], states
