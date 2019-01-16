#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Author: Saizheng Zhang, work at Facebook AI Research, NYC
"""Contains generative models described in the paper Personalizing Dialogue
Agents: I have a dog, do you have pets too? `(Zhang et al. 2018)
<https://arxiv.org/pdf/1801.07243.pdf>`_.

Might need to do these:
pip install torchtext
pip install stop-words
"""
import os
import pickle
import copy
import random
import re
import time
import math

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.metrics import _f1_score
from parlai.core.utils import round_sigfigs
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
try:
    import torchtext.vocab as vocab
except ImportError:
    raise ImportError('Please `pip install torchtext`')

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

from parlai.core.params import ParlaiParser
ParlaiParser()  # instantiate unused parser to set PARLAI_HOME

stopwords_customized = []
with open(os.path.join(os.environ['PARLAI_HOME'], 'projects', 'personachat', 'stopwords.txt'), 'r') as handle:
    stopwords_customized = []
    for line in handle:
        if line == '\n':
            pass
        else:
            stopwords_customized.append(line.replace('\n', ''))

try:
    from stop_words import get_stop_words
except ImportError:
    raise ImportError('Please `pip install stop-words`')

STOP_WORDS = get_stop_words('en') + [',', '.', '!', '?']
STOP_WORDS.remove('not')
STOP_WORDS = STOP_WORDS + stopwords_customized


def tokenize(self, sent):
    words = [w.lower() for w in re.findall(r"[\w']+|[.,!?;:\']", sent)]
    return words


class Seq2seqAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.
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

    ENC_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=1024,
                           help='size of the hidden layers')
        agent.add_argument('-emb', '--embeddingsize', type=int, default=300,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.5,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-att', '--attention', default=None,
                           help='None, concat, general, local. If local,'
                                'something like "local,N" should be the keyword, '
                                'where N stands for specified attention length '
                                'while decoding. For more details see: '
                                'https://arxiv.org/pdf/1508.04025.pdf')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-rc', '--rank-candidates', type='bool',
                           default=False,
                           help='rank candidates if available. this is done by'
                                ' computing the mean score per token for each '
                                'candidate and selecting the highest scoring.')
        agent.add_argument('-tr', '--truncate', type=int, default=100,
                           help='truncate input & output lengths to speed up '
                           'training (may reduce accuracy). This fixes all '
                           'input and output to have a maximum length and to '
                           'be similar in length to one another by throwing '
                           'away extra tokens. This reduces the total amount '
                           'of padding in the batches.')
        agent.add_argument('-enc', '--encoder', default='lstm',
                           choices=Seq2seqAgent.ENC_OPTS.keys(),
                           help='Choose between different encoder modules.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'] + list(Seq2seqAgent.ENC_OPTS.keys()),
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights.')
        agent.add_argument('-opt', '--optimizer', default='adam',
                           choices=Seq2seqAgent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        # customized for personachat
        agent.add_argument('--personachat_interact', action='store_true',
                           help='interact during validation')
        agent.add_argument('--personachat_sharelt', action='store_true',
            help='share lookup table among self.lt, self.lt_per, self.e2o')
        agent.add_argument('--interactive-mode', type=bool, default=False,
            help='helps print nicer text during interactive mode')


    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        self.sharelt = opt['personachat_sharelt']
        self.interactive_mode = opt['interactive_mode']
        with open('s2s_opt.pkl', 'wb') as handle:
            pickle.dump(opt, handle)
        if shared:
            self.answers = shared['answers']
            self.START = shared['START']
            self.END = shared['END']
            self.truncate = shared['truncate']
            self.dict = shared['dict']
        else:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.

            # self.answer is tracking the last output from model itself.
            self.answers = [None] * opt['batchsize']
            self.log_perp = 0.
            self.n_log_perp = 0.

            # check for cuda
            self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            self.dict = DictionaryAgent(opt)

            states = None
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, states = self.load(opt['model_file'])
                # override options with stored ones
                opt = self.override_opt(new_opt)

            if opt.get('personachat_symbol_words', None):
                for w in opt['personachat_symbol_words']:
                    self.dict.add_to_dict([w])
            self.id = 'Seq2Seq'
            # we use START markers to start our output
            self.START = self.dict.start_token
            self.START_TENSOR = torch.LongTensor(self.dict.parse(self.START))
            # we use END markers to end our output
            self.END = self.dict.end_token
            self.END_TENSOR = torch.LongTensor(self.dict.parse(self.END))
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict.txt2vec(self.dict.null_token)[0]

            # reorder dictionary tokens
            self.dict.ind2tok[1] = '__END__'
            self.dict.tok2ind['__END__'] = 1
            self.dict.ind2tok[2] = '__UNK__'
            self.dict.tok2ind['__UNK__'] = 2
            self.dict.ind2tok[3] = '__START__'
            self.dict.tok2ind['__START__'] = 3

            # store important params directly
            hsz = opt['hiddensize']
            emb = opt['embeddingsize']
            self.hidden_size = hsz
            self.emb_size = emb
            self.num_layers = opt['numlayers']
            self.learning_rate = opt['learningrate']
            self.rank = opt['rank_candidates']
            self.longest_label = 1
            self.truncate = opt['truncate']
            self.attention = opt['attention']
            self.dropout = opt['dropout']

            # set up tensors
            self.zeros = torch.zeros(self.num_layers, 1, hsz)
            self.xs = torch.LongTensor(1, 1)
            self.ys = torch.LongTensor(1, 1)
            self.zs = torch.LongTensor(1, 1)
            self.cands = torch.LongTensor(1, 1, 1)
            self.cand_scores = torch.FloatTensor(1)
            self.cand_lengths = torch.LongTensor(1)

            # set up modules
            self.criterion = nn.NLLLoss()
            # lookup table stores word embeddings
            self.lt = nn.Embedding(len(self.dict), emb,
                                   padding_idx=self.NULL_IDX,
                                   scale_grad_by_freq=False)
            self.lt.weight[1:].data.normal_(0, 0.1)

            if not states:
                # initializing model from scratch, load glove vectors
                Glove = vocab.GloVe(
                    name='840B',
                    dim=300,
                    cache=os.path.join(
                        os.environ['PARLAI_HOME'],
                        'data',
                        'models',
                        'glove_vectors'
                    )
                )
                for w in self.dict.freq:
                    if w in Glove.stoi:
                        self.lt.weight.data[self.dict[w]] = Glove.vectors[Glove.stoi[w]]

            # encoder captures the input text
            enc_class = Seq2seqAgent.ENC_OPTS[opt['encoder']]
            self.encoder = enc_class(emb, hsz, opt['numlayers'], dropout=self.dropout)
            # decoder produces our output states
            if opt['decoder'] == 'shared':
                self.decoder = self.encoder
            elif opt['decoder'] == 'same':
                self.decoder = enc_class(emb, hsz, opt['numlayers'], dropout=self.dropout)
            else:
                dec_class = Seq2seqAgent.ENC_OPTS[opt['decoder']]
                self.decoder = dec_class(emb, hsz, opt['numlayers'], dropout=self.dropout)
            # linear layer helps us produce outputs from final decoder state
            if self.sharelt:
                self.h2e = nn.Linear(hsz, emb)
                self.e2o = nn.Linear(emb, len(self.dict))
                self.e2o.weight = self.lt.weight
            else:
                self.h2o = nn.Linear(hsz, len(self.dict) - 1)

            if not self.attention:
                pass
            elif self.attention.startswith('local'):
                self.max_length = int(self.attention.split(',')[1])
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz * 2, self.max_length)
                # combines attention weights with encoder outputs
                self.attn_combine = nn.Linear(hsz * 2, emb)

            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz * 2, hsz)
                self.attn_v = nn.Linear(hsz, 1)
                self.attn_combine = nn.Linear(hsz + emb, emb)
            elif self.attention == 'general':
                self.attn = nn.Linear(hsz, hsz)
                self.attn.weight.data = torch.eye(hsz)
                self.attn_combine = nn.Linear(hsz + emb, emb)

            # set up optims for each module
            lr = opt['learningrate']

            optim_class = Seq2seqAgent.OPTIM_OPTS[opt['optimizer']]
            self.optims = {
                'lt': optim_class(self.lt.parameters(), lr=lr),
                'encoder': optim_class(self.encoder.parameters(), lr=lr),
                'decoder': optim_class(self.decoder.parameters(), lr=lr),
            }
            if self.sharelt:
                self.optims['h2e'] = optim_class(self.h2e.parameters(), lr=lr)
                self.optims['e2o'] = optim_class(self.e2o.parameters(), lr=lr)
            else:
                self.optims['h2o'] = optim_class(self.h2o.parameters(), lr=lr)

            # load attention parameters into optims
            for attn_name in ['attn', 'attn_v', 'attn_combine']:
                if hasattr(self, attn_name):
                    self.optims[attn_name] = optim_class(getattr(self, attn_name).parameters(), lr=lr)

            if states:
                # set loaded states if applicable
                self.set_states(states)

            if self.use_cuda:
                self.cuda()

        self.loss = 0.0
        self.loss_c = 0
        self.reset()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.
        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                      'encoder', 'decoder', 'personachat_sharelt', 'personachat_reweight',
                      'personachat_guidesoftmax', 'personachat_useprevdialog', 'personachat_printattn',
                      'personachat_attnsentlevel', 'personachat_tfidfperp', 'personachat_learnreweight',
                      'personachat_embshareonly_pm_dec', 'attention'}
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
        return self.dict.vec2txt(vec)

    def cuda(self):
        """Push parameters to the GPU."""
        self.START_TENSOR = self.START_TENSOR.cuda()
        self.END_TENSOR = self.END_TENSOR.cuda()
        self.zeros = self.zeros.cuda()
        self.xs = self.xs.cuda()
        self.ys = self.ys.cuda()
        self.zs = self.zs.cuda()
        self.cands = self.cands.cuda()
        self.cand_scores = self.cand_scores.cuda()
        self.cand_lengths = self.cand_lengths.cuda()
        self.criterion.cuda()
        self.lt.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        if self.sharelt:
            self.h2e.cuda()
            self.e2o.cuda()
        else:
            self.h2o.cuda()
        if not self.attention:
            pass
        elif self.attention.startswith('local'):
            self.attn.cuda()
            self.attn_combine.cuda()
        elif self.attention == 'concat':
            self.attn.cuda()
            self.attn_v.cuda()
            self.attn_combine.cuda()
        elif self.attention == 'general':
            self.attn.cuda()
            self.attn_combine.cuda()
        for optimizer in self.optims.values():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    def hidden_to_idx(self, hidden, is_training=False):
        """Convert hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)

        hidden = F.dropout(hidden.squeeze(0), p=self.dropout, training=is_training)
        if self.sharelt:
            e = self.h2e(hidden)
            scores = self.e2o(e)[:,1:]
        else:
            scores = self.h2o(hidden)
        scores = F.log_softmax(scores)
        _max_score, idx = scores.max(1)

        return idx, scores

    def zero_grad(self):
        """Zero out optimizers."""
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        for optimizer in self.optims.values():
            optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def reset_log_perp(self):
        self.log_perp = 0.
        self.n_log_perp = 0.

    def share(self):
        shared = super().share()
        shared['answers'] = self.answers
        shared['START'] = self.START
        shared['END'] = self.END
        shared['truncate'] = self.truncate
        shared['dict'] = self.dict
        return shared

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            if self.truncate > 0:
                obv_parsed = [self.dict[w] for w in self.parse(observation['text'])]
                len_cur = len(obv_parsed)
                len_left = self.truncate - len_cur
                prev_dialogue = ' '.join([self.dict[w] for w in self.parse(prev_dialogue)][-len_left:]) if len_left > 0 else ''
                observation['text'] = ' '.join(obv_parsed) if len_left > 0 else ' '.join(obv_parsed[-self.truncate:])
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def _encode(self, xs, is_training=False):
        """Call encoder and return output and hidden states."""
        batchsize = len(xs)
        x_lens = [x for x in torch.sum((xs>0).int(), dim=1).data]

        # first encode context
        xes = self.lt(xs)

        xes = F.dropout(xes, p=2*self.dropout, training=is_training)
        # project from emb_size to hidden_size dimensions
        if self.zeros.size(1) != batchsize:
            self.zeros.resize_(self.num_layers, batchsize, self.hidden_size).fill_(0)
        h0 = Variable(self.zeros)
        xes_packed = pack_padded_sequence(xes.transpose(0, 1), x_lens)

        if type(self.encoder) == nn.LSTM:
            encoder_output_packed, hidden = self.encoder(xes_packed, (h0, h0))
            encoder_output, _ = pad_packed_sequence(encoder_output_packed)
            if type(self.decoder) != nn.LSTM:
                hidden = hidden[0]
        else:
            encoder_output_packed, hidden = self.encoder(xes_packed, h0)
            encoder_output, _ = pad_packed_sequence(encoder_output_packed)
            if type(self.decoder) == nn.LSTM:
                hidden = (hidden, h0)
        encoder_output = encoder_output.transpose(0, 1)

        if not self.attention:
            pass
        elif self.attention.startswith('local'):
            if encoder_output.size(1) > self.max_length:
                offset = encoder_output.size(1) - self.max_length
                encoder_output = encoder_output.narrow(1, offset, self.max_length)

        hidden = tuple(F.dropout(h, p=2*self.dropout, training=is_training) for h in hidden)
        return encoder_output, hidden

    def _apply_attention(self, xes, encoder_output, hidden, attn_mask=None):
        """Apply attention to encoder hidden layer."""
        if self.attention.startswith('concat'):
            hidden_expand = hidden[-1].unsqueeze(1).expand(hidden.size()[1], encoder_output.size()[1], hidden.size()[2])
            attn_w_premask = self.attn_v(F.tanh(self.attn(torch.cat((encoder_output, hidden_expand), 2)))).squeeze(2)
            attn_weights = F.softmax(attn_w_premask * attn_mask.float() - (1 - attn_mask.float()) * 1e20)

        if self.attention.startswith('general'):
            hidden_expand = hidden[-1].unsqueeze(1)
            attn_w_premask = torch.bmm(self.attn(hidden_expand), encoder_output.transpose(1, 2)).squeeze(1)
            attn_weights = F.softmax(attn_w_premask * attn_mask.float() - (1 - attn_mask.float()) * 1e20)

        if self.attention.startswith('local'):
            attn_weights = F.softmax(self.attn(torch.cat((xes[0], hidden[-1]), 1)))
            if attn_weights.size(1) > encoder_output.size(1):
                attn_weights = attn_weights.narrow(1, 0, encoder_output.size(1) )

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_output).squeeze(1)

        output = torch.cat((xes[0], attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.tanh(output)

        return output

    def _decode_and_train(self, batchsize, xes, ys, encoder_output, hidden, attn_mask):
        # update the model based on the labels
        self.zero_grad()
        loss = 0

        output_lines = [[] for _ in range(batchsize)]

        # keep track of longest label we've ever seen
        self.longest_label = max(self.longest_label, ys.size(1))
        for i in range(ys.size(1)):
            if type(self.decoder) == nn.LSTM:
                h = hidden[0]
            output = self._apply_attention(xes, encoder_output, h, attn_mask) if self.attention else xes

            output, hidden = self.decoder(output, hidden)
            preds, scores = self.hidden_to_idx(output, is_training=True)
            y = ys.select(1, i)
            loss += self.criterion(scores*y.ne(0).float().unsqueeze(1), (y-1)*y.ne(0).long())
            # use the true token as the next input instead of predicted
            # this produces a biased prediction but better training
            xes = self.lt(y).unsqueeze(0)
            xes = F.dropout(xes, p=self.dropout, training=True)
            for b in range(batchsize):
                # convert the output scores to tokens
                token = self.v2t([(preds+1).data[b]])
                output_lines[b].append(token)

        self.loss += loss.cpu().item()
        self.loss_c += 1
        loss.backward()
        self.update_params()

        if random.random() < 0.01 and not self.interactive_mode:
            # sometimes output a prediction for debugging
            print('prediction:', ' '.join(output_lines[0]),
                  '\nlabel:', self.dict.vec2txt(ys.data[0]))

        return output_lines

    def _decode_perp(self, batchsize, xes, ys, encoder_output, hidden, attn_mask, zs):
        # calculate the perplexity
        log_perp = 0.
        for i in range(zs.size(1)):
            if type(self.decoder) == nn.LSTM:
                h = hidden[0]
            if self.attention:
                output = self._apply_attention(xes, encoder_output, h, attn_mask)
            else:
                output = xes

            output, hidden = self.decoder(output, hidden)
            preds, scores = self.hidden_to_idx(output, is_training=False)
            y = zs.select(1, i)

            log_perp += scores[[i for i in range(len(y))], [int(k) for k in ((y-1)*y.ne(self.NULL_IDX).long()).cpu().data.numpy()]]*y.ne(self.NULL_IDX).float()
            # use the true token as the next input instead of predicted
            # this produces a biased prediction but better training
            xes = self.lt(y).unsqueeze(0)
        n_zs = zs.ne(self.NULL_IDX).float().sum()
        log_perp = (-log_perp).sum()
        self.log_perp += log_perp.cpu().item()
        self.n_log_perp += n_zs.cpu().item()


    def _decode_only(self, batchsize, xes, ys, encoder_output, hidden, attn_mask, zs):
        # just produce a prediction without training the model

        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0

        output_lines = [[] for _ in range(batchsize)]
        # now, generate a response from scratch
        while(total_done < batchsize) and max_len < self.longest_label:
            # keep producing tokens until we hit END or max length for each
            # example in the batch
            if type(self.decoder) == nn.LSTM:
                h = hidden[0]
            output = self._apply_attention(xes, encoder_output, h, attn_mask) if self.attention else xes

            output, hidden = self.decoder(output, hidden)
            preds, scores = self.hidden_to_idx(output, is_training=False)

            xes = self.lt((preds+1).unsqueeze(0))
            max_len += 1
            for b in range(batchsize):
                if not done[b]:
                    # only add more tokens for examples that aren't done yet
                    token = self.v2t([(preds+1).data[b]])
                    if token == self.END:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
                    else:
                        output_lines[b].append(token)

        if random.random() < 0.01 and not self.interactive_mode:
            # sometimes output a prediction for debugging
            print('prediction:', ' '.join(output_lines[0]))

        return output_lines

    def _score_candidates(self, cands, xe, encoder_output, hidden, attn_mask):
        # score each candidate separately

        # cands are exs_with_cands x cands_per_ex x words_per_cand
        # cview is total_cands x words_per_cand
        if type(self.decoder) == nn.LSTM:
            hidden, cell = hidden
        cview = cands.view(-1, cands.size(2))
        cands_xes = xe.expand(xe.size(0), cview.size(0), xe.size(2))
        sz = hidden.size()
        cands_hn = (
            hidden.view(sz[0], sz[1], 1, sz[2])
            .expand(sz[0], sz[1], cands.size(1), sz[2])
            .contiguous()
            .view(sz[0], -1, sz[2])
        )
        if type(self.decoder) == nn.LSTM:
            cands_cn = (
                cell.view(sz[0], sz[1], 1, sz[2])
                .expand(sz[0], sz[1], cands.size(1), sz[2])
                .contiguous()
                .view(sz[0], -1, sz[2])
            )

        sz = encoder_output.size()
        cands_encoder_output = (
            encoder_output.contiguous()
            .view(sz[0], 1, sz[1], sz[2])
            .expand(sz[0], cands.size(1), sz[1], sz[2])
            .contiguous()
            .view(-1, sz[1], sz[2])
        )

        sz = attn_mask.size()
        cands_attn_mask = (
            attn_mask.contiguous()
            .view(sz[0], 1, sz[1])
            .expand(sz[0], cands.size(1), sz[1])
            .contiguous()
            .view(-1, sz[1])
        )

        cand_scores = Variable(
                    self.cand_scores.resize_(cview.size(0)).fill_(0))
        cand_lengths = Variable(
                    self.cand_lengths.resize_(cview.size(0)).fill_(0))

        for i in range(cview.size(1)):
            output = self._apply_attention(cands_xes, cands_encoder_output, cands_hn, cands_attn_mask) if self.attention else cands_xes
            if type(self.decoder) == nn.LSTM:
                output, (cands_hn, cands_cn) = self.decoder(output, (cands_hn, cands_cn))
            else:
                output, cands_hn = self.decoder(output, cands_hn)
            preds, scores = self.hidden_to_idx(output, is_training=False)
            cs = cview.select(1, i)
            non_nulls = cs.ne(self.NULL_IDX)
            cand_lengths += non_nulls.long()
            score_per_cand = torch.gather(scores, 1, ((cs-1)*cs.ne(0).long()).unsqueeze(1))
            cand_scores += score_per_cand.squeeze() * non_nulls.float()
            cands_xes = self.lt(cs).unsqueeze(0)

        # set empty scores to -1, so when divided by 0 they become -inf
        cand_scores -= cand_lengths.eq(0).float()
        # average the scores per token
        cand_scores /= cand_lengths.float()

        cand_scores = cand_scores.view(cands.size(0), cands.size(1))
        srtd_scores, text_cand_inds = cand_scores.sort(1, True)
        text_cand_inds = text_cand_inds.data

        return text_cand_inds

    def predict(self, xs, ys=None, cands=None, zs=None):
        """Produce a prediction from our model.
        Update the model using the targets if available, otherwise rank
        candidates as well if they are available.
        """
        batchsize = len(xs)
        text_cand_inds = None
        is_training = ys is not None

        self.encoder.train(mode=is_training)
        self.decoder.train(mode=is_training)
        encoder_output, hidden = self._encode(xs, is_training)


        # next we use END as an input to kick off our decoder
        x = Variable(self.START_TENSOR)
        xe = self.lt(x).unsqueeze(1)
        xes = xe.expand(xe.size(0), batchsize, xe.size(2))

        # list of output tokens for each example in the batch
        output_lines = None

        attn_mask = Variable(xs.data.ne(0), requires_grad=False)

        if is_training:
            output_lines = self._decode_and_train(batchsize, xes, ys,
                                                  encoder_output, hidden, attn_mask)

        else:
            if cands is not None:
                text_cand_inds = self._score_candidates(cands, xe,
                                                        encoder_output, hidden, attn_mask)

            output_lines = self._decode_only(batchsize, xes, ys,
                                             encoder_output, hidden, attn_mask, zs)
            if zs is not None:
                self._decode_perp(batchsize, xes, ys,
                                  encoder_output, hidden, attn_mask, zs)

        return output_lines, text_cand_inds

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        def valid(obs):
            # check if this is an example our model should actually process
            return 'text' in obs and len(obs['text']) > 0
        # valid examples and their indices
        try:
            valid_inds, exs = zip(*[(i, ex) for i, ex in
                                    enumerate(observations) if valid(ex)])
        except ValueError:
            # zero examples to process in this batch, so zip failed to unpack
            return None, None, None, None, None, None, None, None

        # set up the input tensors
        batchsize = len(exs)

        # `x` text is already tokenized and truncated
        parsed = [self.parse(ex['text']) for ex in exs]
        x_lens = [len(x) for x in parsed]
        ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

        exs = [exs[k] for k in ind_sorted]
        valid_inds = [valid_inds[k] for k in ind_sorted]
        parsed = [parsed[k] for k in ind_sorted]

        max_x_len = max([len(x) for x in parsed])
        xs = torch.LongTensor(batchsize, max_x_len).fill_(self.NULL_IDX)
        # right-padded with zeros
        for i, x in enumerate(parsed):
            for j, idx in enumerate(x):
                xs[i][j] = idx

        if self.use_cuda:
            # copy to gpu
            self.xs.resize_(xs.size())
            self.xs.copy_(xs, )
            xs = Variable(self.xs)
        else:
            xs = Variable(xs)

        # set up the target tensors
        ys = None
        labels = None
        if any(['labels' in ex for ex in exs]):
            # randomly select one of the labels to update on, if multiple
            # append END to each label
            labels = [random.choice(ex.get('labels', [''])) for ex in exs]
            parsed = [self.parse(y + ' ' + self.END) for y in labels if y]
            max_y_len = max(len(y) for y in parsed)
            if self.truncate > 0 and max_y_len > self.truncate:
                parsed = [y[:self.truncate] for y in parsed]
                max_y_len = self.truncate
            ys = torch.LongTensor(batchsize, max_y_len).fill_(self.NULL_IDX)
            for i, y in enumerate(parsed):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            if self.use_cuda:
                # copy to gpu
                self.ys.resize_(ys.size())
                self.ys.copy_(ys, )
                ys = Variable(self.ys)
            else:
                ys = Variable(ys)


        # set up the target tensors for validation and test
        zs = None
        eval_labels = None
        if any(['eval_labels' in ex for ex in exs]):
            # randomly select one of the labels to update on, if multiple
            # append END to each label
            eval_labels = [random.choice(ex.get('eval_labels', [''])) for ex in exs]
            parsed = [self.parse(y + ' ' + self.END) for y in eval_labels if y]
            max_y_len = max(len(y) for y in parsed)
            if self.truncate > 0 and max_y_len > self.truncate:
                parsed = [y[:self.truncate] for y in parsed]
                max_y_len = self.truncate
            zs = torch.LongTensor(batchsize, max_y_len).fill_(self.NULL_IDX)
            for i, y in enumerate(parsed):
                for j, idx in enumerate(y):
                    zs[i][j] = idx
            if self.use_cuda:
                # copy to gpu
                self.zs.resize_(zs.size())
                self.zs.copy_(zs, )
                zs = Variable(self.zs)
            else:
                zs = Variable(zs)


        # set up candidates
        cands = None
        valid_cands = None
        if ys is None and self.rank:
            # only do ranking when no targets available and ranking flag set
            parsed = []
            valid_cands = []
            for i, v in enumerate(valid_inds):
                if 'label_candidates' in observations[v]:
                    # each candidate tuple is a pair of the parsed version and
                    # the original full string
                    cs = list(observations[v]['label_candidates'])
                    parsed.append([self.parse(c) for c in cs])
                    valid_cands.append((i, v, cs))
            if len(parsed) > 0:
                # TODO: store lengths of cands separately, so don't have zero
                #       padding for varying number of cands per example
                # found cands, pack them into tensor
                max_c_len = max(max(len(c) for c in cs) for cs in parsed)
                max_c_cnt = max(len(cs) for cs in parsed)
                cands = torch.LongTensor(len(parsed), max_c_cnt, max_c_len).fill_(self.NULL_IDX)
                for i, cs in enumerate(parsed):
                    for j, c in enumerate(cs):
                        for k, idx in enumerate(c):
                            cands[i][j][k] = idx
                if self.use_cuda:
                    # copy to gpu
                    self.cands.resize_(cands.size())
                    self.cands.copy_(cands, )
                    cands = Variable(self.cands)
                else:
                    cands = Variable(cands)
        return xs, ys, labels, valid_inds, cands, valid_cands, zs, eval_labels

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field

        if self.opt['datatype'] in ['valid', 'test'] and self.opt['personachat_interact']:
            print('OBSVS:' + observations[0]['text'])

        xs, ys, labels, valid_inds, cands, valid_cands, zs, eval_labels = self.batchify(observations)

        if xs is None:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available

        predictions, text_cand_inds = self.predict(xs, ys, cands, zs)

        if self.opt['datatype'] in ['valid', 'test'] and self.opt['personachat_interact']:
            print('MODEL:' + ' '.join(predictions[0]))
            print('TRUE :' + observations[0]['eval_labels'][0])

        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            curr = batch_reply[valid_inds[i]]
            curr['text'] = ' '.join(c for c in (predictions[i][:predictions[i].index(self.END)] if self.END in predictions[i] else predictions[i]))
            curr_pred = curr['text']
            if labels is not None:
                self.answers[valid_inds[i]] = labels[i]
            else:
                self.answers[valid_inds[i]] = curr_pred

        if text_cand_inds is not None:
            for i in range(len(valid_cands)):
                order = text_cand_inds[i]
                _ , batch_idx, curr_cands = valid_cands[i]
                curr = batch_reply[batch_idx]
                curr['text_candidates'] = [curr_cands[idx] for idx in order
                                           if idx < len(curr_cands)]

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'lt'):
            model = {}
            model['lt'] = self.lt.state_dict()
            model['encoder'] = self.encoder.state_dict()
            model['decoder'] = self.decoder.state_dict()
            if self.sharelt:
                model['h2e'] = self.h2e.state_dict()
                model['e2o'] = self.e2o.state_dict()
            else:
                model['h2o'] = self.h2o.state_dict()
            for attn_name in ['attn', 'attn_v', 'attn_combine']:
                if hasattr(self, attn_name):
                    model[attn_name] = getattr(self, attn_name).state_dict()
            model['optims'] = {k: v.state_dict()
                               for k, v in self.optims.items()}
            model['longest_label'] = self.longest_label
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
            model = torch.load(read, map_location=lambda cpu, _: cpu)

        return model['opt'], model

    def set_states(self, states):
        """Set the state dicts of the modules from saved states."""
        self.lt.load_state_dict(states['lt'])
        self.encoder.load_state_dict(states['encoder'])
        self.decoder.load_state_dict(states['decoder'])
        if self.sharelt:
            self.h2e.load_state_dict(states['h2e'])
            self.e2o.load_state_dict(states['e2o'])
        else:
            self.h2o.load_state_dict(states['h2o'])
        for attn_name in ['attn', 'attn_v', 'attn_combine']:
            if hasattr(self, attn_name):
                getattr(self, attn_name).load_state_dict(states[attn_name])
        for k, v in states['optims'].items():
            self.optims[k].load_state_dict(v)
        self.longest_label = states['longest_label']

    def report_loss(self):
        print("The loss is {}".format(self.loss/(self.loss_c+1e-10)))
        self.loss = 0.0
        self.loss_c = 0


class PersonachatSeqseqAgentBasic(Seq2seqAgent):
    """
    This agent can choose if we will use the persona(s) or not, and if we will
    use previous history or not. If so, they will be concatenated in a sequential
    manner by persona(s) first and then previous history.
    """
    @staticmethod
    def add_cmdline_args(argparser):
        Seq2seqAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('PersonachatSeqseqAgentBasic Arguments')
        agent.add_argument('--personachat_useprevdialog', action='store_true', \
            help='using previous dialog history')

    def __init__(self, opt, shared=None):
        self.usepersona = opt['task'].split(':', 1)[1]
        self.usepreviousdialog = opt['personachat_useprevdialog']
        self.batch_idx = shared and shared.get('batchindex') or 0
        super().__init__(opt, shared)

    def observe(self, obs):
        observation = obs.copy()
        if len(observation) == 1 and 'episode_done' in observation:
            # this is the case where observation = {'episode_done': True}
            self.observation = observation
            self.episode_done = observation['episode_done']
            return observation
        else:
            if self.episode_done == True:
                self.prev_dialog = ''
                self.last_obs = ''
                self.persona_given = ''
            text_split = observation['text'].split('\n')
            if self.usepersona:
                self.persona_given = ''
                for t in text_split:
                    if 'persona' in t:
                        t = t.replace('your persona: ', '').replace('their persona: ', '')
                        self.persona_given += t +'\n'
            else:
                if self.usepreviousdialog:
                    self.prev_dialog += self.last_obs if self.last_obs == '' else self.last_obs + '\n'
                    if self.answers[self.batch_idx] is not None and self.prev_dialog != '':
                        self.prev_dialog += self.answers[self.batch_idx] + '\n'
                    self.answers[self.batch_idx] = None
            observation['text'] = text_split[-1]
            self.last_obs = observation['text']
            self.episode_done = observation['episode_done']
            observation['text'] = self.persona_given + self.prev_dialog + observation['text']
            self.observation = observation
            return observation


class PersonachatSeqseqAgentSplit(Agent):
    """Agent which takes an input sequence and produces an output sequence.
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

    ENC_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=1024,
                           help='size of the hidden layers')
        agent.add_argument('-emb', '--embeddingsize', type=int, default=300,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.5,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-bi', '--bidirectional', action='store_true',
                           help='bidirectional')
        agent.add_argument('-att', '--attention', default=None,
                           help='None, concat, general, local. If local,'
                                'something like "local,N" should be the keyword, '
                                'where N stands for specified attention length '
                                'while decoding. For more details see: '
                                'https://arxiv.org/pdf/1508.04025.pdf')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-rc', '--rank-candidates', type='bool',
                           default=False,
                           help='rank candidates if available. this is done by'
                                ' computing the mean score per token for each '
                                'candidate and selecting the highest scoring.')
        agent.add_argument('-tr', '--truncate', type=int, default=100,
                           help='truncate input & output lengths to speed up '
                           'training (may reduce accuracy). This fixes all '
                           'input and output to have a maximum length and to '
                           'be similar in length to one another by throwing '
                           'away extra tokens. This reduces the total amount '
                           'of padding in the batches.')
        agent.add_argument('-enc', '--encoder', default='lstm',
                           choices=Seq2seqAgent.ENC_OPTS.keys(),
                           help='Choose between different encoder modules.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'] + list(Seq2seqAgent.ENC_OPTS.keys()),
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights.')
        agent.add_argument('-opt', '--optimizer', default='adam',
                           choices=Seq2seqAgent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        agent.add_argument('--personachat_useprevdialog', action='store_true', \
            help='using previous dialog history')
        agent.add_argument('--personachat_printattn', action='store_true', \
            help='save attn')
        agent.add_argument('--personachat_attnsentlevel', action='store_true', \
            help='use dowue-arthur attn')
        agent.add_argument('--personachat_sharelt', action='store_true', \
            help='share lookup table among self.lt, self.lt_per, self.e2o')
        agent.add_argument('--personachat_reweight', default=None, \
            help='use reweight persona word embedding')
        agent.add_argument('--personachat_guidesoftmax', action='store_true', \
            help='guide softmax weights by similarity during training')
        agent.add_argument('--personachat_newsetting', default='', \
            help='new settings like downweight,sharpsoftmax')
        agent.add_argument('--personachat_interact', action='store_true', \
            help='interact during validation')
        agent.add_argument('--personachat_pdmn', action='store_true', \
            help='use profile dialog memory network')
        agent.add_argument('--personachat_tfidfperp', action='store_true', \
            help='use tf-idf perplexity')
        agent.add_argument('--personachat_learnreweight', action='store_true', \
            help='set reweight to be learnable')
        agent.add_argument('--personachat_embshareonly_pm_dec', action='store_true', \
            help='set reweight to be learnable')
        agent.add_argument('--personachat_s2sinit', action='store_true', \
            help='init use s2s model')
        agent.add_argument('--interactive-mode', type=bool, default=False,
            help='helps print nicer text during interactive mode')
        agent.add_argument('--use-persona', type=str, default='self',
            choices=['self', 'none', 'other', 'both'],
            help='if task does not specify persona, specify here')


    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        self.interactive_mode = opt['interactive_mode']
        try:
            self.usepersona = opt['task'].split(':', 1)[1]
        except:
            self.usepersona = opt['use_persona']
        self.usepreviousdialog = opt['personachat_useprevdialog']
        self.attnsentlevel = opt['personachat_attnsentlevel']
        self.sharelt = opt['personachat_sharelt']
        self.reweight = opt['personachat_reweight']
        self.newsetting = opt['personachat_newsetting']
        self.embshareonly_pm_dec = opt['personachat_embshareonly_pm_dec']
        self.s2sinit = opt['personachat_s2sinit']
        self.batch_idx = shared and shared.get('batchindex') or 0
        self.metrics = {'loss': 0, 'num_tokens': 0}

        if shared:
            self.answers = shared['answers']
            self.START = shared['START']
            self.END = shared['END']
            self.dict = shared['dictionary']
        else:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.

            # self.answer is tracking the last output from model itself.
            self.answers = [None] * opt['batchsize']

            # monitor perplexity
            self.log_perp = 0.
            self.n_log_perp = 0.

            # monitor sim_score
            self.sim_score = 0.
            self.n_sim_score = 0.

            # attn_w_visual
            self. attn_w_visual_list = []

            # check for cuda
            self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            self.dict = DictionaryAgent(opt)

            states = None
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                opt['model_file'] = opt['model_file']
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, states = self.load(opt['model_file'])
                # override options with stored ones
                opt = self.override_opt(new_opt)
                self.usepreviousdialog = opt['personachat_useprevdialog']
                self.attnsentlevel = opt['personachat_attnsentlevel']
                self.sharelt = opt['personachat_sharelt']
                self.reweight = opt['personachat_reweight']
                self.newsetting = opt['personachat_newsetting']
                self.embshareonly_pm_dec = opt['personachat_embshareonly_pm_dec']
                self.s2sinit = opt['personachat_s2sinit']

            if opt.get('personachat_symbol_words', None):
                for w in opt['personachat_symbol_words']:
                    self.dict.add_to_dict([w])
            self.id = 'Seq2Seq'
            # we use START markers to start our output
            self.START = self.dict.start_token
            self.START_TENSOR = torch.LongTensor([self.dict[self.START]])
            # we use END markers to end our output
            self.END = self.dict.end_token
            self.END_TENSOR = torch.LongTensor([self.dict[self.END]])
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict[self.dict.null_token]

            # reorder dictionary tokens
            self.dict.ind2tok[1] = self.dict.end_token
            self.dict.tok2ind[self.dict.end_token] = 1
            self.dict.ind2tok[2] = self.dict.unk_token
            self.dict.tok2ind[self.dict.unk_token] = 2
            self.dict.ind2tok[3] = self.dict.start_token
            self.dict.tok2ind[self.dict.start_token] = 3

            # store important params directly
            hsz = opt['hiddensize']
            emb = opt['embeddingsize']
            self.hidden_size = hsz
            self.emb_size = emb
            self.num_layers = opt['numlayers']
            self.learning_rate = opt['learningrate']
            self.rank = opt['rank_candidates']
            self.longest_label = 1
            self.truncate = opt['truncate']
            self.attention = opt['attention']
            self.dropout = opt['dropout']
            self.bidirectional = opt['bidirectional']
            self.num_dirs = 2 if self.bidirectional else 1
            self.printattn_time = time.time()

            # set up tensors
            self.zeros = torch.zeros(self.num_layers * self.num_dirs, 1, hsz)
            self.xs = torch.LongTensor(1, 1)
            self.xs_persona = torch.LongTensor(1, 1)
            self.ys = torch.LongTensor(1, 1)
            self.zs = torch.LongTensor(1, 1)
            self.cands = torch.LongTensor(1, 1, 1)
            self.cand_scores = torch.FloatTensor(1)
            self.cand_lengths = torch.LongTensor(1)

            # set up modules
            self.criterion = nn.NLLLoss()
            self.criterion_guidesoftmax = nn.NLLLoss()

            # lookup table stores word embeddings
            self.lt = nn.Embedding(len(self.dict), emb,
                                   padding_idx=self.NULL_IDX,
                                   scale_grad_by_freq=False)
            self.lt.weight[1:].data.normal_(0, 0.1)

            # lookup table for persona embedding
            self.lt_per = nn.Embedding(len(self.dict), emb,
                                       padding_idx=self.NULL_IDX,
                                       scale_grad_by_freq=False)
            self.lt_per.weight[1:].data.uniform_(0, 1)
            if self.sharelt:
                self.lt_per.weight = self.lt.weight

            if self.embshareonly_pm_dec:
                # lookup table for persona embedding
                self.lt_enc = nn.Embedding(len(self.dict), emb,
                                           padding_idx=self.NULL_IDX,
                                           scale_grad_by_freq=False)
                self.lt_enc.weight[1:].data.uniform_(0, 1)

            #lookup table for rescale perplexity
            self.lt_rescaleperp = nn.Embedding(len(self.dict), 1,
                                               padding_idx=self.NULL_IDX,
                                               scale_grad_by_freq=False)

            #lookup table for reweight
            self.lt_reweight = nn.Embedding(len(self.dict), 1,
                                            padding_idx=self.NULL_IDX,
                                            scale_grad_by_freq=False)

            if not states:
                # initializing model from scratch, load glove vectors
                Glove = vocab.GloVe(
                    name='840B',
                    dim=300,
                    cache=os.path.join(
                        os.environ['PARLAI_HOME'],
                        'data',
                        'models',
                        'glove_vectors'
                    )
                )
                for w in self.dict.freq:
                    if w in Glove.stoi:
                        self.lt.weight.data[self.dict[w]] = Glove.vectors[Glove.stoi[w]]

                if not self.sharelt:
                    for w in self.dict.freq:
                        if w in Glove.stoi:
                            self.lt_per.weight.data[self.dict[w]] = Glove.vectors[Glove.stoi[w]]

                if self.embshareonly_pm_dec:
                    for w in self.dict.freq:
                        if w in Glove.stoi:
                            self.lt_enc.weight.data[self.dict[w]] = Glove.vectors[Glove.stoi[w]]

                for w in self.dict.freq:
                    self.lt_reweight.weight.data[self.dict[w]] = self.f_word(Glove, w)

                for w in self.dict.freq:
                    self.lt_rescaleperp.weight.data[self.dict[w]] = self.f_word_2(Glove, w, usetop=True, th=3000)

            self.lt_meta = copy.deepcopy(self.lt_per)
            self.lt_meta.weight.requires_grad = False

            self.lt_reweight_meta = copy.deepcopy(self.lt_reweight)
            self.lt_reweight_meta.weight.requires_grad = False

            # encoder captures the input text
            enc_class = Seq2seqAgent.ENC_OPTS[opt['encoder']]
            self.encoder = enc_class(emb, hsz, opt['numlayers'], dropout=self.dropout, bidirectional=self.bidirectional)
            self.encoder_persona = enc_class(emb, hsz, opt['numlayers'], dropout=self.dropout, bidirectional=self.bidirectional)

            # decoder produces our output states
            if opt['decoder'] == 'shared':
                self.decoder = self.encoder
            elif opt['decoder'] == 'same':
                self.decoder = enc_class(emb, hsz, opt['numlayers'], dropout=self.dropout)
            else:
                dec_class = Seq2seqAgent.ENC_OPTS[opt['decoder']]
                self.decoder = dec_class(emb, hsz, opt['numlayers'], dropout=self.dropout)
            # linear layer helps us produce outputs from final decoder state
            if self.sharelt:
                self.h2e = nn.Linear(hsz, emb)
                self.e2o = nn.Linear(emb, len(self.dict))
                self.e2o.weight = self.lt.weight
            else:
                self.h2o = nn.Linear(hsz, len(self.dict) - 1)

            hszXdirs = hsz * self.num_dirs
            if not self.attention:
                pass
            elif self.attention.startswith('local'):
                self.max_length = int(self.attention.split(',')[1])
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz * 2, self.max_length)
                # combines attention weights with encoder outputs
                self.attn_combine = nn.Linear(hsz * 2, emb)

            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz)
                self.attn_v = nn.Linear(hsz, 1)
                self.attn_combine = nn.Linear(hszXdirs + emb, emb)
            elif self.attention == 'general':
                if self.opt['personachat_attnsentlevel']:
                    self.attn = nn.Linear(emb, emb)
                    self.attn.weight.data = torch.eye(emb)
                    self.attn_h2attn = nn.Linear(hsz, emb)
                    self.attn_combine = nn.Linear(emb + emb, emb)
                else:
                    self.attn = nn.Linear(hsz, hszXdirs)
                    self.attn.weight.data = torch.eye(hsz)
                    self.attn_combine = nn.Linear(hszXdirs + emb, emb)

            self.persona_gate = Variable(torch.zeros(hsz).cuda(), requires_grad=True) \
                                if self.use_cuda else Variable(torch.zeros(hsz), requires_grad=True)
            self.persona_gate_m = Variable(torch.zeros(hsz).cuda(), requires_grad=True) \
                                if self.use_cuda else Variable(torch.zeros(hsz), requires_grad=True)
            # set up optims for each module
            lr = opt['learningrate']

            optim_class = Seq2seqAgent.OPTIM_OPTS[opt['optimizer']]
            self.optims = {
                'lt': optim_class(self.lt.parameters(), lr=lr),
                'lt_per':  optim_class(self.lt_per.parameters(), lr=lr),
                'encoder': optim_class(self.encoder.parameters(), lr=lr),
                'encoder_persona': optim_class(self.encoder_persona.parameters(), lr=lr),
                'decoder': optim_class(self.decoder.parameters(), lr=lr),
                'persona_gate': optim_class([self.persona_gate], lr=lr),
                'persona_gate_m': optim_class([self.persona_gate_m], lr=lr)
            }
            if self.embshareonly_pm_dec:
                self.optims['lt_enc'] = optim_class(self.lt_enc.parameters(), lr=lr)
            if self.opt['personachat_learnreweight']:
                self.optims['lt_reweight'] = optim_class(self.lt_reweight.parameters(), lr=lr)
            if self.sharelt:
                self.optims['h2e'] = optim_class(self.h2e.parameters(), lr=lr)
                self.optims['e2o'] = optim_class(self.e2o.parameters(), lr=lr)
            else:
                self.optims['h2o'] = optim_class(self.h2o.parameters(), lr=lr)

            if self.reweight and 'learn' in self.reweight:
                self.optims['lt_reweight'] = optim_class(self.h2o.parameters(), lr=lr)

            # load attention parameters into optims
            for attn_name in ['attn', 'attn_v', 'attn_combine', 'attn_h2attn']:
                if hasattr(self, attn_name):
                    self.optims[attn_name] = optim_class(getattr(self, attn_name).parameters(), lr=lr)

            if states:
                # set loaded states if applicable
                self.set_states(states)

            if self.s2sinit and self.sharelt:
                with open('s2s_opt.pkl', 'rb') as handle:
                    s2s_opt = pickle.load(handle)
                s2s_opt, s2s_states = self.load(s2s_opt['model_file'])
                for part in ['encoder', 'decoder', 'h2e', 'e2o', 'lt']:
                    getattr(self, part).load_state_dict(s2s_states[part])

            if self.use_cuda:
                self.cuda()

        self.loss = 0.0
        self.loss_c = 0
        if self.opt['personachat_guidesoftmax']:
            self.loss_guide = 0.
        self.reset()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.
        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                      'encoder', 'decoder', 'personachat_sharelt', 'personachat_reweight',
                      'personachat_guidesoftmax', 'personachat_useprevdialog', 'personachat_printattn',
                      'personachat_attnsentlevel', 'personachat_tfidfperp', 'personachat_learnreweight',
                      'personachat_embshareonly_pm_dec', 'attention'}
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
        return self.dict.vec2txt(vec)

    def cuda(self):
        """Push parameters to the GPU."""
        self.START_TENSOR = self.START_TENSOR.cuda()
        self.END_TENSOR = self.END_TENSOR.cuda()
        self.zeros = self.zeros.cuda()
        self.xs = self.xs.cuda()
        self.xs_persona = self.xs_persona.cuda()
        self.ys = self.ys.cuda()
        self.zs = self.zs.cuda()
        self.cands = self.cands.cuda()
        self.cand_scores = self.cand_scores.cuda()
        self.cand_lengths = self.cand_lengths.cuda()
        self.persona_gate.cuda()
        self.persona_gate_m.cuda()
        self.criterion.cuda()
        self.lt.cuda()
        self.lt_per.cuda()
        self.lt_meta.cuda()
        self.lt_reweight.cuda()
        self.lt_reweight_meta.cuda()
        self.lt_rescaleperp.cuda()
        if self.embshareonly_pm_dec:
            self.lt_enc.cuda()
        self.encoder.cuda()
        self.encoder_persona.cuda()
        self.decoder.cuda()
        if self.sharelt:
            self.h2e.cuda()
            self.e2o.cuda()
        else:
            self.h2o.cuda()
        if not self.attention:
            pass
        elif self.attention.startswith('local'):
            self.attn.cuda()
            self.attn_combine.cuda()
        elif self.attention == 'concat':
            self.attn.cuda()
            self.attn_v.cuda()
            self.attn_combine.cuda()
        elif self.attention == 'general':
            self.attn.cuda()
            if self.opt['personachat_attnsentlevel']:
                self.attn_h2attn.cuda()
            self.attn_combine.cuda()
        for optimizer in self.optims.values():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    def hidden_to_idx(self, hidden, is_training=False, topk=False):
        """Convert hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)
        hidden = F.dropout(hidden.squeeze(0), p=self.dropout, training=is_training)
        if self.sharelt:
            e = self.h2e(hidden)
            scores = self.e2o(e)[:,1:]
        else:
            scores = self.h2o(hidden)
        scores = F.log_softmax(scores)
        _max_score, idx = scores.max(1)
        if topk:
            _, idx = torch.topk(scores, 20)
        return idx, scores

    def zero_grad(self):
        """Zero out optimizers."""
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        for optimizer in self.optims.values():
            optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def reset_log_perp(self):
        self.log_perp = 0.
        self.n_log_perp = 0.

    def reset_sim_score(self):
        self.sim_score = 0.
        self.n_sim_score = 0.

    def share(self):
        shared = super().share()
        shared['answers'] = self.answers
        shared['START'] = self.START
        shared['END'] = self.END
        shared['dictionary'] = self.dict
        return shared

    def f_word(self, Glove, word, ifvector=True):
        stop_words = STOP_WORDS + [
            self.dict.start_token, self.dict.unk_token, self.dict.end_token,
            self.dict.null_token,
        ]
        w = word
        if w in stop_words:
            return 0.
        try:
            glove_idx = Glove.stoi[w] if w != ',' else 1
            f_w = 1e6 * 1/(glove_idx)**1.07
        except:
            f_w = 0.
        value = 1.0 / (1.0 + math.log(1.0 + f_w))
        return value

    def f_word_2(self, Glove, word, usetop=True, th=500):
        stop_words = [
            self.dict.start_token, self.dict.unk_token, self.dict.end_token,
            self.dict.null_token,
        ]
        w = word
        if w in stop_words:
            return 0.
        try:
            glove_idx = Glove.stoi[w] if w != ',' else 1
            f_w = 1e6 * 1/(glove_idx)**1.07
            if usetop:
                if glove_idx > th:
                    return 0.
            else:
                if glove_idx < th:
                    return 0.
        except:
            f_w = 0. if usetop else 1.
        value = 1.0 if f_w != 0. else 0.
        return value


    def observe(self, obs):
        observation = obs.copy()
        if len(observation) == 1 and 'episode_done' in observation:
           # this is the case where observation = {'episode_done': True}
           self.observation = observation
           self.episode_done = observation['episode_done']
           return observation
        elif 'text' in observation:
            if self.episode_done == True:
                self.prev_dialog = ''
                self.last_obs = ''
                self.persona_given = ''
            text_split = observation['text'].split('\n')
            if self.usepersona:
                self.persona_given = ''
                for t in text_split:
                    if 'persona' in t:
                        t = t.replace('your persona: ', '').replace('their persona: ', '')
                        self.persona_given += t +'\n'
            else:
                if self.usepreviousdialog:
                    self.prev_dialog += self.last_obs if self.last_obs == '' else self.last_obs + '\n'
                    if self.answers[self.batch_idx] is not None and self.prev_dialog != '':
                        self.prev_dialog += self.answers[self.batch_idx] + '\n'
                    self.answers[self.batch_idx] = None
            observation['text'] = text_split[-1]
            self.last_obs = observation['text']
            self.episode_done = observation['episode_done']
            observation['text'] = self.prev_dialog + observation['text']
            observation['persona'] = self.persona_given

            if len(observation['persona']) == 0:
                observation['persona'] = '__START__'
            self.observation = observation
            return observation
        else:
            self.observation = observation
            return observation

    def _encode_persona(self, xs, ys=None, is_training=False):
        """Call encoder and return output and hidden states."""

        guide_indices = None
        if self.opt['personachat_guidesoftmax'] and ys is not None:
            xs_size = xs.size()
            xes = self.lt_meta(xs.view(xs_size[0], xs_size[1] * xs_size[2]))
            xes = xes.view(xs_size[0], xs_size[1], xs_size[2], xes.size(2))
            f_xes = self.lt_reweight_meta(xs.view(xs_size[0], xs_size[1] * xs_size[2]))
            f_xes = f_xes.view(xs_size[0], xs_size[1], xs_size[2], f_xes.size(2))
            f_xes_norm = f_xes.sum(2).unsqueeze(2)
            f_xes_norm = f_xes_norm + (1. - f_xes_norm.ne(0).float())
            f_xes = f_xes / f_xes_norm
            xes = xes * f_xes
            xes = xes.sum(dim=2)

            yes = self.lt_meta(ys)
            f_yes = self.lt_reweight_meta(ys)
            f_yes_norm = f_yes.sum(1).unsqueeze(1)
            f_yes = f_yes / (f_yes_norm + 1e-10)
            yes = yes * f_yes
            yes = yes.sum(dim=1)

            random_guidesoftmax = True
            #TODO: increasing temperature of guide softmax
            if 'sharpsoftmax' in self.newsetting:
                temp = 3
            else:
                temp = 1
            if random_guidesoftmax:
                pre_softmax = torch.bmm(xes, yes.unsqueeze(2)).squeeze()
                mask = pre_softmax.ne(0)
                p_indices = F.softmax(temp * pre_softmax * mask.float() - (1 - mask.float()) * 1e20) * mask.float()
                guide_indices = torch.multinomial(p_indices, 1)[:,0]
                self.p_indices = p_indices
                self.guide_indices = guide_indices
                self.pre_softmax = pre_softmax
            else:
                values, guide_indices = torch.max(torch.bmm(xes, yes.unsqueeze(2)).squeeze(), 1)

        if self.opt['personachat_attnsentlevel']:
            xs_size = xs.size()
            xes = self.lt_per(xs.view(xs_size[0], xs_size[1] * xs_size[2]))
            xes = xes.view(xs_size[0], xs_size[1], xs_size[2], xes.size(2))
            xes = F.dropout(xes, p=self.dropout, training=is_training)

            if self.reweight:
                f_xes = self.lt_reweight(xs.view(xs_size[0], xs_size[1] * xs_size[2]))
                f_xes = f_xes.view(xs_size[0], xs_size[1], xs_size[2], f_xes.size(2))
                f_xes_norm = f_xes.sum(2).unsqueeze(2)
                f_xes_norm = f_xes_norm + (1. - f_xes_norm.ne(0).float())
                f_xes = f_xes / f_xes_norm
                xes = xes * f_xes
                xes = xes.sum(dim=2)
            else:
                xes = xes.sum(dim=2)

            return xes, None, guide_indices

        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs)

        if self.zeros.size(1) != batchsize:
            self.zeros.resize_(self.num_layers * self.num_dirs, batchsize, self.hidden_size).fill_(0)
        h0 = Variable(self.zeros)
        xes = xes.transpose(0,1)

        if type(self.encoder_persona) == nn.LSTM:
            encoder_output, hidden = self.encoder_persona(xes, (h0, h0))
            if type(self.decoder) != nn.LSTM:
                hidden = hidden[0]
        else:
            encoder_output, hidden = self.encoder(xes, h0)
            if type(self.decoder) == nn.LSTM:
                hidden = (hidden, h0)
        encoder_output = encoder_output.transpose(0, 1)

        if not self.attention:
            pass
        elif self.attention.startswith('local'):
            if encoder_output.size(1) > self.max_length:
                offset = encoder_output.size(1) - self.max_length
                encoder_output = encoder_output.narrow(1, offset, self.max_length)

        return encoder_output, hidden, guide_indices


    def _encode(self, xs, is_training=False):
        """Call encoder and return output and hidden states."""
        batchsize = len(xs)

        # first encode context
        if self.embshareonly_pm_dec:
            xes = self.lt_enc(xs)
        else:
            xes = self.lt(xs)
        xes = F.dropout(xes, p=2*self.dropout, training=is_training)

        if self.zeros.size(1) != batchsize:
            self.zeros.resize_(self.num_layers, batchsize, self.hidden_size).fill_(0)
        h0 = Variable(self.zeros)
        xes = xes.transpose(0, 1)

        if type(self.encoder) == nn.LSTM:
            encoder_output, hidden = self.encoder(xes, (h0, h0))
            if type(self.decoder) != nn.LSTM:
                hidden = hidden[0]
        else:
            encoder_output, hidden = self.encoder(xes, h0)
            if type(self.decoder) == nn.LSTM:
                hidden = (hidden, h0)
        encoder_output = encoder_output.transpose(0, 1)

        if not self.attention:
            pass
        elif self.attention.startswith('local'):
            if encoder_output.size(1) > self.max_length:
                offset = encoder_output.size(1) - self.max_length
                encoder_output = encoder_output.narrow(1, offset, self.max_length)

        #dropout
        if self.newsetting != '':
            hidden = tuple(F.dropout(h, p=2*self.dropout, training=is_training) for h in hidden)
        return encoder_output, hidden

    def _apply_attention(self, xes, encoder_output, hidden, attn_mask=None):
        """Apply attention to encoder hidden layer."""
        if self.attention.startswith('concat'):
            hidden_expand = hidden[-1].unsqueeze(1).expand(hidden.size()[1], encoder_output.size()[1], hidden.size()[2])
            attn_w_premask = self.attn_v(F.tanh(self.attn(torch.cat((encoder_output, hidden_expand), 2)))).squeeze(2)
            attn_w_premask = attn_w_premask * attn_mask.float() - (1 - attn_mask.float()) * 1e20
            attn_weights = F.softmax(attn_w_premask)

        if self.attention.startswith('general'):
            hidden_expand = hidden[-1].unsqueeze(1)
            attn_w_premask = torch.bmm(self.attn(hidden_expand), encoder_output.transpose(1, 2)).squeeze(1)
            attn_w_premask = attn_w_premask * attn_mask.float() - (1 - attn_mask.float()) * 1e20
            attn_weights = F.softmax(attn_w_premask)

        if self.attention.startswith('local'):
            attn_weights = F.softmax(self.attn(torch.cat((xes[0], hidden[-1]), 1)))
            if attn_weights.size(1) > encoder_output.size(1):
                attn_weights = attn_weights.narrow(1, 0, encoder_output.size(1) )

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_output).squeeze(1)
        output = torch.cat((xes[0], attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.tanh(output)

        return output, attn_weights, attn_w_premask

    def _decode_and_train(self, batchsize, xes, ys, encoder_output_persona, hidden_persona, hidden, attn_mask, guide_indices):
        # update the model based on the labels
        self.zero_grad()
        loss = 0
        loss_guide = 0

        output_lines = [[] for _ in range(batchsize)]

        # keep track of longest label we've ever seen
        self.longest_label = max(self.longest_label, ys.size(1))

        if self.opt['personachat_attnsentlevel']:
            hidden = hidden
        else:
            hidden = hidden

        attn_w_visual_tmp = []
        for i in range(ys.size(1)):
            if type(self.decoder) == nn.LSTM:
                h = hidden[0]
            if self.attention:
                if self.opt['personachat_attnsentlevel']:
                    h = self.attn_h2attn(h)
                output, attn_weights, attn_w_premask = self._apply_attention(xes, encoder_output_persona, h, attn_mask)
            else:
                output = xes

            output, hidden = self.decoder(output, hidden)
            preds, scores = self.hidden_to_idx(output, is_training=True)
            y = ys.select(1, i)

            loss += self.criterion(scores*y.ne(self.NULL_IDX).float().unsqueeze(1), (y-1)*y.ne(self.NULL_IDX).long())

            if self.opt['personachat_guidesoftmax']:
                loss_guidesoftmax = self.criterion_guidesoftmax(
                                    F.log_softmax(attn_w_premask)*y.ne(self.NULL_IDX).float().unsqueeze(1), guide_indices*y.ne(self.NULL_IDX).long())

                #TODO: down-weight loss_guidesoftmax
                if 'downweight' in self.newsetting:
                    loss_guidesoftmax = 0.5 * loss_guidesoftmax
                loss += loss_guidesoftmax
                loss_guide += loss_guidesoftmax

            if self.opt['personachat_printattn']:
                attn_weights = [n for n in attn_weights.cpu().item()]
                if self.opt['personachat_attnsentlevel']:
                    attn_words = [' '.join([w for w in self.dict.tokenize(self.dict.vec2txt(per))]) for per in self.parsed[0]]
                else:
                    attn_words = [w for w in self.dict.tokenize(self.dict.vec2txt(self.parsed[0]))]
                word_pred = self.dict[int(y.cpu().item())]
                attn_w_visual_tmp.append((word_pred, attn_weights, attn_words))

            # use the true token as the next input instead of predicted
            # this produces a biased prediction but better training
            if self.embshareonly_pm_dec:
                xes = self.lt_enc(y).unsqueeze(0)
            else:
                xes = self.lt(y).unsqueeze(0)
            xes = F.dropout(xes, p=self.dropout, training=True)
            for b in range(batchsize):
                # convert the output scores to tokens
                token = self.v2t([(preds+1).data[b]])
                output_lines[b].append(token)

        if self.opt['personachat_printattn']:
            self.attn_w_visual_list.append(attn_w_visual_tmp)
            with open('attn_w_visual.pkl', 'wb') as handle:
                pickle.dump(self.attn_w_visual_list, handle)
            self.printattn_time = time.time()

        self.loss += loss.cpu().item()
        if hasattr(self, 'loss_guide'):
            self.loss_guide += loss_guide.cpu().item()
        self.loss_c += 1
        loss.backward()
        self.update_params()

        if random.random() < 0.01 and not self.interactive_mode:
            # sometimes output a prediction for debugging
            print('prediction:', ' '.join(output_lines[0]),
                  '\nlabel:', self.dict.vec2txt(ys.data[0]))

        return output_lines

    def _decode_perp(self, batchsize, xes, ys, encoder_output_persona, hidden_persona, hidden, attn_mask, zs):
        # calculate the perplexity
        log_perp = 0.
        for i in range(zs.size(1)):
            if type(self.decoder) == nn.LSTM:
                h = hidden[0]
            if self.attention:
                if self.opt['personachat_attnsentlevel']:
                    h = self.attn_h2attn(h)
                output, attn_weights, attn_w_premask = self._apply_attention(xes, encoder_output_persona, h, attn_mask)
            else:
                output = xes

            output, hidden = self.decoder(output, hidden)
            preds, scores = self.hidden_to_idx(output, is_training=False)
            y = zs.select(1, i)
            if self.opt['personachat_tfidfperp']:
                log_perp += self.lt_rescaleperp(y)[:, 0]*scores[[i for i in range(len(y))], [int(k) for k in ((y-1)*y.ne(self.NULL_IDX).long()).cpu().data.numpy()]]*y.ne(self.NULL_IDX).float()
            else:
                log_perp += scores[[i for i in range(len(y))], [int(k) for k in ((y-1)*y.ne(self.NULL_IDX).long()).cpu().data.numpy()]]*y.ne(self.NULL_IDX).float()
            # use the true token as the next input instead of predicted
            # this produces a biased prediction but better training
            if self.embshareonly_pm_dec:
                xes = self.lt_enc(y).unsqueeze(0)
            else:
                xes = self.lt(y).unsqueeze(0)
        if self.opt['personachat_tfidfperp']:
            n_zs = zs.ne(self.NULL_IDX).float().sum()
        else:
            n_zs = zs.ne(self.NULL_IDX).float().sum()
        log_perp = (-log_perp).sum()
        self.log_perp += log_perp.cpu().item()
        self.n_log_perp += n_zs.cpu().item()
        self.metrics['loss'] += log_perp.cpu().item()
        self.metrics['num_tokens'] += n_zs.cpu().item()


    def _decode_only(self, batchsize, xes, ys, encoder_output_persona, hidden_persona, hidden, attn_mask, zs):

        # just produce a prediction without training the model
        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0

        output_lines = [[] for _ in range(batchsize)]

        if self.opt['personachat_attnsentlevel']:
            hidden = hidden
        else:
            hidden = hidden

        # now, generate a response from scratch
        while(total_done < batchsize) and max_len < self.longest_label:
            # keep producing tokens until we hit END or max length for each
            # example in the batch
            if type(self.decoder) == nn.LSTM:
                h = hidden[0]
            if self.attention:
                if self.opt['personachat_attnsentlevel']:
                    h = self.attn_h2attn(h)
                output,attn_weight, _ = self._apply_attention(xes, encoder_output_persona, h, attn_mask)
            else:
                output = xes

            output, hidden = self.decoder(output, hidden)

            topk = True

            if topk:
                word_used = [[] for _ in range(batchsize)]

            preds, scores = self.hidden_to_idx(output, is_training=False, topk=topk)

            if topk:
                new_preds = []
                for b in range(batchsize):
                    for pk in list(preds.cpu().data.numpy()[b]):
                        wk = self.v2t([pk])
                        if wk not in word_used[b] or wk in STOP_WORDS:
                            word_used[b].append(wk)
                            break
                    new_preds.append(wk)
                for b in range(batchsize):
                    preds[b, 0] = self.parse(new_preds[b])[0]
                preds = preds[:,0]
            if self.embshareonly_pm_dec:
                xes = self.lt_enc((preds + 1).unsqueeze(0))
            else:
                xes = self.lt((preds + 1).unsqueeze(0))
            max_len += 1
            for b in range(batchsize):
                if not done[b]:
                    # only add more tokens for examples that aren't done yet
                    pred_idx = (preds + 1)[b].item()
                    if pred_idx == self.dict[self.END]:
                        done[b] = True
                        total_done += 1
                    elif pred_idx != self.dict[self.START]:
                        output_lines[b].append(self.dict[pred_idx])

        if random.random() < 1 and not self.interactive_mode:
            # sometimes output a prediction for debugging
            print('prediction:', ' '.join(output_lines[0]))

        return output_lines

    def _score_candidates(self, cands, xe, encoder_output_persona, hidden_persona, hidden, attn_mask):
        # score each candidate separately

        # cands are exs_with_cands x cands_per_ex x words_per_cand
        # cview is total_cands x words_per_cand

        if self.opt['personachat_attnsentlevel']:
            hidden = hidden
        else:
            hidden = hidden

        if type(self.decoder) == nn.LSTM:
            hidden, cell = hidden
        cview = cands.view(-1, cands.size(2))
        cands_xes = xe.expand(xe.size(0), cview.size(0), xe.size(2))
        sz = hidden.size()
        cands_hn = (
            hidden.view(sz[0], sz[1], 1, sz[2])
            .expand(sz[0], sz[1], cands.size(1), sz[2])
            .contiguous()
            .view(sz[0], -1, sz[2])
        )
        if type(self.decoder) == nn.LSTM:
            cands_cn = (
                cell.view(sz[0], sz[1], 1, sz[2])
                .expand(sz[0], sz[1], cands.size(1), sz[2])
                .contiguous()
                .view(sz[0], -1, sz[2])
            )

        sz = encoder_output_persona.size()
        cands_encoder_output_persona = (
            encoder_output_persona.contiguous()
            .view(sz[0], 1, sz[1], sz[2])
            .expand(sz[0], cands.size(1), sz[1], sz[2])
            .contiguous()
            .view(-1, sz[1], sz[2])
        )

        sz = attn_mask.size()
        cands_attn_mask = (
            attn_mask.contiguous()
            .view(sz[0], 1, sz[1])
            .expand(sz[0], cands.size(1), sz[1])
            .contiguous()
            .view(-1, sz[1])
        )

        cand_scores = Variable(
                    self.cand_scores.resize_(cview.size(0)).fill_(0))
        cand_lengths = Variable(
                    self.cand_lengths.resize_(cview.size(0)).fill_(0))

        for i in range(cview.size(1)):
            if self.attention:
                if self.opt['personachat_attnsentlevel']:
                    h = self.attn_h2attn(cands_hn)
                else:
                    h = cands_hn
                output, _ , _ = self._apply_attention(cands_xes, cands_encoder_output_persona, h, cands_attn_mask)
            else:
                output = cands_xes

            if type(self.decoder) == nn.LSTM:
                output, (cands_hn, cands_cn) = self.decoder(output, (cands_hn, cands_cn))
            else:
                output, cands_hn = self.decoder(output, cands_hn)
            preds, scores = self.hidden_to_idx(output, is_training=False)
            cs = cview.select(1, i)
            non_nulls = cs.ne(self.NULL_IDX)
            cand_lengths += non_nulls.long()
            score_per_cand = torch.gather(scores, 1, ((cs-1)*cs.ne(0).long()).unsqueeze(1))
            cand_scores += score_per_cand.squeeze() * non_nulls.float()
            if self.embshareonly_pm_dec:
                cands_xes = self.lt_enc(cs).unsqueeze(0)
            else:
                cands_xes = self.lt(cs).unsqueeze(0)

        # set empty scores to -1, so when divided by 0 they become -inf
        cand_scores -= cand_lengths.eq(0).float()
        # average the scores per token
        cand_scores /= cand_lengths.float()

        cand_scores = cand_scores.view(cands.size(0), cands.size(1))
        srtd_scores, text_cand_inds = cand_scores.sort(1, True)
        text_cand_inds = text_cand_inds.data

        return text_cand_inds

    def predict(self, xs, xs_persona, ys=None, cands=None, zs=None):
        """Produce a prediction from our model.
        Update the model using the targets if available, otherwise rank
        candidates as well if they are available.
        """
        batchsize = len(xs)
        text_cand_inds = None
        is_training = ys is not None
        self.encoder.train(mode=is_training)
        self.encoder_persona.train(mode=is_training)
        self.decoder.train(mode=is_training)

        encoder_output_persona, hidden_persona, guide_indices = self._encode_persona(xs_persona, ys, is_training)
        encoder_output, hidden = self._encode(xs, is_training)


        # next we use END as an input to kick off our decoder
        x = Variable(self.START_TENSOR)
        xe = self.lt(x).unsqueeze(1)
        xes = xe.expand(xe.size(0), batchsize, xe.size(2))

        # list of output tokens for each example in the batch
        output_lines = None

        if self.opt['personachat_attnsentlevel']:
            attn_mask = Variable(xs_persona.data.sum(2).ne(0), requires_grad=False)
        else:
            attn_mask = Variable(xs_persona.data.ne(0), requires_grad=False)

        if is_training:
            output_lines = self._decode_and_train(batchsize, xes, ys,
                                                  encoder_output_persona, hidden_persona, hidden, attn_mask, guide_indices)

        else:
            if cands is not None:
                text_cand_inds = self._score_candidates(cands, xe,
                                                        encoder_output_persona, hidden_persona, hidden, attn_mask)

            output_lines = self._decode_only(batchsize, xes, ys,
                                             encoder_output_persona, hidden_persona, hidden, attn_mask, zs)
            if zs is not None:
                self._decode_perp(batchsize, xes, ys,
                                  encoder_output_persona, hidden_persona, hidden, attn_mask, zs)

        return output_lines, text_cand_inds


    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        def valid(obs):
            # check if this is an example our model should actually process
            return 'text' in obs and len(obs['text']) > 0
        # valid examples and their indices
        try:
            valid_inds, exs = zip(*[(i, ex) for i, ex in
                                    enumerate(observations) if valid(ex)])
        except ValueError:
            # zero examples to process in this batch, so zip failed to unpack
            return None, None, None, None, None, None, None, None, None

        # set up the input tensors
        batchsize = len(exs)

        if self.opt['personachat_attnsentlevel']:
           parsed = []
           for ex in exs:
               per_ = [p.strip() for p in ex['persona'].split('.')]
               per = [self.parse(p) for p in per_ if p != '']
               parsed.append(per)
        else:
            parsed = [self.parse(ex['persona']) for ex in exs]

        x_lens = [len(x) for x in parsed]
        ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])
        exs = [exs[k] for k in ind_sorted]
        valid_inds = [valid_inds[k] for k in ind_sorted]
        parsed_persona = [parsed[k] for k in ind_sorted]
        self.parsed = parsed_persona

        if self.opt['personachat_attnsentlevel']:
            max_n_per = max([len(per) for per in parsed_persona])
            max_len_per = max([len(p) for per in parsed_persona for p in per])
            xs_persona = torch.LongTensor(batchsize, max_n_per, max_len_per).fill_(self.NULL_IDX)
            for i, per in enumerate(parsed_persona):
                for j, p in enumerate(per):
                    xs_persona[i][j][:len(p)] = torch.LongTensor(p)
            if self.use_cuda:
                self.xs_persona.resize_(xs_persona.size())
                self.xs_persona.copy_(xs_persona, )
                xs_persona = Variable(self.xs_persona)
            else:
                xs_persona = Variable(xs_persona)
        else:
            max_x_len = max([len(x) for x in parsed_persona])
            xs_persona = torch.LongTensor(batchsize, max_x_len).fill_(self.NULL_IDX)
            # right-padded with zeros
            for i, x in enumerate(parsed_persona):
                for j, idx in enumerate(x):
                    xs_persona[i][j] = idx
            if self.use_cuda:
                # copy to gpu
                self.xs_persona.resize_(xs_persona.size())
                self.xs_persona.copy_(xs_persona, )
                xs_persona = Variable(self.xs_persona)
            else:
                xs_persona = Variable(xs_persona)


        parsed_x = [self.parse(ex['text']) for ex in exs]
        max_x_len = max([len(x) for x in parsed_x])
        xs = torch.LongTensor(batchsize, max_x_len).fill_(self.NULL_IDX)
        # right-padded with zeros
        for i, x in enumerate(parsed_x):
            offset = max_x_len - len(x)
            for j, idx in enumerate(x):
                xs[i][j + offset] = idx
        if self.use_cuda:
            # copy to gpu
            self.xs.resize_(xs.size())
            self.xs.copy_(xs, )
            xs = Variable(self.xs)
        else:
            xs = Variable(xs)

        # set up the target tensors
        ys = None
        labels = None
        if any(['labels' in ex for ex in exs]):
            # randomly select one of the labels to update on, if multiple
            # append END to each label
            labels = [random.choice(ex.get('labels', [''])) for ex in exs]
            parsed = [self.parse(y + ' ' + self.END) for y in labels if y]
            max_y_len = max(len(y) for y in parsed)
            if self.truncate > 0 and max_y_len > self.truncate:
                parsed = [y[:self.truncate] for y in parsed]
                max_y_len = self.truncate
            ys = torch.LongTensor(batchsize, max_y_len).fill_(self.NULL_IDX)
            for i, y in enumerate(parsed):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            if self.use_cuda:
                # copy to gpu
                self.ys.resize_(ys.size())
                self.ys.copy_(ys, )
                ys = Variable(self.ys)
            else:
                ys = Variable(ys)


        # set up the target tensors for validation and test
        zs = None
        eval_labels = None

        if any(['eval_labels' in ex for ex in exs]):
            if not (len(exs) == 1 and 'eval_labels' in exs[0] and exs[0]['eval_labels']==['']):
                # randomly select one of the labels to update on, if multiple
                # append END to each label
                eval_labels = [random.choice(ex.get('eval_labels', [''])) for ex in exs]
                parsed = [self.parse(y + ' ' + self.END) for y in eval_labels if y]
                max_y_len = max(len(y) for y in parsed)
                if self.truncate > 0 and max_y_len > self.truncate:
                    parsed = [y[:self.truncate] for y in parsed]
                    max_y_len = self.truncate
                zs = torch.LongTensor(batchsize, max_y_len).fill_(self.NULL_IDX)
                for i, y in enumerate(parsed):
                    for j, idx in enumerate(y):
                        zs[i][j] = idx
                if self.use_cuda:
                    # copy to gpu
                    self.zs.resize_(zs.size())
                    self.zs.copy_(zs, non_blocking=True)
                    zs = Variable(self.zs)
                else:
                    zs = Variable(zs)

        # set up candidates
        cands = None
        valid_cands = None
        if ys is None and self.rank:
            # only do ranking when no targets available and ranking flag set
            parsed = []
            valid_cands = []
            for i, v in enumerate(valid_inds):
                if 'label_candidates' in observations[v]:
                    # each candidate tuple is a pair of the parsed version and
                    # the original full string
                    cs = list(observations[v]['label_candidates'])
                    parsed.append([self.parse(c) for c in cs])
                    valid_cands.append((i, v, cs))
            if len(parsed) > 0:
                # TODO: store lengths of cands separately, so don't have zero
                #       padding for varying number of cands per example
                # found cands, pack them into tensor
                max_c_len = max(max(len(c) for c in cs) for cs in parsed)
                max_c_cnt = max(len(cs) for cs in parsed)
                cands = torch.LongTensor(len(parsed), max_c_cnt, max_c_len).fill_(self.NULL_IDX)
                for i, cs in enumerate(parsed):
                    for j, c in enumerate(cs):
                        for k, idx in enumerate(c):
                            cands[i][j][k] = idx
                if self.use_cuda:
                    # copy to gpu
                    self.cands.resize_(cands.size())
                    self.cands.copy_(cands, )
                    cands = Variable(self.cands)
                else:
                    cands = Variable(cands)

        return xs, xs_persona, ys, labels, valid_inds, cands, valid_cands, zs, eval_labels


    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field

        if self.opt['datatype'] in ['valid', 'test'] and self.opt['personachat_interact']:
            print('OBSVS:' + observations[0]['text'])
            print('PERS: ' + observations[0]['persona'])
            var = input('enter your message: ')
            observations[0]['text'] = var

        xs, xs_persona, ys, labels, valid_inds, cands, valid_cands, zs, eval_labels = self.batchify(observations)

        if xs is None:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available
        predictions, text_cand_inds = self.predict(xs, xs_persona, ys, cands, zs)

        if self.opt['datatype'] in ['valid', 'test'] and self.opt['personachat_interact']:
            print('MODEL:' + ' '.join(predictions[0]))
            f1_best = 0
            msg_best = ''
            for msg in self.teacher.data_dialogs['train']['messages']:
                f1_tmp = _f1_score(' '.join(predictions[0]), [msg[1][1]])
                msg_best = msg[1][1] if f1_tmp > f1_best else msg_best
                f1_best = f1_tmp if f1_tmp > f1_best else f1_best
            print('BEST: {}'.format(msg_best))
            print('TRUE :' + observations[0]['eval_labels'][0])

        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            curr = batch_reply[valid_inds[i]]
            curr['text'] = ' '.join(c for c in (predictions[i][:predictions[i].index(self.END)] if self.END in predictions[i] else predictions[i]))
            curr_pred = curr['text']
            if labels is not None:
                self.answers[valid_inds[i]] = labels[i]
            else:
                self.answers[valid_inds[i]] = curr_pred

        if text_cand_inds is not None:
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
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'lt'):
            model = {}
            model['lt'] = self.lt.state_dict()
            model['lt_per'] = self.lt_per.state_dict()
            model['lt_reweight'] = self.lt_reweight.state_dict()
            if self.embshareonly_pm_dec:
                model['lt_enc'] = self.lt_enc.state_dict()
            model['encoder'] = self.encoder.state_dict()
            model['encoder_persona'] = self.encoder_persona.state_dict()
            model['decoder'] = self.decoder.state_dict()
            if self.sharelt:
                model['h2e'] = self.h2e.state_dict()
                model['e2o'] = self.e2o.state_dict()
            else:
                model['h2o'] = self.h2o.state_dict()
            model['persona_gate'] = self.persona_gate
            model['persona_gate_m'] = self.persona_gate_m
            for attn_name in ['attn', 'attn_v', 'attn_combine', 'attn_h2attn']:
                if hasattr(self, attn_name):
                    model[attn_name] = getattr(self, attn_name).state_dict()
            model['optims'] = {k: v.state_dict()
                               for k, v in self.optims.items()}
            model['longest_label'] = self.longest_label
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
            model = torch.load(read, map_location=lambda cpu, _: cpu)

        return model['opt'], model

    def set_states(self, states):
        """Set the state dicts of the modules from saved states."""
        self.lt.load_state_dict(states['lt'])
        self.lt_per.load_state_dict(states['lt_per'])
        self.lt_reweight.load_state_dict(states['lt_reweight'])
        if self.embshareonly_pm_dec:
            self.lt_enc.load_state_dict(states['lt_enc'])
        self.encoder.load_state_dict(states['encoder'])
        self.encoder_persona.load_state_dict(states['encoder_persona'])
        self.decoder.load_state_dict(states['decoder'])
        if self.sharelt:
            self.h2e.load_state_dict(states['h2e'])
            self.e2o.load_state_dict(states['e2o'])
        else:
            self.h2o.load_state_dict(states['h2o'])
        self.persona_gate = states['persona_gate']
        self.persona_gate_m = states['persona_gate_m']
        for attn_name in ['attn', 'attn_v', 'attn_combine', 'attn_h2attn']:
            if hasattr(self, attn_name):
                getattr(self, attn_name).load_state_dict(states[attn_name])
        for k, v in states['optims'].items():
            if k in ['attn', 'attn_v', 'attn_combine', 'attn_h2attn']:
                if hasattr(self, attn_name):
                    getattr(self, attn_name).load_state_dict(states[attn_name])
            else:
                self.optims[k].load_state_dict(v)
        self.longest_label = states['longest_label']

    def report(self):
        m = {}
        if self.metrics['num_tokens'] > 0:
            m['loss'] = self.metrics['loss'] / self.metrics['num_tokens']
            m['ppl'] = math.exp(m['loss'])
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

    def report_loss(self):
        if hasattr(self, 'loss_guide'):
            print("The loss is {}, model loss is: {}, guidesoftmax loss is: {}".format(self.loss/(self.loss_c+1e-10),
                (self.loss - self.loss_guide)/(self.loss_c+1e-10), (self.loss_guide)/(self.loss_c+1e-10)))
        else:
            print("The loss is {}".format(self.loss/(self.loss_c+1e-10)))
        loss_record = self.loss/(self.loss_c+1e-10)
        self.loss = 0.0
        if hasattr(self, 'loss_guide'):
            self.loss_guide = 0.
        self.loss_c = 0
        return loss_record
