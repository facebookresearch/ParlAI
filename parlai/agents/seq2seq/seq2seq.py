# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import torch
import os
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

    ENC_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.005,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0,
                           help='dropout rate')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'local'],
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
                           'input and output to have a maximum length and to '
                           'be similar in length to one another by throwing '
                           'away extra tokens. This reduces the total amount '
                           'of padding in the batches.')
        agent.add_argument('-enc', '--encoder', default='gru',
                           choices=Seq2seqAgent.ENC_OPTS.keys(),
                           help='Choose between different encoder modules.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'] + list(Seq2seqAgent.ENC_OPTS.keys()),
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
        agent.add_argument('-emb', '--embedding-init', default='random',
                           choices=['random', 'glove'],
                           help='Choose between initialization strategies '
                                'for word embeddings. Default is random, '
                                'but can also preinitialize from Glove')
        agent.add_argument('-lm', '--language-model', type='bool',
                           default=False,
                           help='enabled language modeling training on the '
                                'concatenated input and label data')

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        if shared:
            # set up shared properties
            # answers contains a batch_size list of the last answer produced
            self.answers = shared['answers']
            # start token
            self.START = shared['START']
            # end token
            self.END = shared['END']
        else:
            # this is not a shared instance of this class, so do full init

            # answers contains a batch_size list of the last answer produced
            self.answers = [None] * opt['batchsize']

            # check for cuda
            self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            states = None
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, states = self.load(opt['model_file'])
                # override model-specific options with stored ones
                opt = self.override_opt(new_opt)

            if opt['dict_file'] is None and opt.get('model_file'):
                # set default dict-file if not set
                opt['dict_file'] = opt['model_file'] + '.dict'

            # load dictionary and basic tokens & vectors
            self.dict = DictionaryAgent(opt)
            self.id = 'Seq2Seq'
            # we use START markers to start our output
            self.START = self.dict.start_token
            self.START_TENSOR = torch.LongTensor(self.dict.parse(self.START))
            # we use END markers to end our output
            self.END = self.dict.end_token
            self.END_TENSOR = torch.LongTensor(self.dict.parse(self.END))
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict.txt2vec(self.dict.null_token)[0]

            # store important params in self
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
            self.bidirectional = opt['bidirectional']
            self.num_dirs = 2 if self.bidirectional else 1
            self.dropout = opt['dropout']
            self.lm = opt['language_model']

            # set up tensors once
            self.zeros = torch.zeros(self.num_layers * self.num_dirs, 1, hsz)
            self.xs = torch.LongTensor(1, 1)
            self.ys = torch.LongTensor(1, 1)
            if self.rank:
                self.cands = torch.LongTensor(1, 1, 1)
                self.cand_scores = torch.FloatTensor(1)
                self.cand_lengths = torch.LongTensor(1)

            # set up modules
            self.criterion = nn.NLLLoss()
            # lookup table stores word embeddings
            self.enc_lt = nn.Embedding(len(self.dict), emb,
                                       padding_idx=self.NULL_IDX,
                                       max_norm=10)

            if opt['lookuptable'] in ['enc_dec', 'all']:
                # share this with the encoder
                self.dec_lt = self.enc_lt
            else:
                self.dec_lt = nn.Embedding(len(self.dict), emb,
                                           padding_idx=self.NULL_IDX,
                                           max_norm=10)

            if not states and opt['embedding_init'] == 'glove':
                # set up pre-initialized vectors from GloVe
                try:
                    import torchtext.vocab as vocab
                except ImportError:
                    raise ImportError('Please install torchtext from'
                                      'github.com/pytorch/text.')
                Glove = vocab.GloVe(name='840B', dim=300)
                # do better than uniform random
                proj = torch.FloatTensor(emb, 300).uniform_(-0.057735, 0.057735) if emb != 300 else None
                for w in self.dict.freq:
                    if w in Glove.stoi:
                        vec = Glove.vectors[Glove.stoi[w]]
                        if emb != 300:
                            vec = torch.mm(proj, vec.unsqueeze(1)).squeeze()
                        self.enc_lt.weight.data[self.dict[w]] = vec
                        self.dec_lt.weight.data[self.dict[w]] = vec

            # encoder captures the input text
            enc_class = Seq2seqAgent.ENC_OPTS[opt['encoder']]
            # decoder produces our output states
            if opt['decoder'] in ['same', 'shared']:
                # use same class as encoder
                self.decoder = enc_class(emb, hsz, opt['numlayers'],
                                         dropout=self.dropout,
                                         batch_first=True)
            else:
                # use set class
                dec_class = Seq2seqAgent.ENC_OPTS[opt['decoder']]
                self.decoder = dec_class(emb, hsz, opt['numlayers'],
                                         dropout=self.dropout,
                                         batch_first=True)
            if opt['decoder'] == 'shared':
                # shared weights: use the decoder to encode
                if self.bidirectional:
                    raise RuntimeError('Cannot share enc/dec and do '
                                       'bidirectional encoding.')
                self.encoder = self.decoder
            else:
                self.encoder = enc_class(emb, hsz, opt['numlayers'],
                                         dropout=self.dropout, batch_first=True,
                                         bidirectional=self.bidirectional)

            # linear layers help us produce outputs from final decoder state
            hszXdirs = hsz * self.num_dirs
            self.h2e = nn.Linear(hsz, emb)  # hidden to embedding
            self.e2o = nn.Linear(emb, len(self.dict))  # embedding to output
            if opt['lookuptable'] in ['dec_out', 'all']:
                # share these weights with the decoder lookup table
                self.e2o.weight = self.dec_lt.weight

            if self.attention == 'local':
                # local attention over fixed set of output states
                if opt['attention_length'] < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = opt['attention_length']
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz + emb, self.max_length)
                # combines attention weights with encoder outputs
                self.attn_combine = nn.Linear(hszXdirs + emb, emb)
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz)
                self.attn_v = nn.Linear(hsz, 1)
                self.attn_combine = nn.Linear(hszXdirs + emb, emb)
            elif self.attention == 'general':
                self.attn = nn.Linear(hsz, hszXdirs)
                self.attn_combine = nn.Linear(hszXdirs + emb, emb)

            # set up optims for each module
            lr = opt['learningrate']
            optim_class = Seq2seqAgent.OPTIM_OPTS[opt['optimizer']]
            kwargs = {'lr': lr}
            if opt['optimizer'] == 'sgd':
                kwargs['momentum'] = 0.95
                kwargs['nesterov'] = True
            self.optims = {
                'enc_lt': optim_class(self.enc_lt.parameters(), **kwargs),
                'decoder': optim_class(self.decoder.parameters(), **kwargs),
                'h2e': optim_class(self.h2e.parameters(), **kwargs),
                'e2o': optim_class(self.e2o.parameters(), **kwargs),
            }
            if opt['decoder'] != 'shared':
                self.optims['encoder'] = optim_class(
                    self.encoder.parameters(), **kwargs)
            if opt['lookuptable'] not in ['enc_dec', 'all']:
                # only add dec if it's separate from enc
                self.optims['dec_lt'] = optim_class(
                    self.dec_lt.parameters(), **kwargs)

            # add attention parameters into optims if available
            for attn_name in ['attn', 'attn_v', 'attn_combine']:
                if hasattr(self, attn_name):
                    self.optims[attn_name] = optim_class(
                        getattr(self, attn_name).parameters(), **kwargs)

            if states is not None:
                # set loaded states if applicable
                self.set_states(states)

            if self.use_cuda:
                self.cuda()

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
        return self.dict.vec2txt(vec)

    def cuda(self):
        """Push parameters to the GPU."""
        self.START_TENSOR = self.START_TENSOR.cuda(async=True)
        self.END_TENSOR = self.END_TENSOR.cuda(async=True)
        self.zeros = self.zeros.cuda(async=True)
        self.xs = self.xs.cuda(async=True)
        self.ys = self.ys.cuda(async=True)
        if self.rank:
            self.cands = self.cands.cuda(async=True)
            self.cand_scores = self.cand_scores.cuda(async=True)
            self.cand_lengths = self.cand_lengths.cuda(async=True)
        self.criterion.cuda()
        self.enc_lt.cuda()
        self.dec_lt.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.h2e.cuda()
        self.e2o.cuda()
        if self.attention != 'none':
            for attn_name in ['attn', 'attn_v', 'attn_combine']:
                if hasattr(self, attn_name):
                    getattr(self, attn_name).cuda()

    def hidden_to_idx(self, hidden, is_training=False):
        """Convert hidden state vectors into indices into the dictionary."""
        # dropout at each step
        e = F.dropout(self.h2e(hidden), p=self.dropout, training=is_training)
        out = F.dropout(self.e2o(e), p=self.dropout, training=is_training)

        # out is batch_size x sequence_length x dict_sz
        if out.size(1) == 1:
            # sequence length is one, just squeeze it so we don't need to cat
            scores = F.log_softmax(out.squeeze(1)).unsqueeze(1)
        else:
            # we need a softmax per token
            # index on argmin(batch_size,seq_length) so fewer cats / bigger ops
            dim = 0 if out.size(0) < out.size(1) else 1
            scores = torch.cat([F.log_softmax(out.select(dim, i)).unsqueeze(dim)
                                for i in range(out.size(dim))], dim)
        _max_score, idx = scores.max(2)
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

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['answers'] = self.answers
        shared['START'] = self.START
        shared['END'] = self.END
        return shared

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()
        if 'text' in observation:
            # put START and END around text
            observation['text'] = '{s} {x} {e}'.format(
                s=self.START, x=observation['text'], e=self.END)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            # get last y
            batch_idx = self.opt.get('batchindex', 0)
            if self.answers[batch_idx] is not None:
                # use our last answer, which is the label during training
                lastY = self.answers[batch_idx]
                prev_dialogue = '{p}\n{s} {y} {e}'.format(
                    p=prev_dialogue, s=self.START, y=lastY, e=self.END)
                self.answers[batch_idx] = None  # forget last y
            # add current observation back in
            observation['text'] = '{p}\n{x}'.format(
                p=prev_dialogue, x=observation['text'])
            # final text: <s> lastx </s> \n <s> lasty </s> \n <s> currx </s>
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def _encode(self, xs, is_training=False):
        """Call encoder and return output and hidden states."""
        self.lastxs = xs
        batchsize = len(xs)

        # first encode context
        xes = F.dropout(self.enc_lt(xs), p=self.dropout, training=is_training)
        # project from emb_size to hidden_size dimensions
        x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
        xes_packed = pack_padded_sequence(xes, x_lens, batch_first=True)

        if self.zeros.size(1) != batchsize:
            self.zeros.resize_(self.num_layers * self.num_dirs,
                               batchsize, self.hidden_size).fill_(0)

        h0 = Variable(self.zeros, requires_grad=False)
        if type(self.encoder) == nn.LSTM:
            encoder_output_packed, hidden = self.encoder(xes_packed, (h0, h0))
            # take elementwise max between forward and backward hidden states
            hidden = (hidden[0].view(-1, self.num_dirs, hidden[0].size(1), hidden[0].size(2)).max(1)[0],
                      hidden[1].view(-1, self.num_dirs, hidden[1].size(1), hidden[1].size(2)).max(1)[0])
            if type(self.decoder) != nn.LSTM:
                hidden = hidden[0]
        else:
            encoder_output_packed, hidden = self.encoder(xes_packed, h0)

            # take elementwise max between forward and backward hidden states
            hidden = hidden.view(-1, self.num_dirs, hidden.size(1), hidden.size(2)).max(1)[0]
            if type(self.decoder) == nn.LSTM:
                hidden = (hidden, h0.narrow(0, 0, 2))
        encoder_output, _ = pad_packed_sequence(encoder_output_packed,
                                                batch_first=True)
        encoder_output = encoder_output

        if self.attention == 'local':
            # if using local attention, narrow encoder_output to max_length
            if encoder_output.size(1) > self.max_length:
                offset = encoder_output.size(1) - self.max_length
                encoder_output = encoder_output.narrow(
                    1, offset, self.max_length)

        return encoder_output, hidden

    def _apply_attention(self, xes, encoder_output, hidden, attn_mask=None):
        """Apply attention to encoder hidden layer."""
        last_hidden = hidden[-1]  # select hidden from last RNN layer
        if self.attention == 'concat':
            hidden_expand = last_hidden.unsqueeze(1).expand(
                last_hidden.size(0), encoder_output.size(1), last_hidden.size(1))
            attn_w_premask = self.attn_v(F.tanh(self.attn(
                torch.cat((encoder_output, hidden_expand), 2)))).squeeze(2)
            attn_weights = F.softmax(attn_w_premask * attn_mask -
                                     (1 - attn_mask) * 1e20)

        elif self.attention == 'general':
            hidden_expand = last_hidden.unsqueeze(1)
            attn_w_premask = torch.bmm(self.attn(hidden_expand),
                                       encoder_output.transpose(1, 2)
                                       ).squeeze(1)
            attn_weights = F.softmax(attn_w_premask * attn_mask -
                                     (1 - attn_mask) * 1e20)

        elif self.attention == 'local':
            attn_weights = F.softmax(self.attn(
                torch.cat((xes.squeeze(1), last_hidden), 1)))
            if attn_weights.size(1) > encoder_output.size(1):
                attn_weights = attn_weights.narrow(
                    1, 0, encoder_output.size(1))

        attn_applied = torch.bmm(
            attn_weights.unsqueeze(1), encoder_output).squeeze(1)

        output = torch.cat((xes.squeeze(1), attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(1)
        output = F.tanh(output)

        return output

    def _decode_and_train(self, batchsize, xes, ys, encoder_output, hidden, attn_mask, lm=False):
        """Update the model based on the labels."""
        self.zero_grad()
        loss = 0

        output_lines = [[] for _ in range(batchsize)]

        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that during prediction
        if not lm:
            self.longest_label = max(self.longest_label, ys.size(1))
        if self.attention != 'none':
            # using attention, produce one token at a time
            for i in range(ys.size(1)):
                h_att = hidden[0] if type(self.decoder) == nn.LSTM else hidden
                output = self._apply_attention(xes, encoder_output, h_att, attn_mask)
                output, hidden = self.decoder(output, hidden)
                preds, scores = self.hidden_to_idx(output, is_training=True)
                y = ys.select(1, i)
                loss += self.criterion(scores.squeeze(1), y)
                # use the true token as the next input instead of predicted
                xes = self.dec_lt(y).unsqueeze(1)
                xes = F.dropout(xes, p=self.dropout, training=True)
                for b in range(batchsize):
                    # convert the output scores to tokens
                    token = self.v2t(preds.data[b])
                    output_lines[b].append(token)
        else:
            # force the entire sequence at once by feeding in START + y[:-2]
            y_in = ys.narrow(1, 0, ys.size(1) - 1)
            xes = torch.cat([xes, self.dec_lt(y_in)], 1)

            output, hidden = self.decoder(xes, hidden)
            preds, scores = self.hidden_to_idx(output, is_training=True)
            for i in range(ys.size(1)):
                # sum loss per-token
                score = scores.select(1, i)
                y = ys.select(1, i)
                loss += self.criterion(score, y)
            for b in range(batchsize):
                output_lines[b].extend(self.v2t(preds.data[b]).split(' '))
        loss.backward()
        self.update_params()

        if random.random() < 0.1:
            # sometimes output a prediction for debugging
            # print('prediction:', ' '.join(output_lines[0]))
            # print('label:', self.v2t(ys.data[0]))
            print('lm' if lm else '  ', 'loss:', loss.data[0])

        return output_lines

    def _decode_only(self, batchsize, xes, ys, encoder_output, hidden, attn_mask):
        """Just produce a prediction without training the model."""
        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0

        output_lines = [[] for _ in range(batchsize)]

        # generate a response from scratch
        while(total_done < batchsize) and max_len < self.longest_label:
            # keep producing tokens until we hit END or max length for each
            # example in the batch
            if self.attention == 'none':
                output = xes
            else:
                h_att = hidden[0] if type(self.decoder) == nn.LSTM else hidden
                output = self._apply_attention(xes, encoder_output, h_att, attn_mask)
            output, hidden = self.decoder(output, hidden)
            preds, _scores = self.hidden_to_idx(output, is_training=False)

            xes = self.dec_lt(preds)
            max_len += 1
            for b in range(batchsize):
                if not done[b]:
                    # only add more tokens for examples that aren't done yet
                    token = self.v2t(preds.data[b])
                    if token == self.END:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
                    else:
                        output_lines[b].append(token)

        if random.random() < 0.2:
            # sometimes output a prediction for debugging
            print('\nprediction:', ' '.join(output_lines[0]))

        return output_lines

    def _score_candidates(self, cands, cand_inds, start, encoder_output, hidden, attn_mask):
        """Rank candidates by their likelihood according to the decoder."""
        if type(self.decoder) == nn.LSTM:
            hidden, cell = hidden
        # score each candidate separately
        # cands are exs_with_cands x cands_per_ex x words_per_cand
        # cview is total_cands x words_per_cand
        cview = cands.view(-1, cands.size(2))
        c_xes = start.expand(cview.size(0), start.size(0), start.size(1))

        if len(cand_inds) != hidden.size(1):
            # only use hidden state from inputs with associated candidates
            cand_indices = torch.LongTensor([i for i, _, _ in cand_inds])
            if self.use_cuda:
                cand_indices = cand_indices.cuda()
            cand_indices = Variable(cand_indices)
            hidden = hidden.index_select(1, cand_indices)

        sz = hidden.size()
        cands_hn = (
            hidden.view(sz[0], sz[1], 1, sz[2])
            .expand(sz[0], sz[1], cands.size(1), sz[2])
            .contiguous()
            .view(sz[0], -1, sz[2])
        )
        if type(self.decoder) == nn.LSTM:
            if len(cand_inds) != cell.size(1):
                # only use cell state from inputs with associated candidates
                cell = cell.index_select(1, cand_indices)
            cands_hn = (cands_hn, cell.view(sz[0], sz[1], 1, sz[2])
                                      .expand(sz[0], sz[1], cands.size(1), sz[2])
                                      .contiguous()
                                      .view(sz[0], -1, sz[2]))

        cand_scores = Variable(
            self.cand_scores.resize_(cview.size(0)).fill_(0))
        cand_lengths = Variable(
            self.cand_lengths.resize_(cview.size(0)).fill_(0))

        if self.attention != 'none':
            # using attention
            sz = encoder_output.size()
            cands_encoder_output = (
                encoder_output.contiguous()
                .view(sz[0], 1, sz[1], sz[2])
                .expand(sz[0], cands.size(1), sz[1], sz[2])
                .contiguous()
                .view(-1, sz[1], sz[2])
            )

            msz = attn_mask.size()
            cands_attn_mask = (
                attn_mask.contiguous()
                .view(msz[0], 1, msz[1])
                .expand(msz[0], cands.size(1), msz[1])
                .contiguous()
                .view(-1, msz[1])
            )
            for i in range(cview.size(1)):
                # process one token at a time
                h_att = cands_hn[0] if type(self.decoder) == nn.LSTM else cands_hn
                output = self._apply_attention(c_xes, cands_encoder_output, h_att, cands_attn_mask)
                output, cands_hn = self.decoder(output, cands_hn)
                _preds, scores = self.hidden_to_idx(output, is_training=False)
                cs = cview.select(1, i)
                non_nulls = cs.ne(self.NULL_IDX)
                cand_lengths += non_nulls.long()
                score_per_cand = torch.gather(scores.select(1, i), 1, cs.unsqueeze(1))
                cand_scores += score_per_cand.squeeze() * non_nulls.float()
                c_xes = self.dec_lt(cs).unsqueeze(1)
        else:
            # process entire sequence at once
            if cview.size(1) > 1:
                # feed in START + cands[:-2]
                cands_in = cview.narrow(1, 0, cview.size(1) - 1)
                c_xes = torch.cat([c_xes, self.dec_lt(cands_in)], 1)
            output, cands_hn = self.decoder(c_xes, cands_hn)
            _preds, scores = self.hidden_to_idx(output, is_training=False)

            for i in range(cview.size(1)):
                # calculate score at each token
                cs = cview.select(1, i)
                non_nulls = cs.ne(self.NULL_IDX)
                cand_lengths += non_nulls.long()
                score_per_cand = torch.gather(scores.select(1, i), 1, cs.unsqueeze(1))
                cand_scores += score_per_cand.squeeze() * non_nulls.float()

        # set empty scores to -1, so when divided by 0 they become -inf
        cand_scores -= cand_lengths.eq(0).float()
        # average the scores per token
        cand_scores /= cand_lengths.float()

        cand_scores = cand_scores.view(cands.size(0), cands.size(1))
        srtd_scores, text_cand_inds = cand_scores.sort(1, True)
        text_cand_inds = text_cand_inds.data

        return text_cand_inds

    def predict(self, xs, ys=None, cands=None, valid_cands=None, lm=False):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available and param is set.
        """
        batchsize = len(xs)
        text_cand_inds = None
        is_training = ys is not None
        self.encoder.train(mode=is_training)
        self.decoder.train(mode=is_training)
        encoder_output, hidden = self._encode(xs, is_training)

        # next we use START as an input to kick off our decoder
        if not lm:
            x = Variable(self.START_TENSOR, requires_grad=False)
            xe = self.dec_lt(x)
            xe = F.dropout(xe, p=self.dropout, training=is_training)
            xes = xe.expand(batchsize, 1, xe.size(1))
        else:
            # during language_model mode, just start with zeros
            xes = Variable(
                self.zeros[0].narrow(1, 0, self.emb_size).unsqueeze(1),
                requires_grad=False
            )

        # list of output tokens for each example in the batch
        output_lines = None

        if self.attention == 'none':
            attn_mask = None
        else:
            attn_mask = xs.ne(0).float()

        if is_training:
            output_lines = self._decode_and_train(batchsize, xes, ys,
                                                  encoder_output, hidden,
                                                  attn_mask, lm=lm)
        else:
            if cands is not None:
                text_cand_inds = self._score_candidates(cands, valid_cands, xe,
                                                        encoder_output, hidden,
                                                        attn_mask)

            output_lines = self._decode_only(batchsize, xes, ys,
                                             encoder_output, hidden,
                                             attn_mask)

        return output_lines, text_cand_inds

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        def valid(obs):
            # check if this is an example our model should actually process
            return 'text' in obs and ('labels' in obs or 'eval_labels' in obs)
        # valid examples and their indices
        valid_inds, exs = zip(*[(i, ex) for i, ex in enumerate(observations) if valid(ex)])

        # set up the input tensors
        batchsize = len(exs)
        if batchsize == 0:
            return None, None, None, None, None, None

        # tokenize the text
        parsed = [self.parse(ex['text']) for ex in exs]
        x_lens = [len(x) for x in parsed]
        ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

        exs = [exs[k] for k in ind_sorted]
        valid_inds = [valid_inds[k] for k in ind_sorted]
        parsed = [parsed[k] for k in ind_sorted]

        max_x_len = max([len(x) for x in parsed])
        if self.truncate > 0:
            # shrink xs to to limit batch computation
            max_x_len = min(max_x_len, self.truncate)
            parsed = [x[-max_x_len:] for x in parsed]
        xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
        # right-padded with zeros
        for i, x in enumerate(parsed):
            for j, idx in enumerate(x):
                xs[i][j] = idx
        if self.use_cuda:
            # copy to gpu
            self.xs.resize_(xs.size())
            self.xs.copy_(xs, async=True)
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
            if self.truncate > 0:
                # shrink ys to to limit batch computation
                max_y_len = min(max_y_len, self.truncate)
                parsed = [y[:max_y_len] for y in parsed]
            ys = torch.LongTensor(batchsize, max_y_len).fill_(0)
            for i, y in enumerate(parsed):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            if self.use_cuda:
                # copy to gpu
                self.ys.resize_(ys.size())
                self.ys.copy_(ys, async=True)
                ys = Variable(self.ys)
            else:
                ys = Variable(ys)

        # set up candidates
        cands = None
        valid_cands = None
        if ys is None and self.rank:
            # only do ranking when no targets available and ranking flag set
            parsed = []
            valid_cands = []
            for i, v in enumerate(valid_inds):
                if 'label_candidates' in observations[i]:
                    # each candidate tuple is a pair of the parsed version and
                    # the original full string
                    cs = list(observations[i]['label_candidates'])
                    parsed.append([self.parse(c) for c in cs])
                    valid_cands.append((i, v, cs))
            if len(parsed) > 0:
                # TODO: store lengths of cands separately, so don't have zero
                #       padding for varying number of cands per example
                # found cands, pack them into tensor
                max_c_len = max(max(len(c) for c in cs) for cs in parsed)
                max_c_cnt = max(len(cs) for cs in parsed)
                cands = torch.LongTensor(len(parsed), max_c_cnt, max_c_len).fill_(0)
                for i, cs in enumerate(parsed):
                    for j, c in enumerate(cs):
                        for k, idx in enumerate(c):
                            cands[i][j][k] = idx
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
        xs, ys, labels, valid_inds, cands, valid_cands = self.batchify(observations)

        if xs is None:
            # no valid examples, just return empty responses
            return batch_reply

        # produce predictions, train on targets if available
        predictions, text_cand_inds = self.predict(xs, ys, cands, valid_cands)
        if self.lm and ys is not None:
            # also train on lm task: given "START", predict
            new_obs = [
                {
                    'text': self.START,
                    'labels': [
                        '{x} {s} {y}'.format(
                            x=obs['text'].replace(self.START, ''),
                            s=self.START,
                            y=random.choice(obs.get('labels', [''])))
                    ]
                } for obs in observations
            ]
            xs, ys, _, _, _, _ = self.batchify(new_obs)
            _, _ = self.predict(xs, ys, lm=True)

        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            curr = batch_reply[valid_inds[i]]
            curr_pred = ' '.join(c for c in predictions[i] if c != self.END
                                 and c != self.dict.null_token)
            curr['text'] = curr_pred
            if labels is not None:
                self.answers[valid_inds[i]] = labels[i]
            else:
                self.answers[valid_inds[i]] = curr_pred

        if text_cand_inds is not None:
            for i in range(len(valid_cands)):
                order = text_cand_inds[i]
                _, batch_idx, curr_cands = valid_cands[i]
                curr = batch_reply[valid_inds[batch_idx]]
                curr['text_candidates'] = [curr_cands[idx] for idx in order
                                           if idx < len(curr_cands)]

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'optims'):
            model = {}
            model['enc_lt'] = self.enc_lt.state_dict()
            if self.opt['lookuptable'] not in ['enc_dec', 'all']:
                # dec_lt is enc_lt
                raise RuntimeError()
                # model['dec_lt'] = self.dec_lt.state_dict()
            if self.opt['decoder'] != 'shared':
                model['encoder'] = self.encoder.state_dict()
            model['decoder'] = self.decoder.state_dict()
            model['h2e'] = self.h2e.state_dict()
            model['e2o'] = self.e2o.state_dict()
            model['optims'] = {k: v.state_dict()
                               for k, v in self.optims.items()}
            model['longest_label'] = self.longest_label
            model['opt'] = self.opt

            for attn_name in ['attn', 'attn_v', 'attn_combine']:
                if hasattr(self, attn_name):
                    model[attn_name] = getattr(self, attn_name).state_dict()

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
            model = torch.load(read)

        return model['opt'], model

    def set_states(self, states):
        """Set the state dicts of the modules from saved states."""
        self.enc_lt.load_state_dict(states['enc_lt'])
        if self.opt['lookuptable'] not in ['enc_dec', 'all']:
            # dec_lt is enc_lt
            raise RuntimeError('dec_lt state should not exist--it is same as enc_lt.')
        if self.opt['decoder'] != 'shared':
            self.encoder.load_state_dict(states['encoder'])
        self.decoder.load_state_dict(states['decoder'])
        self.h2e.load_state_dict(states['h2e'])
        self.e2o.load_state_dict(states['e2o'])
        for attn_name in ['attn', 'attn_v', 'attn_combine']:
            if attn_name in states:
                getattr(self, attn_name).load_state_dict(states[attn_name])

        for k, v in states['optims'].items():
            self.optims[k].load_state_dict(v)
        self.longest_label = states['longest_label']
