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
import torch.nn.functional as F
import torch
import os
import random


class Seq2seqAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    For more information, see Sequence to Sequence Learning with Neural
    Networks `(Sutskever et al. 2014) <https://arxiv.org/abs/1409.3215>`_.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers and embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.5,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-att', '--attention', type=int, default=0,
                           help='if greater than 0, use attention of specified'
                                ' length while decoding')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-rc', '--rank-candidates', type='bool',
                           default=False,
                           help='rank candidates if available. this is done by'
                                ' computing the mean score per token for each '
                                'candidate and selecting the highest scoring.')
        agent.add_argument('-tr', '--truncate', type='bool', default=True,
                           help='truncate input & output lengths to speed up '
                           'training (may reduce accuracy). This fixes all '
                           'input and output to have a maximum length and to '
                           'be similar in length to one another by throwing '
                           'away extra tokens. This reduces the total amount '
                           'of padding in the batches.')
        agent.add_argument('-enc', '--encoder', default='gru',
                           choices=['rnn', 'gru', 'lstm'],
                           help='Choose between different encoder modules.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared', 'rnn', 'gru', 'lstm'],
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights.')

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        if not shared:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.

            # check for cuda
            self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, self.states = self.load(opt['model_file'])
                # override options with stored ones
                opt = self.override_opt(new_opt)

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

            # store important params directly
            hsz = opt['hiddensize']
            self.hidden_size = hsz
            self.num_layers = opt['numlayers']
            self.learning_rate = opt['learningrate']
            self.rank = opt['rank_candidates']
            self.longest_label = 1
            self.truncate = opt['truncate']
            self.attention = opt['attention']

            # set up tensors
            self.zeros = torch.zeros(self.num_layers, 1, hsz)
            self.xs = torch.LongTensor(1, 1)
            self.ys = torch.LongTensor(1, 1)
            self.cands = torch.LongTensor(1, 1, 1)
            self.cand_scores = torch.FloatTensor(1)
            self.cand_lengths = torch.LongTensor(1)

            # set up modules
            self.criterion = nn.NLLLoss()
            # lookup table stores word embeddings
            self.lt = nn.Embedding(len(self.dict), hsz,
                                   padding_idx=self.NULL_IDX,
                                   scale_grad_by_freq=True)
            opt_to_class = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
            # encoder captures the input text
            enc_class = opt_to_class[opt['encoder']]
            self.encoder = enc_class(hsz, hsz, opt['numlayers'])
            # decoder produces our output states
            if opt['decoder'] == 'shared':
                self.decoder = self.encoder
            elif opt['decoder'] == 'same':
                self.decoder = enc_class(hsz, hsz, opt['numlayers'])
            else:
                dec_class = opt_to_class[opt['decoder']]
                self.decoder = dec_class(hsz, hsz, opt['numlayers'])
            # linear layer helps us produce outputs from final decoder state
            self.h2o = nn.Linear(hsz, len(self.dict))
            # droput on the linear layer helps us generalize
            self.dropout = nn.Dropout(opt['dropout'])

            self.use_attention = False
            # if attention is greater than 0, set up additional members
            if self.attention > 0:
                self.use_attention = True
                self.max_length = self.attention
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz * 2, self.max_length)
                # combines attention weights with encoder outputs
                self.attn_combine = nn.Linear(hsz * 2, hsz)

            # set up optims for each module
            lr = opt['learningrate']
            self.optims = {
                'lt': optim.SGD(self.lt.parameters(), lr=lr),
                'encoder': optim.SGD(self.encoder.parameters(), lr=lr),
                'decoder': optim.SGD(self.decoder.parameters(), lr=lr),
                'h2o': optim.SGD(self.h2o.parameters(), lr=lr),
            }

            if hasattr(self, 'states'):
                # set loaded states if applicable
                self.set_states(self.states)

            if self.use_cuda:
                self.cuda()

        self.reset()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'numlayers'}
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
        self.cands = self.cands.cuda(async=True)
        self.cand_scores = self.cand_scores.cuda(async=True)
        self.cand_lengths = self.cand_lengths.cuda(async=True)
        self.criterion.cuda()
        self.lt.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.h2o.cuda()
        self.dropout.cuda()
        if self.use_attention:
            self.attn.cuda()
            self.attn_combine.cuda()

    def hidden_to_idx(self, hidden, dropout=False):
        """Convert hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)
        hidden = hidden.squeeze(0)
        scores = self.h2o(hidden)
        if dropout:
            scores = self.dropout(scores)
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
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def _encode(self, xs):
        """Call encoder and return output and hidden states."""
        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs).transpose(0, 1)
        if self.zeros.size(1) != batchsize:
            self.zeros.resize_(self.num_layers, batchsize, self.hidden_size).fill_(0)
        h0 = Variable(self.zeros)
        if type(self.encoder) == nn.LSTM:
            encoder_output, hidden = self.encoder(xes, (h0, h0))
            if type(self.decoder) != nn.LSTM:
                hidden = hidden[0]
        else:
            encoder_output, hidden = self.encoder(xes, h0)
            if type(self.decoder) == nn.LSTM:
                hidden = (hidden, h0)
        encoder_output = encoder_output.transpose(0, 1)

        if self.use_attention:
            if encoder_output.size(1) > self.max_length:
                offset = encoder_output.size(1) - self.max_length
                encoder_output = encoder_output.narrow(1, offset, self.max_length)

        return encoder_output, hidden


    def _apply_attention(self, xes, encoder_output, encoder_hidden):
        """Apply attention to encoder hidden layer."""
        attn_weights = F.softmax(self.attn(torch.cat((xes[0], encoder_hidden[-1]), 1)))

        if attn_weights.size(1) > encoder_output.size(1):
            attn_weights = attn_weights.narrow(1, 0, encoder_output.size(1) )

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_output).squeeze(1)

        output = torch.cat((xes[0], attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        return output


    def _decode_and_train(self, batchsize, xes, ys, encoder_output, hidden):
        # update the model based on the labels
        self.zero_grad()
        loss = 0

        output_lines = [[] for _ in range(batchsize)]

        # keep track of longest label we've ever seen
        self.longest_label = max(self.longest_label, ys.size(1))
        for i in range(ys.size(1)):
            output = self._apply_attention(xes, encoder_output, hidden) if self.use_attention else xes

            output, hidden = self.decoder(output, hidden)
            preds, scores = self.hidden_to_idx(output, dropout=True)
            y = ys.select(1, i)
            loss += self.criterion(scores, y)
            # use the true token as the next input instead of predicted
            # this produces a biased prediction but better training
            xes = self.lt(y).unsqueeze(0)
            for b in range(batchsize):
                # convert the output scores to tokens
                token = self.v2t([preds.data[b]])
                output_lines[b].append(token)

        loss.backward()
        self.update_params()

        if random.random() < 0.1:
            # sometimes output a prediction for debugging
            print('prediction:', ' '.join(output_lines[0]),
                  '\nlabel:', self.dict.vec2txt(ys.data[0]))

        return output_lines

    def _decode_only(self, batchsize, xes, ys, encoder_output, hidden):
        # just produce a prediction without training the model
        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0

        output_lines = [[] for _ in range(batchsize)]

        # now, generate a response from scratch
        while(total_done < batchsize) and max_len < self.longest_label:
            # keep producing tokens until we hit END or max length for each
            # example in the batch
            output = self._apply_attention(xes, encoder_output, hidden) if self.use_attention else xes

            output, hidden = self.decoder(output, hidden)
            preds, scores = self.hidden_to_idx(output, dropout=False)

            xes = self.lt(preds.unsqueeze(0))
            max_len += 1
            for b in range(batchsize):
                if not done[b]:
                    # only add more tokens for examples that aren't done yet
                    token = self.v2t([preds.data[b]])
                    if token == self.END:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
                    else:
                        output_lines[b].append(token)

        if random.random() < 0.1:
            # sometimes output a prediction for debugging
            print('prediction:', ' '.join(output_lines[0]))

        return output_lines

    def _score_candidates(self, cands, xe, encoder_output, hidden):
        # score each candidate separately

        # cands are exs_with_cands x cands_per_ex x words_per_cand
        # cview is total_cands x words_per_cand
        cview = cands.view(-1, cands.size(2))
        cands_xes = xe.expand(xe.size(0), cview.size(0), xe.size(2))
        sz = hidden.size()
        cands_hn = (
            hidden.view(sz[0], sz[1], 1, sz[2])
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

        cand_scores = Variable(
                    self.cand_scores.resize_(cview.size(0)).fill_(0))
        cand_lengths = Variable(
                    self.cand_lengths.resize_(cview.size(0)).fill_(0))

        for i in range(cview.size(1)):
            output = self._apply_attention(cands_xes, cands_encoder_output, cands_hn) \
                    if self.use_attention else cands_xes

            output, cands_hn = self.decoder(output, cands_hn)
            preds, scores = self.hidden_to_idx(output, dropout=False)
            cs = cview.select(1, i)
            non_nulls = cs.ne(self.NULL_IDX)
            cand_lengths += non_nulls.long()
            score_per_cand = torch.gather(scores, 1, cs.unsqueeze(1))
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

    def predict(self, xs, ys=None, cands=None):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available.
        """
        batchsize = len(xs)
        text_cand_inds = None
        encoder_output, hidden = self._encode(xs)

        # next we use END as an input to kick off our decoder
        x = Variable(self.START_TENSOR)
        xe = self.lt(x).unsqueeze(1)
        xes = xe.expand(xe.size(0), batchsize, xe.size(2))

        # list of output tokens for each example in the batch
        output_lines = None

        if ys is not None:
            output_lines = self._decode_and_train(batchsize, xes, ys,
                                                  encoder_output, hidden)

        else:
            if cands is not None:
                text_cand_inds = self._score_candidates(cands, xe,
                                                        encoder_output, hidden)

            output_lines = self._decode_only(batchsize, xes, ys,
                                             encoder_output, hidden)

        return output_lines, text_cand_inds

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        # valid examples
        exs = [ex for ex in observations if 'text' in ex]
        # the indices of the valid (non-empty) tensors
        valid_inds = [i for i, ex in enumerate(observations) if 'text' in ex]

        # set up the input tensors
        batchsize = len(exs)
        # tokenize the text
        xs = None
        if batchsize > 0:
            parsed = [self.parse(ex['text']) for ex in exs]
            max_x_len = max([len(x) for x in parsed])
            if self.truncate:
                # shrink xs to to limit batch computation
                min_x_len = min([len(x) for x in parsed])
                max_x_len = min(min_x_len + 12, max_x_len, 48)
                parsed = [x[-max_x_len:] for x in parsed]
            xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
            # pack the data to the right side of the tensor for this model
            for i, x in enumerate(parsed):
                offset = max_x_len - len(x)
                for j, idx in enumerate(x):
                    xs[i][j + offset] = idx
            if self.use_cuda:
                # copy to gpu
                self.xs.resize_(xs.size())
                self.xs.copy_(xs, async=True)
                xs = Variable(self.xs)
            else:
                xs = Variable(xs)

        # set up the target tensors
        ys = None
        if batchsize > 0 and any(['labels' in ex for ex in exs]):
            # randomly select one of the labels to update on, if multiple
            # append END to each label
            labels = [random.choice(ex.get('labels', [''])) + ' ' + self.END for ex in exs]
            parsed = [self.parse(y) for y in labels]
            max_y_len = max(len(y) for y in parsed)
            if self.truncate:
                # shrink ys to to limit batch computation
                min_y_len = min(len(y) for y in parsed)
                max_y_len = min(min_y_len + 12, max_y_len, 48)
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
            for i in valid_inds:
                if 'label_candidates' in observations[i]:
                    # each candidate tuple is a pair of the parsed version and
                    # the original full string
                    cs = list(observations[i]['label_candidates'])
                    parsed.append([self.parse(c) for c in cs])
                    valid_cands.append((i, cs))
            if len(parsed) > 0:
                # TODO: store lengths of cands separately, so don't have zero
                # padding for varying number of cands per example
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

        return xs, ys, valid_inds, cands, valid_cands

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, valid_inds, cands, valid_cands = self.batchify(observations)

        if xs is None:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available

        predictions, text_cand_inds = self.predict(xs, ys, cands)

        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            curr = batch_reply[valid_inds[i]]
            curr['text'] = ' '.join(c for c in predictions[i] if c != self.END
                                    and c != self.dict.null_token)

        if text_cand_inds is not None:
            for i in range(len(valid_cands)):
                order = text_cand_inds[i]
                batch_idx, curr_cands = valid_cands[i]
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
            model['h2o'] = self.h2o.state_dict()
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
            model = torch.load(read)

        return model['opt'], model

    def set_states(self, states):
        """Set the state dicts of the modules from saved states."""
        self.lt.load_state_dict(states['lt'])
        self.encoder.load_state_dict(states['encoder'])
        self.decoder.load_state_dict(states['decoder'])
        self.h2o.load_state_dict(states['h2o'])
        self.longest_label = states['longest_label']
