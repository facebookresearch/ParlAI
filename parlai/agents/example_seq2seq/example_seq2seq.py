# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import PaddingUtils
from parlai.core.thread_utils import SharedTable

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

import copy


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, numlayers):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=numlayers,
                          batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, numlayers):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=numlayers,
                          batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        emb = self.embedding(input)
        rel = F.relu(emb)
        output, hidden = self.gru(rel, hidden)
        scores = self.softmax(self.out(output))
        return scores, hidden


class ExampleSeq2seqAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    This model is based of Sean Robertson's seq2seq tutorial
    `here <http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`_.
    """

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Seq2Seq Arguments')
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
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-rf', '--report-freq', type=float, default=0.001,
                           help='Report frequency of prediction during eval.')
        ExampleSeq2seqAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        # initialize defaults first
        super().__init__(opt, shared)

        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        if opt.get('numthreads', 1) > 1:
            torch.set_num_threads(1)
        self.id = 'Seq2Seq'

        if not shared:
            # set up model from scratch
            self.dict = DictionaryAgent(opt)
            hsz = opt['hiddensize']
            nl = opt['numlayers']

            # encoder captures the input text
            self.encoder = EncoderRNN(len(self.dict), hsz, nl)
            # decoder produces our output states
            self.decoder = DecoderRNN(len(self.dict), hsz, nl)

            if self.use_cuda:
                self.encoder.cuda()
                self.decoder.cuda()

            if opt.get('numthreads', 1) > 1:
                self.encoder.share_memory()
                self.decoder.share_memory()
        else:
            # ... copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']

            if 'encoder' in shared:
                # hogwild shares model as well
                self.encoder = shared['encoder']
                self.decoder = shared['decoder']

        if hasattr(self, 'encoder'):
            # we set up a model for original instance and multithreaded ones
            self.criterion = nn.NLLLoss()

            # set up optims for each module
            lr = opt['learningrate']
            self.optims = {
                'encoder': optim.SGD(self.encoder.parameters(), lr=lr),
                'decoder': optim.SGD(self.decoder.parameters(), lr=lr),
            }

            self.longest_label = 1
            self.hiddensize = opt['hiddensize']
            self.numlayers = opt['numlayers']
            # we use END markers to end our output
            self.END_IDX = self.dict[self.dict.end_token]
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict[self.dict.null_token]
            # we use START markers to start our output
            self.START_IDX = self.dict[self.dict.start_token]
            self.START = torch.LongTensor([self.START_IDX])
            if self.use_cuda:
                self.START = self.START.cuda()

        self.reset()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def zero_grad(self):
        """Zero out optimizer."""
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        for optimizer in self.optims.values():
            optimizer.step()

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['opt'] = self.opt
        shared['dict'] = self.dict

        if self.opt.get('numthreads', 1) > 1:
            # we're doing hogwild so share the model too
            shared['encoder'] = self.encoder
            shared['decoder'] = self.decoder

        return shared

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def predict(self, xs, ys=None, is_training=False):
        """Produce a prediction from our model.

        Update the model using the targets if available.
        """
        bsz = xs.size(0)
        zeros = Variable(torch.zeros(self.numlayers, bsz, self.hiddensize))
        if self.use_cuda:
            zeros = zeros.cuda()
        starts = Variable(self.START)
        starts = starts.expand(bsz, 1)  # expand to batch size

        if is_training:
            loss = 0
            self.zero_grad()
            self.encoder.train()
            self.decoder.train()
            target_length = ys.size(1)
            # save largest seen label for later
            self.longest_label = max(target_length, self.longest_label)

            encoder_outputs, encoder_hidden = self.encoder(xs, zeros)

            # Teacher forcing: Feed the target as the next input
            y_in = ys.narrow(1, 0, ys.size(1) - 1)
            decoder_input = torch.cat([starts, y_in], 1)
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          encoder_hidden)

            scores = decoder_output.view(-1, decoder_output.size(-1))
            loss = self.criterion(scores, ys.view(-1))
            loss.backward()
            self.update_params()

            _max_score, idx = decoder_output.max(2)
            predictions = idx
        else:
            # just predict
            self.encoder.eval()
            self.decoder.eval()
            encoder_output, encoder_hidden = self.encoder(xs, zeros)
            decoder_hidden = encoder_hidden

            predictions = []
            scores = []
            done = [False for _ in range(bsz)]
            total_done = 0
            decoder_input = starts

            for _ in range(self.longest_label):
                # generate at most longest_label tokens
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden)
                _max_score, idx = decoder_output.max(2)
                preds = idx
                decoder_input = preds
                predictions.append(preds)

                # check if we've produced the end token
                for b in range(bsz):
                    if not done[b]:
                        # only add more tokens for examples that aren't done
                        if preds.data[b][0] == self.END_IDX:
                            # if we produced END, we're done
                            done[b] = True
                            total_done += 1
                if total_done == bsz:
                    # no need to generate any more
                    break
            predictions = torch.cat(predictions, 1)

        return predictions

    def vectorize(self, observations):
        """Convert a list of observations into input & target tensors."""
        is_training = any(('labels' in obs for obs in observations))
        # utility function for padding text and returning lists of indices
        # parsed using the provided dictionary
        xs, ys, labels, valid_inds, _, _ = PaddingUtils.pad_text(
            observations, self.dict, end_idx=self.END_IDX,
            null_idx=self.NULL_IDX, dq=False, eval_labels=True)
        if xs is None:
            return None, None, None, None, None

        # move lists of indices returned above into tensors
        xs = torch.LongTensor(xs)
        if self.use_cuda:
            xs = xs.cuda()
        xs = Variable(xs)

        if ys is not None:
            ys = torch.LongTensor(ys)
            if self.use_cuda:
                ys = ys.cuda()
            ys = Variable(ys)

        return xs, ys, labels, valid_inds, is_training

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # `labels` stores the true labels returned in the `ys` vector
        # `valid_inds` tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, labels, valid_inds, is_training = self.vectorize(observations)

        if xs is None:
            # no valid examples, just return empty responses
            return batch_reply

        predictions = self.predict(xs, ys, is_training)

        # maps returns predictions back to the right `valid_inds`
        # in the example above, a prediction `world` should reply to `hello`
        PaddingUtils.map_predictions(
            predictions.cpu().data, valid_inds, batch_reply, observations,
            self.dict, self.END_IDX, labels=labels,
            answers=labels, ys=ys.data if ys is not None else None,
            report_freq=self.opt.get('report_freq', 0))

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]
