# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import PaddingUtils
from .modules import RNNModel

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

from collections import deque

import copy
import os
import math
import random
import re

class LanguageModelAgent(Agent):
    """ Agent which trains an RNN on a language modeling task.

    It is adapted from the language model featured in Pytorch's examples repo
    here: <https://github.com/pytorch/examples/tree/master/word_language_model>.
    """

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        argparser.set_defaults(batch_sort=False)
        LanguageModelAgent.dictionary_class().add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Language Model Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=200,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=200,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=20,
                           help='initial learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.2,
                           help='dropout rate')
        agent.add_argument('-clip', '--gradient-clip', type=float, default=0.25,
                           help='gradient clipping')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('-rnn', '--rnn-class', default='LSTM',
                           help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
        agent.add_argument('-sl', '--seq-len', type=int, default=35,
                           help='sequence length')
        agent.add_argument('-tied', '--emb-tied', action='store_true',
                           help='tie the word embedding and softmax weights')
        agent.add_argument('-seed', '--random-seed', type=int, default=1111,
                           help='random seed')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-tr', '--truncate-pred', type=int, default=50,
                           help='truncate predictions')
        agent.add_argument('-rf', '--report-freq', type=float, default=0.1,
                           help='report frequency of prediction during eval')

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = self.opt  # there is a deepcopy in the init
        self.states = {}
        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        self.batchsize = opt.get('batchsize', 1)

        if shared:
            # set up shared properties
            self.dict = shared['dict']

            if 'model' in shared:
                # model is shared during hogwild
                self.model = shared['model']
                self.states = shared['states']

            # get NULL token and END token
            self.NULL_IDX = self.dict[self.dict.null_token]
            self.END_IDX = self.dict[self.dict.end_token]

        else:
            # this is not a shared instance of this class, so do full init
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
            self.id = 'LanguageModel'

            # get NULL token and END token
            self.NULL_IDX = self.dict[self.dict.null_token]
            self.END_IDX = self.dict[self.dict.end_token]

            # set model
            self.model = RNNModel(opt, len(self.dict))

            if self.states:
                # set loaded states if applicable
                self.model.load_state_dict(self.states['model'])

            if self.use_cuda:
                self.model.cuda()

        self.next_observe = []
        self.next_batch = []

        self.is_training = True

        if hasattr(self, 'model'):
            # if model was built, do more setup
            self.clip = opt.get('gradient_clip', 0.25)
            # set up criteria
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX)
            if self.use_cuda:
                # push to cuda
                self.criterion.cuda()
            # set up criterion for eval: we do not want to average over size
            self.eval_criterion = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX, size_average=False)
            if self.use_cuda:
                # push to cuda
                self.eval_criterion.cuda()
            # init hidden state
            self.hidden = self.model.init_hidden(self.batchsize)
            # init tensor of end tokens
            self.ends = torch.LongTensor([self.END_IDX for _ in range(self.batchsize)])
            if self.use_cuda:
                self.ends = self.ends.cuda()
            # set up optimizer
            self.lr = opt['learningrate']
            best_val_loss = None

        self.reset()


    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'dropout',
                      'seq_len', 'emb_tied'}
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

    def zero_grad(self):
        """Zero out optimizer."""
        self.model.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        for p in self.model.parameters():
            p.data.add_(-self.lr, p.grad.data)

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['dict'] = self.dict
        shared['NULL_IDX'] = self.NULL_IDX
        shared['END_IDX'] = self.END_IDX
        if self.opt.get('numthreads', 1) > 1:
            shared['model'] = self.model
            self.model.share_memory()
            shared['states'] = self.states
        return shared

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        #shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        seq_len = self.opt['seq_len']
        is_training = True
        if 'eval_labels' in obs:
            is_training = False

        if is_training:
            if 'text' in obs:
                vec = self.parse(obs['text'])
                vec.append(self.END_IDX)
                self.next_observe += vec
            if 'labels' in obs:
                vec = self.parse(obs['labels'][0])
                vec.append(self.END_IDX)
                self.next_observe += vec
            if len(self.next_observe) < (seq_len + 1):
                # not enough to return to make a batch
                # we handle this case in vectorize
                # labels indicates that we are training
                self.observation = {'labels': ''}
                return self.observation
            else:
                vecs_to_return = []
                total = len(self.next_observe) // (seq_len + 1)
                for _ in range(total):
                    observe = self.next_observe[:(seq_len + 1)]
                    self.next_observe = self.next_observe[(seq_len + 1):]
                    vecs_to_return.append(observe)
                dict_to_return = {'text': '', 'labels': '', 'text2vec': vecs_to_return}
                self.observation = dict_to_return
                return dict_to_return
        else:
            self.observation = obs
            return obs

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def get_target_loss(self, data, hidden, targets, y_lens):
        """Calculates the loss with respect to the targets, token by token,
           where each output token is conditioned on either the input or the
           previous target token.
        """
        loss = 0.0
        bsz = data.size(0)

        # feed in inputs without end token
        output, hidden = self.model(data.transpose(0,1), hidden)
        self.hidden = self.repackage_hidden(hidden)
        # feed in end tokens
        output, hidden = self.model(Variable(self.ends[:bsz].view(1,bsz)), self.hidden)
        self.hidden = self.repackage_hidden(hidden)
        output_flat = output.view(-1, len(self.dict))
        loss += self.eval_criterion(output_flat, targets.select(1,0).view(-1)).data

        for i in range(1, targets.size(1)):
            output, hidden = self.model(targets.select(1,i-1).view(1, bsz), self.hidden, no_pack=True)
            self.hidden = self.repackage_hidden(hidden)
            output_flat = output.view(-1, len(self.dict))
            loss += self.eval_criterion(output_flat, targets.select(1,i).view(-1)).data

        return loss/float(sum(y_lens))

    def get_predictions(self, data):
        """Generates predictions word by word until we either reach the end token
           or some max length (opt['truncate_pred']).
        """
        token_list = []
        bsz = data.size(0)
        done = [False for _ in range(bsz)]
        total_done = 0
        hidden = self.model.init_hidden(bsz)

        i = 0
        while total_done < bsz and i <= self.opt['truncate_pred']:
            if i == 0:
                # feed in input without end tokens
                output, hidden = self.model(data.transpose(0,1), hidden)
                hidden = self.repackage_hidden(hidden)
                # feed in end tokens
                output, hidden = self.model(Variable(self.ends[:bsz].view(1,bsz)), hidden)
            else:
                output, hidden = self.model(Variable(word_idx.view(1, bsz)), hidden, no_pack=True)
            hidden = self.repackage_hidden(hidden)
            word_weights = output.squeeze().data.exp()
            if bsz > 1:
                value, word_idx = torch.max(word_weights, 1)
            else:
                value, word_idx = torch.max(word_weights, 0)
            # mark end indices for items in batch
            for k in range(word_idx.size(0)):
                if not done[k]:
                    if int(word_idx[k]) == self.END_IDX:
                        done[k] = True
                        total_done += 1
            token_list.append(word_idx.view(bsz, 1))
            i += 1

        return torch.cat(token_list,1)

    def predict(self, data, hidden, targets=None, is_training=True, y_lens=None):
        """Produce a prediction from our model.
        """
        loss_dict = None
        output = None
        predictions = None
        if is_training:
            self.model.train()
            self.zero_grad()
            output, hidden = self.model(data, hidden)
            loss = self.criterion(output.view(-1, len(self.dict)), targets.view(-1))
            loss.backward(retain_graph=True)
            self.update_params()
            loss_dict = {'lmloss': loss.data}
            loss_dict['lmppl'] = math.exp(loss.data)
        else:
            self.model.eval()
            predictions = self.get_predictions(data)
            loss_dict = {}
            bsz = data.size(0)
            if bsz != self.batchsize:
                self.hidden = self.model.init_hidden(bsz)
            loss = self.get_target_loss(data, self.hidden, targets, y_lens)
            loss_dict['loss'] = loss
            loss_dict['ppl'] = math.exp(loss)

        return output, hidden, loss_dict, predictions

    def vectorize(self, observations, seq_len, is_training):
        """Convert a list of observations into input & target tensors."""
        labels = None
        valid_inds = None
        y_lens = None
        if is_training:
            for obs in observations:
                if obs:
                    if 'text2vec' in obs:
                        self.next_batch += obs['text2vec']
            if len(self.next_batch) <= self.batchsize:
                return None, None, None, None, None
            else:
                data_list = []
                targets_list = []
                # total is the number of batches
                total = len(self.next_batch)//self.batchsize
                for i in range(total):
                    batch = self.next_batch[:self.batchsize]
                    self.next_batch = self.next_batch[self.batchsize:]

                    source = torch.LongTensor(batch).t().contiguous()
                    data = Variable(source[:seq_len])
                    targets = Variable(source[1:])

                    if self.use_cuda:
                        data = data.cuda()
                        targets = targets.cuda()

                    data_list.append(data)
                    targets_list.append(targets)
        else:
            # here we get valid examples and pad them with zeros
            xs, ys, labels, valid_inds, _, y_lens = PaddingUtils.pad_text(
                observations, self.dict, self.END_IDX, self.NULL_IDX)
            if self.use_cuda:
                xs = Variable(xs).cuda()
                ys = Variable(ys).cuda()
            else:
                xs = Variable(xs)
                ys = Variable(ys)
            data_list = [xs]
            targets_list = [ys]

        return data_list, targets_list, labels, valid_inds, y_lens

    def batch_act(self, observations):
        batch_reply = [{'id': self.getID()} for _ in range(len(observations))]
        if any(['labels' in obs for obs in observations]):
            # if we are starting a new training epoch, reinitialize hidden
            if self.is_training == False:
                self.hidden = self.model.init_hidden(self.batchsize)
            self.is_training = True
            data_list, targets_list, _, _, y_lens = self.vectorize(observations, self.opt['seq_len'], self.is_training)
        else:
            # if we just finished training, reinitialize hidden
            if self.is_training == True:
                self.hidden = self.model.init_hidden(self.batchsize)
                self.is_training = False
            data_list, targets_list, labels, valid_inds, y_lens = self.vectorize(observations, self.opt['seq_len'], self.is_training)

        if data_list is None:
            # not enough data to batch act yet, return empty responses
            return batch_reply

        batch_reply = []
        # during evaluation, len(data_list) is always 1
        # during training, len(dat_list) >= 0: vectorize returns a list containing all batches available at the time it is called
        for i in range(len(data_list)):
            temp_dicts = [{'id': self.getID()} for _ in range(len(observations))]
            output, hidden, loss_dict, predictions = self.predict(data_list[i], self.hidden, targets_list[i], self.is_training, y_lens)
            self.hidden = self.repackage_hidden(hidden)

            if predictions is not None:
                # map predictions back to the right order
                PaddingUtils.map_predictions(
                    predictions, valid_inds, temp_dicts, observations,
                    self.dict, self.END_IDX, report_freq=self.opt['report_freq'])

            if loss_dict is not None:
                if 'metrics' in temp_dicts[0]:
                    for k, v in loss_dict.items():
                        temp_dicts[0]['metrics'][k] = v
                else:
                    temp_dicts[0]['metrics'] = loss_dict

            batch_reply += temp_dicts

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
