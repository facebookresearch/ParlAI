#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import PaddingUtils, round_sigfigs
from parlai.core.thread_utils import SharedTable
from .modules import RNNModel

import torch
from torch.autograd import Variable
import torch.nn as nn

import os
import math
import json


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
        agent = argparser.add_argument_group('Language Model Arguments')
        agent.add_argument('--init-model', type=str, default=None,
                           help='load dict/features/weights/opts from this file')
        agent.add_argument('-hs', '--hiddensize', type=int, default=200,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=200,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
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
        agent.add_argument('-pt', '--person-tokens', type='bool', default=True,
                           help='append person1 and person2 tokens to text')
        # learning rate parameters
        agent.add_argument('-lr', '--learningrate', type=float, default=20,
                           help='initial learning rate')
        agent.add_argument('-lrf', '--lr-factor', type=float, default=1.0,
                           help='mutliply learning rate by this factor when the \
                           validation loss does not decrease')
        agent.add_argument('-lrp', '--lr-patience', type=int, default=10,
                           help='wait before decreasing learning rate')
        agent.add_argument('-lrm', '--lr-minimum', type=float, default=0.1,
                           help='minimum learning rate')
        agent.add_argument('-sm', '--sampling-mode', type='bool', default=False,
                           help='sample when generating tokens instead of taking \
                           the max and do not produce UNK token (when bs=1)')
        LanguageModelAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = self.opt  # there is a deepcopy in the init
        self.metrics = {
            'loss': 0,
            'num_tokens': 0,
            'lmloss': 0,
            'lm_num_tokens': 0
        }
        self.states = {}
        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        self.batchsize = opt.get('batchsize', 1)
        self.use_person_tokens = opt.get('person_tokens', True)
        self.sampling_mode = opt.get('sampling_mode', False)

        if shared:
            # set up shared properties
            self.opt = shared['opt']
            opt = self.opt
            self.dict = shared['dict']

            self.model = shared['model']
            self.metrics = shared['metrics']

            # get NULL token and END token
            self.NULL_IDX = self.dict[self.dict.null_token]
            self.END_IDX = self.dict[self.dict.end_token]

            if 'states' in shared:
                self.states = shared['states']

            if self.use_person_tokens:
                # add person1 and person2 tokens
                self.dict.add_to_dict(self.dict.tokenize("PERSON1"))
                self.dict.add_to_dict(self.dict.tokenize("PERSON2"))

        else:
            # this is not a shared instance of this class, so do full init
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

            # for backwards compatibility: will only be called for older models
            # for which .opt file does not exist
            if (init_model is not None and
                    not os.path.isfile(init_model + '.opt')):
                new_opt = self.load_opt(init_model)
                # load model parameters if available
                print('[ Setting opt from {} ]'.format(
                    init_model
                ))
                # since .opt file does not exist, save one for future use
                print("Saving opt file at:", init_model + ".opt")
                with open(init_model + '.opt', 'w') as handle:
                    json.dump(new_opt, handle)
                opt = self.override_opt(new_opt)

            if ((init_model is not None and
                    os.path.isfile(init_model + '.dict')) or
                    opt['dict_file'] is None):
                opt['dict_file'] = init_model + '.dict'

            # load dictionary and basic tokens & vectors
            self.dict = DictionaryAgent(opt)
            self.id = 'LanguageModel'

            # get NULL token and END token
            self.NULL_IDX = self.dict[self.dict.null_token]
            self.END_IDX = self.dict[self.dict.end_token]

            if self.use_person_tokens:
                # add person1 and person2 tokens
                self.dict.add_to_dict(self.dict.tokenize("PERSON1"))
                self.dict.add_to_dict(self.dict.tokenize("PERSON2"))

            # set model
            self.model = RNNModel(opt, len(self.dict))

            if init_model is not None:
                self.load(init_model)

            if self.use_cuda:
                self.model.cuda()

        self.next_observe = []
        self.next_batch = []

        self.is_training = True

        self.clip = opt.get('gradient_clip', 0.25)
        # set up criteria
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX,
                                             size_average=False)
        if self.use_cuda:
            # push to cuda
            self.criterion.cuda()
        # init hidden state
        self.hidden = self.model.init_hidden(self.batchsize)
        # init tensor of end tokens
        self.ends = torch.LongTensor([self.END_IDX for _ in range(self.batchsize)])
        if self.use_cuda:
            self.ends = self.ends.cuda()
        # set up model and learning rate scheduler parameters
        self.lr = opt['learningrate']
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.best_val_loss = self.states.get('best_val_loss', None)
        self.lr_factor = opt['lr_factor']
        if self.lr_factor < 1.0:
            self.lr_patience = opt['lr_patience']
            self.lr_min = opt['lr_minimum']
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=self.lr_factor, verbose=True,
                patience=self.lr_patience, min_lr=self.lr_min)
            # initial step for scheduler if self.best_val_loss is initialized
            if self.best_val_loss is not None:
                self.scheduler.step(self.best_val_loss)
        else:
            self.scheduler = None

        self.reset()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.
        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'dropout',
                      'seq_len', 'emb_tied', 'truncate_pred', 'report_freq',
                      'person_tokens', 'learningrate'}
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
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics.clear()
        self.metrics['loss'] = 0
        self.metrics['lmloss'] = 0
        self.metrics['num_tokens'] = 0
        self.metrics['lm_num_tokens'] = 0

    def report(self):
        m = {}
        if self.metrics['num_tokens'] > 0:
            m['loss'] = self.metrics['loss'] / self.metrics['num_tokens']
            m['ppl'] = math.exp(m['loss'])
        if self.metrics['lm_num_tokens'] > 0:
            m['lmloss'] = self.metrics['lmloss'] / self.metrics['lm_num_tokens']
            m['lmppl'] = math.exp(m['lmloss'])
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['opt'] = self.opt
        shared['dict'] = self.dict
        shared['NULL_IDX'] = self.NULL_IDX
        shared['END_IDX'] = self.END_IDX
        shared['model'] = self.model
        if self.opt.get('numthreads', 1) > 1:
            if type(self.metrics) == dict:
                # move metrics and model to shared memory
                self.metrics = SharedTable(self.metrics)
                self.model.share_memory()
            shared['states'] = {  # only need to pass optimizer states
                'optimizer': self.optimizer.state_dict(),
            }
        shared['metrics'] = self.metrics
        return shared

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        seq_len = self.opt['seq_len']
        is_training = True
        if 'labels' not in obs:
            is_training = False

        if is_training:
            if 'text' in obs:
                if self.use_person_tokens:
                    obs['text'] = 'PERSON1 ' + obs['text']
                vec = self.parse(obs['text'])
                vec.append(self.END_IDX)
                self.next_observe += vec
            if 'labels' in obs:
                if self.use_person_tokens:
                    labels = [
                        'PERSON2 ' + label
                        for label in obs['labels']
                        if label != ''
                    ]
                    obs['labels'] = tuple(labels)
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
            if 'text' in obs:
                if self.use_person_tokens:
                    obs['text'] = 'PERSON1 ' + obs['text']
            if 'eval_labels' in obs:
                if self.use_person_tokens:
                    eval_labels = [
                        'PERSON2 ' + label
                        for label in obs['eval_labels']
                        if label != ''
                    ]
                    obs['eval_labels'] = tuple(eval_labels)
            self.observation = obs
            return obs

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if isinstance(h, Variable):
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def get_target_loss(self, data, hidden, targets):
        """Calculates the loss with respect to the targets, token by token,
           where each output token is conditioned on either the input or the
           previous target token.
        """
        loss = 0.0
        bsz = data.size(0)

        # during interactive mode, when no targets exist, we return 0
        if targets is None:
            return loss

        # feed in inputs without end token
        output, hidden = self.model(data.transpose(0, 1), hidden)
        self.hidden = self.repackage_hidden(hidden)
        # feed in end tokens
        output, hidden = self.model(Variable(self.ends[:bsz].view(1, bsz)), self.hidden)
        self.hidden = self.repackage_hidden(hidden)
        output_flat = output.view(-1, len(self.dict))
        loss += self.criterion(output_flat, targets.select(1, 0).view(-1)).data

        for i in range(1, targets.size(1)):
            output, hidden = self.model(
                targets.select(1, i - 1).view(1, bsz),
                self.hidden,
                no_pack=True
            )
            self.hidden = self.repackage_hidden(hidden)
            output_flat = output.view(-1, len(self.dict))
            loss += self.criterion(output_flat, targets.select(1, i).view(-1)).data

        return loss

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
        word_idx = None
        while total_done < bsz and i <= self.opt['truncate_pred']:
            if i == 0:
                # feed in input without end tokens
                output, hidden = self.model(data.transpose(0, 1), hidden)
                hidden = self.repackage_hidden(hidden)
                # feed in end tokens
                output, hidden = self.model(
                    Variable(self.ends[:bsz].view(1, bsz)), hidden
                )
            else:
                output, hidden = self.model(
                    Variable(word_idx.view(1, bsz)), hidden, no_pack=True
                )
            hidden = self.repackage_hidden(hidden)
            word_weights = output.squeeze().data.exp()
            if bsz > 1:
                _, word_idx = torch.max(word_weights, 1)
            else:
                if self.sampling_mode:
                    unk_idx = self.dict[self.dict.unk_token]
                    # make word_weights have smaller norm so that calculated
                    # norm does not blow up
                    word_weights = word_weights.div(1e10)
                    # make word_weights have L2 norm 1
                    ww_norm = torch.norm(word_weights, p=2)
                    word_weights = word_weights.div(ww_norm)
                    # square distribution
                    word_weights = torch.mul(word_weights, word_weights)
                    # sample distribution
                    word_idx = torch.multinomial(word_weights, 1)
                    # do not produce UNK token
                    while word_idx == unk_idx:
                        word_idx = torch.multinomial(word_weights, 1)
                else:
                    _, word_idx = torch.max(word_weights, 0)
            # mark end indices for items in batch
            word_idx = word_idx.view(-1)
            for k in range(word_idx.size(0)):
                if not done[k]:
                    if int(word_idx[k]) == self.END_IDX:
                        done[k] = True
                        total_done += 1
            token_list.append(word_idx.view(bsz, 1))
            i += 1

        return torch.cat(token_list, 1)

    def predict(self, data, hidden, targets=None, is_training=True, y_lens=None):
        """Produce a prediction from our model."""
        output = None
        predictions = None
        if is_training:
            self.model.train()
            self.zero_grad()
            output, hidden = self.model(data, hidden)
            loss = self.criterion(output.view(-1, len(self.dict)), targets.view(-1))
            # save loss to metrics
            target_tokens = targets.ne(self.NULL_IDX).float().sum().item()
            self.metrics['lmloss'] += loss.double().item()
            self.metrics['lm_num_tokens'] += target_tokens
            # average loss per token
            loss /= target_tokens
            loss.backward(retain_graph=True)
            self.update_params()
        else:
            self.model.eval()
            predictions = self.get_predictions(data)
            bsz = data.size(0)
            if bsz != self.batchsize:
                self.hidden = self.model.init_hidden(bsz)
            if targets is not None:
                loss = self.get_target_loss(data, self.hidden, targets)
                self.metrics['loss'] += loss
                self.metrics['num_tokens'] += sum(y_lens)

        return output, hidden, predictions

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
                total = len(self.next_batch) // self.batchsize
                for _ in range(total):
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
                observations, self.dict, end_idx=self.END_IDX,
                null_idx=self.NULL_IDX)

            if self.use_cuda:
                if xs is not None:
                    xs = Variable(torch.LongTensor(xs)).cuda()
                if ys is not None:
                    ys = Variable(torch.LongTensor(ys)).cuda()
            else:
                if xs is not None:
                    xs = Variable(torch.LongTensor(xs))
                if ys is not None:
                    ys = Variable(torch.LongTensor(ys))
            data_list = [xs]
            targets_list = [ys]

        return data_list, targets_list, labels, valid_inds, y_lens

    def batch_act(self, observations):
        batch_reply = [{'id': self.getID()} for _ in range(len(observations))]
        if any(['labels' in obs for obs in observations]):
            # if we are starting a new training epoch, reinitialize hidden
            if not self.is_training:
                self.hidden = self.model.init_hidden(self.batchsize)
            self.is_training = True
            data_list, targets_list, _c, _v, y_lens = self.vectorize(
                observations, self.opt['seq_len'], self.is_training
            )
        else:
            # if we just finished training, reinitialize hidden
            if self.is_training:
                self.hidden = self.model.init_hidden(self.batchsize)
                self.is_training = False
            data_list, targets_list, labels, valid_inds, y_lens = self.vectorize(
                observations, self.opt['seq_len'], self.is_training
            )

        if data_list is None:
            # not enough data to batch act yet, return empty responses
            return batch_reply

        batch_reply = []
        # during evaluation, len(data_list) is always 1
        # during training, len(dat_list) >= 0: vectorize returns a list
        #     containing all batches available at the time it is called
        for i in range(len(data_list)):
            temp_dicts = [{'id': self.getID()} for _ in range(len(observations))]
            # ignore case when we do not return any valid indices
            if data_list[i] is not None:
                output, hidden, predictions = self.predict(
                    data_list[i], self.hidden, targets_list[i],
                    self.is_training, y_lens
                )
                self.hidden = self.repackage_hidden(hidden)

                if predictions is not None:
                    # map predictions back to the right order
                    PaddingUtils.map_predictions(
                        predictions.cpu(), valid_inds, temp_dicts, observations,
                        self.dict, self.END_IDX, report_freq=self.opt['report_freq'])

            batch_reply += temp_dicts

        # for prediction metrics computations, we get rid of PERSON1 and PERSON2 tokens
        if not self.is_training:
            for reply in batch_reply:
                if 'text' in reply:
                    reply['text'] = reply['text'].replace('PERSON1 ', '')
                    reply['text'] = reply['text'].replace('PERSON2 ', '')

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
            model['best_val_loss'] = self.best_val_loss

            with open(path, 'wb') as write:
                torch.save(model, write)
            # save opt file
            with open(path + '.opt', 'w') as handle:
                json.dump(self.opt, handle)

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()

    def receive_metrics(self, metrics_dict):
        if 'loss' in metrics_dict and self.scheduler is not None:
            self.scheduler.step(metrics_dict['loss'])

    def load_opt(self, path):
        """Return opt, states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        return states['opt']

    def load(self, path):
        """Load model states."""
        if os.path.isfile(path):
            # load model parameters if available
            print('[ Loading existing model params from {} ]'.format(path))
            self.states = torch.load(path, map_location=lambda cpu, _: cpu)
            self.model.load_state_dict(self.states['model'])
