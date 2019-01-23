#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from seq2seq.models.seq2seq import Seq2seq
    from seq2seq.models.EncoderRNN import EncoderRNN
    from seq2seq.models.DecoderRNN import DecoderRNN
except ImportError:
    raise ImportError('Please install IBM\'s seq2seq package at '
                      'https://github.com/IBM/pytorch-seq2seq')

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import maintain_dialog_history, PaddingUtils, round_sigfigs

import torch
from torch import optim
import torch.nn as nn

from collections import deque

import os
import math


class IbmSeq2seqAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    For more information, see IBM's repository at
    https://github.com/IBM/pytorch-seq2seq.
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
        agent = argparser.add_argument_group('IBM Seq2Seq Arguments')
        agent.add_argument('--init-model', type=str, default=None,
                           help='load dict/features/weights/opts from this file')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.005,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.5,
                           help='dropout rate')
        agent.add_argument('-clip', '--gradient-clip', type=float, default=-1,
                           help='gradient clipping using l2 norm')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', type='bool', default=True,
                           help='Enable/disable attention over encoded state.')
        agent.add_argument('--maxlength-in', type=int, default=50,
                           help='Maximum input token length.')
        agent.add_argument('--maxlength-out', type=int, default=50,
                           help='Maximum output token length.')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-tr', '--truncate', type=int, default=-1,
                           help='truncate input & output lengths to speed up '
                           'training (may reduce accuracy). This fixes all '
                           'input and output to have a maximum length. This '
                           'reduces the total amount '
                           'of padding in the batches.')
        agent.add_argument('-rnn', '--rnncell', default='gru',
                           help='Choose between different types of RNNs.')
        agent.add_argument('-opt', '--optimizer', default='adam',
                           choices=IbmSeq2seqAgent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        agent.add_argument('-histr', '--history-replies',
                           default='label_else_model', type=str,
                           choices=['none', 'model', 'label',
                                    'label_else_model'],
                           help='Keep replies in the history, or not.')
        IbmSeq2seqAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = self.opt  # there is a deepcopy in the init

        # all instances may need some params
        self.truncate = opt['truncate'] if opt['truncate'] > 0 else None
        self.metrics = {'loss': 0, 'num_tokens': 0}
        self.history = {}
        self.batch_idx = shared and shared.get('batchindex') or 0
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
        else:
            # this is not a shared instance of this class, so do full init
            # answers contains a batch_size list of the last answer produced
            self.answers = [None] * opt['batchsize']

            if self.use_cuda:
                torch.cuda.set_device(opt['gpu'])

            # check first for 'init_model' for loading model from file
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                init_model = opt['init_model']
            # next check for 'model_file'
            elif opt.get('model_file') and os.path.isfile(opt['model_file']):
                init_model = opt['model_file']
            else:
                init_model = None

            if init_model is not None:
                # load model parameters if available
                print('Loading existing model params from ' + init_model)
                new_opt, self.states = self.load(init_model)
                # override model-specific options with stored ones
                opt = self.override_opt(new_opt)

            if opt['dict_file'] is None:
                if init_model is not None and os.path.isfile(init_model + '.dict'):
                    # check first to see if a dictionary exists
                    opt['dict_file'] = init_model + '.dict'
                elif opt.get('model_file'):
                    # otherwise, set default dict-file if it is not set
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

            encoder = EncoderRNN(
                len(self.dict), opt['maxlength_in'], opt['hiddensize'],
                dropout_p=opt['dropout'], input_dropout_p=opt['dropout'],
                n_layers=opt['numlayers'], rnn_cell=opt['rnncell'],
                bidirectional=opt['bidirectional'],
                variable_lengths=True)
            decoder = DecoderRNN(
                len(self.dict), opt['maxlength_out'],
                opt['hiddensize'] * 2 if opt['bidirectional'] else opt['hiddensize'],
                dropout_p=opt['dropout'], input_dropout_p=opt['dropout'],
                n_layers=opt['numlayers'], rnn_cell=opt['rnncell'],
                bidirectional=opt['bidirectional'],
                sos_id=self.START_IDX, eos_id=self.END_IDX,
                use_attention=opt['attention'])
            self.model = Seq2seq(encoder, decoder)

            if self.states:
                # set loaded states if applicable
                self.model.load_state_dict(self.states['model'])

            if self.use_cuda:
                self.model.cuda()

        # set up criteria
        self.criterion = nn.NLLLoss(ignore_index=self.NULL_IDX,
                                    size_average=False)
        if self.use_cuda:
            self.criterion.cuda()

        if 'train' in opt.get('datatype', ''):
            # if model was built, do more setup
            self.clip = opt['gradient_clip']

            # set up tensors once
            self.START = torch.LongTensor([self.START_IDX])

            if self.use_cuda:
                # push to cuda
                self.START = self.START.cuda()

            # set up optimizer
            lr = opt['learningrate']
            optim_class = IbmSeq2seqAgent.OPTIM_OPTS[opt['optimizer']]
            kwargs = {'lr': lr}
            if opt['optimizer'] == 'sgd':
                kwargs['momentum'] = 0.95
                kwargs['nesterov'] = True

            self.optimizer = optim_class(
                [p for p in self.model.parameters() if p.requires_grad],
                **kwargs
            )
            if self.states:
                if self.states['optimizer_type'] != opt['optimizer']:
                    print('WARNING: not loading optim state since optim class '
                          'changed.')
                else:
                    self.optimizer.load_state_dict(self.states['optimizer'])
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', factor=0.5, patience=3, verbose=True)

        self.reset()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                      'attention', 'maxlength-in', 'maxlength-out'}
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
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.history.clear()
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics.clear()
        self.metrics['loss'] = 0
        self.metrics['num_tokens'] = 0

    def report(self):
        m = {}
        if self.metrics['num_tokens'] > 0:
            m['loss'] = self.metrics['loss'] / self.metrics['num_tokens']
            m['ppl'] = math.exp(m['loss'])
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['answers'] = self.answers
        shared['dict'] = self.dict
        shared['START_IDX'] = self.START_IDX
        shared['END_IDX'] = self.END_IDX
        shared['NULL_IDX'] = self.NULL_IDX
        shared['model'] = self.model
        if self.opt.get('numthreads', 1) > 1:
            self.model.share_memory()
        return shared

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        if not obs.get('preprocessed', False):
            obs['text2vec'] = maintain_dialog_history(
                self.history, obs,
                reply=self.answers[self.batch_idx],
                historyLength=self.truncate,
                useReplies=self.opt.get('history_replies'),
                dict=self.dict,
                useStartEndIndices=False)
        else:
            obs['text2vec'] = deque(obs['text2vec'], maxlen=self.truncate)
        self.observation = obs
        self.answers[self.batch_idx] = None
        return obs

    def predict(self, xs, ys=None, is_training=False):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available and param is set.
        """
        x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
        start = self.START.detach()
        starts = start.expand(len(xs), 1)

        if is_training:
            self.model.train()
            self.zero_grad()
            y_in = torch.cat([starts, ys], 1)
            out, hid, result = self.model(xs, x_lens, y_in, teacher_forcing_ratio=True)
            scores = torch.cat(out)
            loss = self.criterion(scores.view(-1, scores.size(-1)), ys.view(-1))
            # save loss to metrics
            target_tokens = ys.ne(self.NULL_IDX).long().sum().item()
            self.metrics['loss'] += loss.double().item()
            self.metrics['num_tokens'] += target_tokens
            # average loss per token
            loss /= target_tokens
            loss.backward()
            self.update_params()
        else:
            self.model.eval()
            out, hid, result = self.model(xs, x_lens)

            if ys is not None:
                # calculate loss on targets
                y_in = torch.cat([starts, ys], 1)
                out, hid, result = self.model(
                    xs, x_lens, y_in, teacher_forcing_ratio=False
                )
                scores = torch.cat(out)
                loss = self.criterion(scores.view(-1, scores.size(-1)), ys.view(-1))
                target_tokens = ys.ne(self.NULL_IDX).long().sum().item()
                self.metrics['loss'] += loss.double().item()
                self.metrics['num_tokens'] += target_tokens

        predictions = torch.cat(result['sequence'], 1)
        return predictions

    def vectorize(self, observations):
        """Convert a list of observations into input & target tensors."""
        is_training = any(['labels' in obs for obs in observations])
        xs, ys, labels, valid_inds, _, _ = PaddingUtils.pad_text(
            observations, self.dict, end_idx=None, null_idx=self.NULL_IDX,
            dq=True, eval_labels=True, truncate=self.truncate)
        if xs is None:
            return None, None, None, None, None, None, None
        xs = torch.LongTensor(xs)

        if self.use_cuda:
            xs = xs.cuda()
        if ys is not None:
            ys = torch.LongTensor(ys)
            if self.use_cuda:
                ys = ys.cuda()

        return xs, ys, labels, valid_inds, is_training

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, labels, valid_inds, is_training = self.vectorize(observations)

        if xs is None:
            # no valid examples, just return empty responses
            return batch_reply

        # produce predictions, train on targets if availables
        predictions = self.predict(xs, ys, is_training)

        if is_training:
            report_freq = 0
        else:
            report_freq = 0.01
        PaddingUtils.map_predictions(
            predictions, valid_inds, batch_reply, observations,
            self.dict, self.END_IDX, report_freq=report_freq, labels=labels,
            answers=self.answers, ys=ys.data)

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
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        return states['opt'], states

    def receive_metrics(self, metrics_dict):
        """Use the metrics to decide when to adjust LR schedule."""
        if 'loss' in metrics_dict:
            self.scheduler.step(metrics_dict['loss'])
