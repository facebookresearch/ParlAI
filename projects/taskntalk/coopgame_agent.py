#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from .modules import ImgNet, ListenNet, StateNet, SpeakNet, PredictNet

import torch
from torch.autograd import Variable
from torch import optim
from torch.autograd import backward as autograd_backward


class CooperativeGameAgent(Agent):
    """
    Base class for both, the questioner and answerer.

    It can be extended to create custom players of games, other than questioner and
    answerer. It has separate modules to listen (observe), speak (act) and update its
    internal state. Each module is a collection of one or more pytorch modules, and can
    be extended and replaced in the agent as per task requirements.
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
        """
        If different strategy for tokenization and de-tokenization of actions is
        required, override this method to return custom subclass.
        """
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        group = argparser.add_argument_group('Cooperative Game Agent Arguments')
        group.add_argument(
            '--optimizer',
            default='adam',
            choices=CooperativeGameAgent.OPTIM_OPTS.keys(),
            help='Choose between pytorch optimizers. Any member of '
            'torch.optim is valid and will be used with '
            'default params except learning rate (as specified '
            'by -lr).',
        )
        group.add_argument(
            '--learning-rate', default=1e-2, type=float, help='Initial learning rate'
        )
        group.add_argument(
            '--no-cuda',
            action='store_true',
            default=False,
            help='disable GPUs even if available',
        )
        group.add_argument(
            '--gpuid',
            type=int,
            default=-1,
            help='which GPU device to use (defaults to cpu)',
        )
        DictionaryAgent.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'CooperativeGameAgent'
        self.actions = []

        # initialize short (h) and long (c) term states
        self.reset()

        # basic modules for listening, state update and speaking
        # questioner will have `PredictNet`, answerer will have `ImgNet`
        self.listen_net = ListenNet(opt['in_vocab_size'], opt['embed_size'])
        self.state_net = StateNet(opt['embed_size'], opt['state_size'])
        self.speak_net = SpeakNet(opt['state_size'], opt['out_vocab_size'])

        # setup optimizer according to command-line arguments
        self.optimizer = self.setup_optimizer()
        # setup dictionary agent
        self.dict_agent = CooperativeGameAgent.dictionary_class()()

        # transfer agent to GPU if applicable
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        if self.use_cuda:
            print('[ Using CUDA for %s ]' % self.id)
            torch.cuda.set_device(opt['gpuid'])
            for module in self.modules:
                module = module.cuda()

    @property
    def modules(self):
        """
        Property to return a list of pytorch modules.

        Override this method while subclassing, if extra modules are added (for example,
        image feature extractor in answerer).
        """
        return [self.listen_net, self.state_net, self.speak_net]

    def setup_optimizer(self):
        """
        Return a ``torch.nn.optim.optimizer`` according to command-line argument
        ``--optimizer``.

        Override this method to setup optimizer with non-default parameters or use
        custom optimizer not available as choice.
        """
        optim_class = CooperativeGameAgent.OPTIM_OPTS[self.opt['optimizer']]
        kwargs = {'lr': self.opt['learning_rate']}
        if self.opt['optimizer'] == 'sgd':
            kwargs['momentum'] = 0.95
            kwargs['nesterov'] = True
        return optim_class([module.parameters() for module in self.modules], **kwargs)

    def tokenize(self, text):
        """
        Convert text observaton (string) to a ``torch.autograd.Variable`` of tokens
        using ``DictionaryAgent``.
        """
        text_tokens = self.dict.txt2vec(text)
        token_vars = Variable(torch.Tensor(text_tokens))
        if self.use_cuda:
            token_vars = token_vars.cuda()
        return token_vars

    def detokenize(self, vec):
        """
        Convert a ``torch.autograd.Variable`` of tokens into a string.
        """
        text_tokens = vec
        if isinstance(text_tokens, Variable):
            text_tokens = list(text_tokens.data)
        return self.dict.vec2txt(text_tokens)

    def observe(self, observation):
        """
        Update state, given a previous reply by other agent.

        In case of questioner, it can be goal description at start of episode.
        """
        self.observation = observation

        # if episode not done, tokenize, embed and update state
        # at the end of dialog episode, perform backward pass and step
        if not observation.get('episode_done', False):
            text_tokens = self.tokenize(observation['text'])
            token_embeds = self.listen_net(text_tokens)
            if 'image' in observation:
                token_embeds = torch.cat((token_embeds, observation['image']), 1)
                token_embeds = token_embeds.squeeze(1)
            self.h_state, self.c_state = self.state_net(
                token_embeds, (self.h_state, self.c_state)
            )
        else:
            if observation.get('reward', None):
                for action in self.actions:
                    action.reinforce(observation['reward'])
                autograd_backward(
                    self.actions, [None for _ in self.actions], retain_graph=True
                )
                # clamp all gradients between (-5, 5)
                for module in self.modules:
                    for parameter in module.parameters():
                        parameter.grad.data.clamp_(min=-5, max=5)
                self.optimizer.step()
            else:
                # start of dialog episode
                self.optimizer.zero_grad()
                self.reset()

    def act(self):
        """
        Based on current state, utter a reply (string) for next round.
        """
        out_distr = self.speak_net(self.h_state)
        if self.opt['datatype'] == 'train':
            action = out_distr.multinomial()
        else:
            _, action = out_distr.max(1)
            action = action.unsqueeze(1)
        self.actions.append(action)
        action_text = self.detokenize(action.squeeze(1))
        return {'text': action_text, 'id': self.id}

    def reset(self, retain_actions=False):
        """
        Reset internal state (and actions, if specified).
        """
        # TODO(kd): share state across other instances during batch training
        self.h_state = Variable(torch.zeros(1, self.opt['hidden_size']))
        self.c_state = Variable(torch.zeros(1, self.opt['hidden_size']))
        if self.use_cuda:
            self.h_state, self.c_state = self.h_state.cuda(), self.c_state.cuda()

        if not retain_actions:
            self.actions = []


class QuestionerAgent(CooperativeGameAgent):
    """
    Base class for questioner agent.

    It is blindfolded, and has an extra ``predict`` method, which performs action at the
    end of dialog episode to accomplish the goal.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add command-line arguments specifically for this agent.

        Default values at according to (Kottur et al. 2017).
        """
        DictionaryAgent.add_cmdline_args(argparser)
        group = argparser.add_argument_group('Questioner Agent Arguments')
        group.add_argument(
            '--q-in-vocab',
            default=13,
            type=int,
            help='Input vocabulary for questioner. Usually includes '
            'total distinct words spoken by answerer, '
            'questioner itself, and words by which the '
            'goal is described.',
        )
        group.add_argument(
            '--q-embed-size',
            default=20,
            type=int,
            help='Size of word embeddings for questioner',
        )
        group.add_argument(
            '--q-state-size',
            default=100,
            type=int,
            help='Size of hidden state of questioner',
        )
        group.add_argument(
            '--q-out-vocab',
            default=3,
            type=int,
            help='Output vocabulary for questioner',
        )
        group.add_argument(
            '--q-num-pred',
            default=12,
            type=int,
            help='Size of output to be predicted (for goal).',
        )
        super().add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        # transfer opt for super class to use
        opt['in_vocab_size'] = opt['q_in_vocab']
        opt['embed_size'] = opt['q_embed_size']
        opt['state_size'] = opt['q_state_size']
        opt['out_vocab_size'] = opt['q_out_vocab']

        # add a module for prediction (override self.modules later)
        self.predict_net = PredictNet(
            opt['embed_size'], opt['state_size'], opt['num_pred']
        )
        super().__init__(opt, shared)
        self.id = 'QuestionerAgent'

    @property
    def modules(self):
        # override and include predict_net as well
        return [self.listen_net, self.state_net, self.speak_net, self.predict_net]

    def predict(self, tasks, num_tokens):
        """
        Extra method to be executed at the end of episode to carry out goal and decide
        reward on the basis of prediction.
        """
        guess_tokens = []
        for _ in range(num_tokens):
            # explicit task dependence
            task_embeds = self.listen_net(tasks)
            prediction = self.predict_net(task_embeds, (self.h_state, self.c_state))
            guess_tokens.append(prediction)
        return guess_tokens


class AnswererAgent(CooperativeGameAgent):
    """
    Base class for answerer agent.

    It holds visual information, and has an extra  ``img_embed`` method, which extracts
    features from visual content.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add command-line arguments specifically for this agent.

        Default values at according to (Kottur et al. 2017).
        """
        DictionaryAgent.add_cmdline_args(argparser)
        group = argparser.add_argument_group('Questioner Agent Arguments')
        group.add_argument(
            '--a-in-vocab',
            default=13,
            type=int,
            help='Input vocabulary for questioner. Usually includes '
            'total distinct words spoken by answerer, questioner '
            'itself, and words by which the goal is described.',
        )
        group.add_argument(
            '--a-embed-size',
            default=20,
            type=int,
            help='Size of word embeddings for questioner',
        )
        group.add_argument(
            '--a-state-size',
            default=100,
            type=int,
            help='Size of hidden state of questioner',
        )
        group.add_argument(
            '--a-out-vocab',
            default=3,
            type=int,
            help='Output vocabulary for questioner',
        )
        group.add_argument(
            '--a-img-feat-size',
            default=12,
            type=int,
            help='Size of output to be predicted (for goal).',
        )
        group.add_argument(
            '--a-memoryless',
            default=False,
            action='store_true',
            help='Whether to remember previous questions/answers ' 'encountered.',
        )
        super().add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        # transfer opt for super class to use
        opt['in_vocab_size'] = opt['a_in_vocab']
        opt['embed_size'] = opt['a_embed_size']
        opt['state_size'] = opt['a_state_size']
        opt['out_vocab_size'] = opt['a_out_vocab']

        # add a module for grounding visual content
        # opt['a_img_input_size'] should be specified through custom arg or
        # subclass, if needed
        self.img_net = ImgNet(opt['a_img_feat_size'], opt.get('a_img_input_size', None))
        super().__init__(opt, shared)
        self.id = 'AnswererAgent'

    @property
    def modules(self):
        # override and include img_net as well
        return [self.img_net, self.listen_net, self.state_net, self.speak_net]

    def img_embed(self, image):
        """
        Extra method to be executed at the end of episode to carry out goal and decide
        reward on the basis of prediction.
        """
        features = self.img_net(image)
        return features
