# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.agents import Agent
from .modules import ListenNet, StateNet, SpeakNet, PredictNet

import torch
from torch.autograd import Variable
from torch import optim
from torch.autograd import backward as autograd_backward


class CooperativeGameAgent(Agent):
    """An agent which can play a cooperative goal-based conversational game.
    Usually these games shall have two agents - one would ask questions and
    another would answer them.

    There is usually an information asymmetry in such games - questioner agent
    will be blind while answerer agent will have the visual information. These
    games shall have a common goal, which can be prediction or image guessing.
    To accomplish this goal, the questioner will perform prediction at the end
    of dialog.

    This class is a base class for both - the questioner and answerer. It has
    separate modules to listen (observe), speak (act) and update its internal
    state. Each module is a collection of one or more pytorch modules. It can
    be extended and replaced in the agent as per task requirements.

    For more information, see Natural Language Does Not Emerge 'Naturally' in
    Multi-Agent Dialog `(Kottur et al. 2017) <https://arxiv.org/abs/1706.08502>`_.
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
        DictionaryAgent.add_cmdline_args(argparser)
        group = argparser.add_argument_group('Cooperative Game Agent Arguments')
        agent.add_argument('--optimizer', default='adam',
                           choices=CooperativeGameAgent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. Any member of torch.optim '
                                'is valid and will be used with default params except learning '
                                'rate (as specified by -lr).')
        group.add_argument('--learning-rate', default=1e-2, type=float,
                           help='Initial learning rate')
        group.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        group.add_argument('--gpuid', type=int, default=-1,
                           help='which GPU device to use (defaults to cpu)')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'CooperativeGameAgent'
        self.actions = []

        # basic modules for listening, state update and speaking
        # questioner will have `PredictNet`, answerer will have `ImgNet` as extras
        self.listen_net = ListenNet(opt['in_vocab_size'], opt['embed_size'])
        self.state_net = StateNet(opt['embed_size'], opt['state_size'])
        self.speak_net = SpeakNet(opt['state_size'], opt['out_vocab_size'])

        # initialize short (h) and long (c) term states
        self.reset()

        # transfer agent to GPU if applicable
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        if self.use_cuda:
            print('[ Using CUDA for %s ]' % self.id)
            torch.cuda.set_device(opt['gpuid'])
            for module in self.modules:
                module = module.cuda()

        # setup dictionary agent
        self.dict_agent = dictionary_class()()

        # setup optimizer according to command-line arguments
        self.optimizer = self.setup_optimizer()

    @property
    def modules(self):
        return [self.listen_net, self.state_net, self.speak_net]

    def setup_optimizer(self):
        optim_class = CooperativeGameAgent.OPTIM_OPTS[opt['optimizer']]
        kwargs = {'lr': self.opt['learning_rate']}
        if opt['optimizer'] == 'sgd':
            kwargs['momentum'] = 0.95
            kwargs['nesterov'] = True
        return optim_class([module.parameters() for module in self.modules],
                            **kwargs)

    def tokenize(self, text):
        text_tokens = self.dict.txt2vec(text)
        # TODO(kd): handle requires_grad=False for valid/test
        return Variable(torch.Tensor(text_tokens), requires_grad=True)

    def detokenize(self, vec):
        text_tokens = vec
        if type(text_tokens) == Variable:
            text_tokens = list(text_tokens.data)
        return self.dict.vec2txt(text_tokens)

    def observe(self, observation):
        self.observation = observation
        if not observation.get('episode_done', False):
            text_tokens = self.tokenize(observation['text'])
            token_embeds = self.listen_net(text_tokens)
            if 'image' in observation:
                token_embeds = torch.cat((token_embeds, observation['image']), 1)
                token_embeds.squeeze_(1)
            self.h_state, self.c_state = self.state_net(token_embeds,
                                                        (self.h_state, self.c_state))
        else:
            if observation.get('reward', None):
                # end of dialog episode, perform backward pass and step
                for action in self.actions:
                    action.reinforce(observation['reward']):
                autograd_backward(self.actions, [None for _ in self.actions],
                                  retain_graph=True)

                # clamp all gradients between (-5, 5)
                for module in self.modules:
                    for parameter in module.parameters():
                        parameter.grad.data.clamp_(min=-5, max=5)
                optimizer.step()
            else:
                # start of dialog episode
                optimizer.zero_grad()
                self.reset()

    def act(self):
        out_distr = self.speak_net(self.h_state)
        if self.opt['datatype'] == 'train':
            action = out_distr.multinomial()
        else:
            _, action = out_distr.max(1)
            action.unsqueeze_(1)
        self.actions.append(action)
        action_text = self.detokenize(action.squeeze(1))
        return {'text': action_text, 'id': self.id}

    def reset(self, retain_actions=False):
        # TODO(kd): share state across other instances during batch training
        self.h_state = Variable(torch.zeros(1, self.opt['hidden_size']))
        self.c_state = Variable(torch.zeros(1, self.opt['hidden_size']))
        if self.use_cuda:
            self.h_state, self.c_state = self.h_state.cuda(), self.c_state.cuda()

        if not retain_actions:
            self.actions = []


class QuestionerAgent(CooperativeGameAgent):

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        group = argparser.add_argument_group('Questioner Agent Arguments')
        parser.add_argument('--q-in-vocab', default=13, type=int,
                            help='Input vocabulary for questioner. Usually includes total '
                                 'distinct words spoken by answerer, questioner itself, '
                                 'and words by which the goal is described.')
        parser.add_argument('--q-embed-size', default=20, type=int,
                            help='Size of word embeddings for questioner')
        parser.add_argument('--q-state-size', default=100, type=int,
                            help='Size of hidden state of questioner')
        parser.add_argument('--q-out-vocab', default=3, type=int,
                            help='Output vocabulary for questioner')
        parser.add_argument('--q-num-pred', default=12, type=int,
                            help='Size of output to be predicted (for goal).')
        super().add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        # transfer opt for super class to use
        opt['in_vocab_size'] = opt['q_in_vocab']
        opt['embed_size'] = opt['q_embed_size']
        opt['state_size'] = opt['q_state_size']
        opt['out_vocab_size'] = opt['q_out_vocab']

        # add a module for prediction (override self.modules later)
        self.predict_net = PredictNet(opt['embed_size'], opt['state_size'], opt['num_pred'])
        super().__init__(opt, shared)
        self.id = 'QuestionerAgent'

    @property
    def modules(self):
        # override and include predict_net as well
        return [self.listen_net, self.state_net, self.speak_net, self.predict_net]

    def predict(self, tasks, num_tokens):
        """Extra method to be executed at the end of episode to carry out goal
        and decide reward on the basis of prediction.
        """
        guess_tokens = []
        for _ in range(num_tokens):
            # explicit task dependence
            task_embeds = self.listen_net(tasks)
            prediction = self.predict_net(task_embeds, (self.h_state, self.c_state))
            guess_tokens.append(prediction)
        return guess_tokens

