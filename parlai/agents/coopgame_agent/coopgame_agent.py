# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.agents import Agent
from .modules import ImgNet, ListenNet, StateNet, SpeakNet, PredictNet

import torch
from torch.autograd import Variable
from torch import optim
from torch.autograd import backward as autograd_backward


class _CooperativeGameAgent(Agent):
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

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        group = argparser.add_argument_group('Cooperative Game Agent Arguments')
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
        self.listen_net = ListenNet(opt['in_vocab_size'], opt['embed_size'])
        self.state_net = StateNet(opt['embed_size'], opt['state_size'])
        self.speak_net = SpeakNet(opt['state_size'], opt['out_vocab_size'])
        self.reset()

        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        if self.use_cuda:
            print('[ Using CUDA for %s ]' % self.id)
            torch.cuda.set_device(opt['gpuid'])
            for module in self.modules:
                module = module.cuda()

        # TODO(kd): allow chosing optimizer through command line
        self.optimizer = optim.Adam([module.parameters() for module in self.modules],
                                    lr=opt['learning_rate'])

    def observe(self, observation):
        self.observation = observation
        if not observation.get('episode_done', False):
            token_embeds = self.listen_net(observation['text'])
            if 'image' in observation:
                token_embeds = torch.cat((token_embeds, observation['image']), 1)
                token_embeds.squeeze_(1)
            self.h_state, self.c_state = self.state_net(token_embeds,
                                                        (self.h_state, self.c_state))
        else:
            if observation.get('reward', None):
                for action in self.actions:
                    action.reinforce(observation['reward']):
                autograd_backward(self.actions, [None for _ in self.actions],
                                  retain_graph=True)

                # clamp all gradients between (-5, 5)
                for module in self.modules:
                    for parameter in module.parameters():
                        parameter.grad.data.clamp_(min=-5, max=5)
                optimizer.step()
                optimizer.zero_grad()
                self.reset()

    def act(self):
        out_distr = self.speak_net(self.h_state)
        if self.opt['datatype'] == 'train':
            actions = out_distr.multinomial()
        else:
            _, actions = out_distr.max(1)
            actions = actions.unsqueeze(1)
        self.actions.append(actions)
        # TODO(kd): de-tokenize action to text
        return {'text': actions.squeeze(1), 'id': self.id}

    def reset(self, retain_actions=False):
        # TODO(kd): share state across other instances during batch training
        self.h_state = Variable(torch.zeros(1, self.opt['hidden_size']))
        self.c_state = Variable(torch.zeros(1, self.opt['hidden_size']))
        if self.use_cuda:
            self.h_state, self.c_state = self.h_state.cuda(), self.c_state.cuda()

        if not retain_actions:
            self.actions = []

    @property
    def modules(self):
        return [self.listen_net, self.state_net, self.speak_net]
