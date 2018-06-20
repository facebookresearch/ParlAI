# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.torch_agent import TorchAgent
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from .modules import VSEpp, ContrastiveLoss
from parlai.core.utils import round_sigfigs

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

import math
import os
import random


class VSEppAgent(TorchAgent):
    """
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        TorchAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Image Caption Model Arguments')
        # agent.add_argument('--embed_size', type=int , default=256,
        #                    help='dimension of word embedding vectors')
        # agent.add_argument('--hidden_size', type=int , default=512,
        #                    help='dimension of lstm hidden states')
        # agent.add_argument('--num_layers', type=int , default=1,
        #                    help='number of layers in lstm')
        # agent.add_argument('--max_pred_length', type=int, default=20,
        #                    help='maximum length of predicted caption in eval mode')
        # agent.add_argument('-lr', '--learning_rate', type=float, default=0.001,
        #                    help='learning rate')
        # agent.add_argument('-opt', '--optimizer', default='adam',
        #                    choices=['sgd', 'adam'],
        #                    help='Choose either sgd or adam as the optimizer.')
        # agent.add_argument('--use_feature_state', type='bool',
        #                    default=True,
        #                    help='Initialize LSTM state with image features')
        # agent.add_argument('--concat_img_feats', type='bool', default=True,
        #                    help='Concat resnet feats to each token during generation')
        VSEppAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'VSEpp Image Caption'
        if not shared:
            self.image_size = opt['image_size']
            self.crop_size = opt['image_cropsize']

            # initialize the transform function using torch vision.
            self.transform = transforms.Compose([
                transforms.Scale(self.image_size),
                transforms.RandomCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.model = VSEpp(opt, self.dict, self.use_cuda)
            self.metrics = {'loss': 0.0, 'r@': []}

            load_model = None
            states = {}
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                load_model = opt['model_file']
            if load_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]'.format(load_model))
                states = self.load(opt['model_file'])

            if states:
                self.model.load_state_dict(states['model'])

            self.criterion = ContrastiveLoss(self.use_cuda)

            if self.use_cuda:
                self.model.cuda()
                self.criterion.cuda()

            self.optimizer = self.model.get_optim()
            if 'optimizer' in states:
                try:
                    self.optimizer.load_state_dict(states['optimizer'])
                except ValueError:
                    print('WARNING: not loading optim state since model '
                          'params changed.')
                if self.use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()

        self.reset()

    def reset(self):
        self.observation = None
        self.episode_done = False
        if hasattr(self, "metrics"):
            self.reset_metrics()

    def reset_metrics(self):
        self.metrics['loss'] = 0.0
        self.metrics['r@'] = []

    def observe(self, observation):
        """Save observation for act."""
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def batch_act(self, observations):
        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]

        is_training = any(['labels' in obs for obs in observations])

        vec_obs = [self.vectorize(obs)
                   for obs in observations]

        # Need 2 different flows for training and for eval/test
        if is_training:
            # shift the labels into the text field so they're ordered
            # by length
            for item in observations:
                if 'labels' in item:
                    item['text'] = item['labels'][0]
                    item['text_vec'] = item['labels_vec'][0]
            xs, x_lens, _, labels, valid_inds = self.map_valid(vec_obs)

            if xs is None:
                return batch_reply

            images = [self.transform(observations[idx]['image'])
                      for idx in valid_inds]
            if self.use_cuda:
                images = images.cuda(async=True)

            predictions, loss = self.predict(images, xs, x_lens, cands=None,
                                             is_training=is_training)

            if loss is not None:
                batch_reply[0]['metrics'] = {'loss': loss.item()}

            unmap_pred = self.unmap_valid(predictions, valid_inds, batch_size)
        else:
            cands =
            pass
        pass

    def predict(self, xs, ys=None, y_lens=None, cands=None, is_training=False):
        loss = None
        if is_training:
            self.model.train()
            self.optimizer.zero_grad()
            img_embs, cap_embs = self.model(xs, ys, y_lens)
            loss, ranks, top1 = self.criterion(img_embs, cap_embs)
            self.metrics['loss'] += loss.item()
            self.metrics['r@'] += ranks
            loss.backward()
            self.optimizer.step()
        else:
            self.model.eval()

        return loss, ranks, top1

    def report(self):
        pass

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['metrics'] = self.metrics
        shared['model'] = self.model
        shared['states'] = {  # only need to pass optimizer states
            'optimizer': self.optimizer.state_dict()
        }
        return shared

    def act(self):
        return self.batch_act([self.observation])[0]
