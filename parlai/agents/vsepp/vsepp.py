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
import numpy as np


class VseppAgent(TorchAgent):
    """
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        TorchAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Image Caption Model Arguments')
        agent.add_argument('--word_dim', default=300, type=int,
                           help='Dimensionality of the word embedding.')
        agent.add_argument('--embed_size', default=1024, type=int,
                           help='Dimensionality of the joint embedding.')
        agent.add_argument('--num_layers', default=1, type=int,
                           help='Number of GRU layers.')
        agent.add_argument('--finetune', type='bool', default=False,
                           help='Finetune the image encoder')
        agent.add_argument('--cnn_type', default='resnet152',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
        agent.add_argument('--no_imgnorm', type='bool', default=False,
                           help='Do not normalize the image embeddings.')
        agent.add_argument('--margin', default=0.2, type=float,
                           help='Rank loss margin.')

        # agent.add_argument('--embed_size', type=int , default=256,
        #                    help='dimension of word embedding vectors')
        # agent.add_argument('--hidden_size', type=int , default=512,
        #                    help='dimension of lstm hidden states')
        # agent.add_argument('--num_layers', type=int , default=1,
        #                    help='number of layers in lstm')
        # agent.add_argument('--max_pred_length', type=int, default=20,
        #                    help='maximum length of predicted caption in eval mode')
        agent.add_argument('-lr', '--learning_rate', type=float, default=0.0002,
                           help='learning rate')
        # agent.add_argument('-opt', '--optimizer', default='adam',
        #                    choices=['sgd', 'adam'],
        #                    help='Choose either sgd or adam as the optimizer.')
        # agent.add_argument('--use_feature_state', type='bool',
        #                    default=True,
        #                    help='Initialize LSTM state with image features')
        # agent.add_argument('--concat_img_feats', type='bool', default=True,
        #                    help='Concat resnet feats to each token during generation')
        VseppAgent.dictionary_class().add_cmdline_args(argparser)

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

    def candidate_helper(self, candidate_vecs, candidates):
        """
        Prepares a list of candidate lists into a format ready for the model
        as pack_padded_sequence requires each candidate must be in descending
        order of length.

        Returns a tuple of:
        (ordered_candidate_tensor, ordered_text_candidate_list,
         candidate_lengths, idx of truth caption*)
        *if exists
        """
        cand_lens = [c.shape[0] for c in candidate_vecs]
        ind_sorted = sorted(range(len(cand_lens)), key=lambda k: -cand_lens[k])
        truth_idx = ind_sorted.index(0)
        # import pdb; pdb.set_trace()
        cand_vecs = [candidate_vecs[k] for k in ind_sorted]
        cands = [candidates[k] for k in ind_sorted]
        cand_lens = [cand_lens[k] for k in ind_sorted]
        cand_lens = torch.LongTensor(cand_lens)

        padded_cands = torch.LongTensor(len(cands),
                                     max(cand_lens)).fill_(self.NULL_IDX)
        if self.use_cuda:
            cand_lens = cand_lens.cuda()
            padded_cands = padded_cands.cuda()

        for i, cand in enumerate(cand_vecs):
            padded_cands[i, :cand.shape[0]] = cand

        return (padded_cands, cands, cand_lens, truth_idx)



    def batch_act(self, observations):
        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]

        is_training = any(['labels' in obs for obs in observations])

        vec_obs = [self.vectorize(obs)
                   for obs in observations]

        # shift the labels into the text field so they're ordered
        # by length
        for item in observations:
            if is_training:
                if 'labels' in item:
                    item['text'] = item['labels'][0]
                    item['text_vec'] = item['labels_vec'][0]
            else:
                if 'eval_labels' in item:
                    item['text'] = item['eval_labels'][0]
                    item['text_vec'] = item['eval_labels_vec'][0]

        xs, x_lens, _, labels, valid_inds = self.map_valid(vec_obs)


        images = torch.stack([self.transform(observations[idx]['image'])
                              for idx in valid_inds])
        if self.use_cuda:
            images = images.cuda(async=True)

        # Need 2 different flows for training and for eval/test
        if is_training:
            loss, top1 = self.predict(images, xs, x_lens, cands=None,
                                      is_training=is_training)

            if loss is not None:
                batch_reply[0]['metrics'] = {'loss': loss.item()}

            predictions = []
            for score_idx in top1:
                predictions.append(labels[score_idx])

            # unmap_pred = self.unmap_valid(top1, valid_inds, batch_size)
        else:
            # NEed some way to determine if we are on test
            # Need to collate then sort the captions by length
            cands = [self.candidate_helper(vec_obs[idx]['candidate_labels_vec'],
                                           vec_obs[idx]['candidate_labels'])
                     for idx in valid_inds]
            _, top1 = self.predict(images, None, None, cands, is_training)
            predictions = []
            for i, score_idx in enumerate(top1):
                predictions.append(cands[i][1][score_idx])

        unmap_pred = self.unmap_valid(predictions, valid_inds, batch_size)

        for rep, pred in zip(batch_reply, unmap_pred):
            if pred is not None:
                rep['text'] = pred
        # print(batch_reply)
        return batch_reply


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
            # Obtain the image embeddings
            img_embs, _ = self.model(xs, None, None)
            ranks = []
            top1 = []
            # Each image has their own caption candidates, so we need to
            # iteratively create the embeddings and rank
            for i, (cap, _, lens, truth_idx) in enumerate(cands):
                _, embs = self.model(None, cap, lens)
                # Now we can compare the
                offset = truth_idx if truth_idx is not None else 0
                _, rank, top = self.criterion(img_embs[i, :].unsqueeze(0), embs, offset)
                ranks += rank
                top1 += top
            self.metrics['r@'] += ranks

        return loss, top1

    def report(self):
        m = {}
        m['loss'] = self.metrics['loss']
        ranks = np.asarray(self.metrics['r@'])
        m['r@1'] = len(np.where(ranks < 1)[0]) / len(ranks)*1.0
        m['r@5'] = len(np.where(ranks < 5)[0]) / len(ranks)*1.0
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

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
