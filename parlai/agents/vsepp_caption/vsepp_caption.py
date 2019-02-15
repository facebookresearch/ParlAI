#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.torch_agent import TorchAgent, Output
from .modules import VSEpp, ContrastiveLoss
from parlai.core.utils import round_sigfigs

import torch
import torchvision.transforms as transforms

import os
import numpy as np


class VseppCaptionAgent(TorchAgent):
    """
    Agent which takes an image and retrieves a caption.

    This agent supports modifying the CNN arch used for the image encoder. The
    model then uses a GRU to encode the different candidate captions. These
    encoders map the captions and images to a joint embedding space, so then
    a similarity metric is used to determine which captions are the best match
    for the images.

    For more information see the following paper:
    - VSE++: Improving Visual-Semantic Embeddings with Hard Negatives
      `(Faghri et al. 2017) <arxiv.org/abs/1707.05612>`
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
        agent.add_argument('--max_violation', type='bool', default=True,
                           help='Use max instead of sum in the rank loss.')
        agent.add_argument('-lr', '--learning_rate', type=float,
                           default=0.001, help='learning rate')
        VseppCaptionAgent.dictionary_class().add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'VSEppImageCaption'
        self.mode = None
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
            self.model = VSEpp(opt, self.dict)
            self.metrics = {'loss': 0.0, 'r@': []}

            self.optimizer = self.model.get_optim()
            load_model = None
            states = {}
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                load_model = opt['model_file']
            if load_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]'.format(load_model))
                states = self.load(opt['model_file'])
            self.criterion = ContrastiveLoss(self.use_cuda)

            if self.use_cuda:
                self.model.cuda()
                self.criterion.cuda()

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
        if hasattr(self, "metrics"):
            self.reset_metrics()

    def reset_metrics(self):
        self.metrics['loss'] = 0.0
        self.metrics['r@'] = []

    def candidate_helper(self, candidate_vecs, candidate_labels, is_testing):
        """
        Prepares a list of candidate lists into a format ready for the model
        as pack_padded_sequence requires each candidate must be in descending
        order of length.

        Returns a tuple of:
        (ordered_candidate_tensor, ordered_text_candidate_list,
         candidate_lengths, idx of truth caption*)
        *if exists -- else it will be None
        """
        cand_lens = [c.shape[0] for c in candidate_vecs]
        ind_sorted = sorted(range(len(cand_lens)), key=lambda k: -cand_lens[k])
        truth_idx = ind_sorted.index(0) if not is_testing else None
        cands = [candidate_labels[k] for k in ind_sorted]
        cand_vecs = [candidate_vecs[k] for k in ind_sorted]
        cand_lens = [cand_lens[k] for k in ind_sorted]
        cand_lens = torch.LongTensor(cand_lens)

        padded_cands = torch.LongTensor(len(candidate_vecs),
                                        max(cand_lens)).fill_(self.NULL_IDX)
        if self.use_cuda:
            cand_lens = cand_lens.cuda()
            padded_cands = padded_cands.cuda()

        for i, cand in enumerate(cand_vecs):
            padded_cands[i, :cand.shape[0]] = cand

        return (padded_cands, cands, cand_lens, truth_idx)

    def batchify(self, *args, **kwargs):
        kwargs['sort'] = True
        return super().batchify(*args, **kwargs)

    def train_step(self, batch):
        images = torch.stack([self.transform(img) for img in batch.image])
        if self.use_cuda:
            images = images.cuda(non_blocking=True)

        text_lengths = torch.LongTensor(batch.label_lengths)
        if self.use_cuda:
            text_lengths = text_lengths.cuda()

        self.model.train()
        self.optimizer.zero_grad()
        img_embs, cap_embs = self.model(images, batch.label_vec, text_lengths)
        loss, ranks, top1 = self.criterion(img_embs, cap_embs)
        self.metrics['loss'] += loss.item()
        self.metrics['r@'] += ranks
        loss.backward()
        self.optimizer.step()
        predictions = []
        for score_idx in top1:
            predictions.append(batch.labels[score_idx])
        return Output(predictions, None)

    def eval_step(self, batch):
        images = torch.stack([self.transform(img) for img in batch.image])
        if self.use_cuda:
            images = images.cuda(non_blocking=True)

        # Need to collate then sort the captions by length
        cands = [
            self.candidate_helper(label_cands_vec, label_cands, self.mode == 'test')
            for label_cands_vec, label_cands in
            zip(batch.candidate_vecs, batch.candidates)
        ]
        self.model.eval()
        # Obtain the image embeddings
        img_embs, _ = self.model(images, None, None)
        ranks = []
        top1 = []
        # Each image has their own caption candidates, so we need to
        # iteratively create the embeddings and rank
        for i, (cap, _, lens, truth_idx) in enumerate(cands):
            _, embs = self.model(None, cap, lens)
            # Hack to pass through the truth label's index to compute the
            # rank and top metrics
            offset = truth_idx if truth_idx is not None else 0
            _, rank, top = self.criterion(img_embs[i, :].unsqueeze(0),
                                          embs, offset)
            ranks += rank
            top1.append(top[0])
        self.metrics['r@'] += ranks
        predictions = []
        for i, score_idx in enumerate(top1):
            predictions.append(cands[i][1][score_idx])
        return Output(predictions, None)

    def report(self):
        m = {}
        m['loss'] = self.metrics['loss']
        ranks = np.asarray(self.metrics['r@'])
        m['r@1'] = len(np.where(ranks < 1)[0]) / len(ranks)
        m['r@5'] = len(np.where(ranks < 5)[0]) / len(ranks)
        m['r@10'] = len(np.where(ranks < 10)[0]) / len(ranks)
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m
