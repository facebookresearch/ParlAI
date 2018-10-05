#!/usr/bin/env python3

# This file is covered under the Apache 2.0 License listed here
# <https://github.com/fartashf/vsepp/blob/master/LICENSE> as it is a
# Derivative Work of the repo.

import torch
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import torchvision.models as models

import numpy as np


class VSEpp(nn.Module):
    """
    Model based on:
    - VSE++: Improving Visual-Semantic Embeddings with Hard Negatives
      `(Faghri et al. 2017) <arxiv.org/abs/1707.05612>`
    Original Implementation found here: <https://github.com/fartashf/vsepp>
    """

    def __init__(self, opt, dict):
        super().__init__()
        self.opt = opt
        self.dict = dict
        self.img_enc = EncoderImage(embed_size=opt['embed_size'],
                                    finetune=opt['finetune'],
                                    cnn_type=opt['cnn_type'],
                                    no_imgnorm=opt['no_imgnorm'])
        self.txt_enc = EncoderText(vocab_size=len(self.dict.tok2ind),
                                   word_dim=opt['word_dim'],
                                   embed_size=opt['embed_size'],
                                   num_layers=opt['num_layers'])

    def forward(self, images, captions, lengths):
        img_emb = self.img_enc(images) if images is not None else None
        cap_emb = self.txt_enc(captions, lengths) if captions is not None else None
        return img_emb, cap_emb

    def get_optim(self):
        kwargs = {'lr': float(self.opt['learning_rate']),
                  'amsgrad': True}
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        if self.opt['finetune']:
            params += list(self.img_enc.cnn.parameters())
        optimizer = optim.Adam(params, **kwargs)
        return optimizer


def dot_sim(im, s):
    """
    Dot product similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def l2norm(X):
    """
    L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss.
    """
    def __init__(self, use_cuda, margin=0, max_violation=True):
        super().__init__()
        self.use_cuda = use_cuda
        self.margin = margin
        self.sim = dot_sim
        self.max_violation = max_violation

    def forward(self, im, caps, offset=0):
        # Compute the similarity of each image/caption pair
        scores = self.sim(im, caps)
        diagonal = scores.diag().view(im.shape[0], 1)
        d1 = diagonal.expand(scores.size())
        d2 = diagonal.t().expand(scores.size())

        # Caption retrieval score
        cost_cap = (self.margin + scores - d1).clamp(min=0)
        # image retrieval score
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(im.shape[0]) > 0.5
        if self.use_cuda:
            mask = mask.cuda()
        cost_cap = cost_cap.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)

        # Compute the metrics (ranks, top1)
        if self.use_cuda:
            sorted_ranks = np.flip(np.argsort(scores.detach().cpu().numpy()), 1)
        else:
            sorted_ranks = np.flip(np.argsort(scores.detach().numpy()), 1)
        top1 = sorted_ranks[:, 0]
        ranks = []
        for idx in range(im.shape[0]):
            ranks.append(np.where(sorted_ranks[idx, :] == (idx + offset))[0][0])

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_cap = cost_cap.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_cap.sum() + cost_im.sum(), ranks, top1


class EncoderImage(nn.Module):
    def __init__(self, embed_size, finetune=False, cnn_type='resnet152',
                 no_imgnorm=False):
        """Load pretrained CNN and replace top fc layer."""
        super().__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch):
        """Load a pretrained CNN and parallelize over GPUs
        """
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
        else:
            model = nn.DataParallel(model)

        return model

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)
        # normalization in the image embedding space
        features = l2norm(features)
        # linear projection to the joint embedding space
        features = self.fc(features)
        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        return features


class EncoderText(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers):
        super().__init__()
        self.embed_size = embed_size
        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        # Forward propagate RNN
        _, out = self.rnn(packed)
        out = out.squeeze(0)
        # normalization in the joint embedding space
        out = l2norm(out)

        return out
