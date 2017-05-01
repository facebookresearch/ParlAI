# Copyright 2004-present Facebook. All Rights Reserved.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable
from .utils import load_embeddings, AverageMeter
from .rnn_reader import RnnDocReader

logger = logging.getLogger('DrQA')


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, updating train examples, and predicting valid examples.
    """

    def __init__(self, opt, word_dict, state_dict=None):
        # Book keeping.
        self.opt = opt
        self.train_loss = AverageMeter()
        self.updates = 0

        # Word embeddings.
        if 'embedding_file' in opt:
            logger.info('[ Loading pre-trained embeddings ]')
            embeddings = load_embeddings(opt, word_dict)
            logger.info('[ Num embeddings = %d ]' % embeddings.size(0))
        else:
            embeddings = None

        # Fine-tuning special words.
        if self.opt['tune_partial'] > 0:
            logger.info('[ Tuning top %d words ]' % self.opt['tune_partial'])
            tune_partial = self.opt['tune_partial']
            for i in range(0, 5):
                logger.info(word_dict[i])
            logger.info('...')
            for i in range(tune_partial - 5, tune_partial):
                logger.info(word_dict[i])
            tune_indices = list(range(tune_partial))
        else:
            tune_indices = None

        # Building network.
        logger.info('[ Initializing model ]')
        self.network = RnnDocReader(
            opt, embeddings, tune_indices, word_dict['<NULL>']
        )
        if state_dict:
            self.network.load_state_dict(state_dict['network'])
        if opt['cuda']:
            self.network.cuda()

        # Building optimizer.
        logger.info('[ Make optimizer (%s) ]' % opt['optimizer'])
        parameters = filter(lambda p: p.requires_grad,
                            self.network.parameters())
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:5]]
            target_s = Variable(ex[5].cuda(async=True))
            target_e = Variable(ex[6].cuda(async=True))
        else:
            inputs = [Variable(e) for e in ex[:5]]
            target_s = Variable(ex[5])
            target_e = Variable(ex[6])

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.data[0], ex[0].size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.network.partial_reset()

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:5]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:5]]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        max_len = self.opt['max_len'] or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])

        return predictions
