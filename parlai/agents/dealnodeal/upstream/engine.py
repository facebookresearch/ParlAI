# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Training utilities.
"""

import argparse
import random
import pdb
import time
import itertools
import sys
import copy
import re

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from .data import STOP_TOKENS


class Criterion(object):
    """Weighted CrossEntropyLoss."""
    def __init__(self, dictionary, device_id=None, bad_toks=[], size_average=True):
        w = torch.Tensor(len(dictionary)).fill_(1)
        for tok in bad_toks:
            w[dictionary.get_idx(tok)] = 0.0
        if device_id is not None:
            w = w.cuda(device_id)
        self.crit = nn.CrossEntropyLoss(w, size_average=size_average)

    def __call__(self, out, tgt):
        return self.crit(out, tgt)


class Engine(object):
    """The training engine.

    Performs training and evaluation.
    """
    def __init__(self, model, args, device_id=None, verbose=False):
        self.model = model
        self.args = args
        self.device_id = device_id
        self.verbose = verbose
        self.opt = optim.SGD(self.model.parameters(), lr=self.args.lr,
            momentum=self.args.momentum,
            nesterov=(self.args.nesterov and self.args.momentum > 0))
        self.crit = Criterion(self.model.word_dict, device_id=device_id)
        self.sel_crit = Criterion(
            self.model.item_dict, device_id=device_id, bad_toks=['<disconnect>', '<disagree>'])
        if self.args.visual:
            self.model_plot = vis.ModulePlot(self.model, plot_weight=False, plot_grad=True)
            self.loss_plot = vis.Plot(['train', 'valid', 'valid_select'],
                'loss', 'loss', 'epoch', running_n=1)
            self.ppl_plot = vis.Plot(['train', 'valid', 'valid_select'],
                'perplexity', 'ppl', 'epoch', running_n=1)

    def forward(model, batch, volatile=True):
        """A helper function to perform a forward pass on a batch."""
        # extract the batch into contxt, input, target and selection target
        ctx, inpt, tgt, sel_tgt = batch
        # create variables
        ctx = Variable(ctx, volatile=volatile)
        inpt = Variable(inpt, volatile=volatile)
        tgt = Variable(tgt, volatile=volatile)
        if sel_tgt is not None:
            sel_tgt = Variable(sel_tgt, volatile=volatile)

        # get context hidden state
        ctx_h = model.forward_context(ctx)
        # create initial hidden state for the language rnn
        lang_h = model.zero_hid(ctx_h.size(1), model.args.nhid_lang)

        # perform forward for the language model
        out, lang_h = model.forward_lm(inpt, lang_h, ctx_h)
        # perform forward for the selection
        sel_out = model.forward_selection(inpt, lang_h, ctx_h)

        return out, lang_h, tgt, sel_out, sel_tgt

    def get_model(self):
        """Extracts the model."""
        return self.model

    def train_pass(self, N, trainset):
        """Training pass."""
        # make the model trainable
        self.model.train()

        total_loss = 0
        start_time = time.time()

        # training loop
        for batch in trainset:
            self.t += 1
            # forward pass
            out, hid, tgt, sel_out, sel_tgt = Engine.forward(self.model, batch, volatile=False)

            # compute LM loss and selection loss
            loss = self.crit(out.view(-1, N), tgt)
            loss += self.sel_crit(sel_out, sel_tgt) * self.model.args.sel_weight
            self.opt.zero_grad()
            # backward step with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)
            self.opt.step()

            if self.args.visual and self.t % 100 == 0:
                self.model_plot.update(self.t)

            total_loss += loss.data[0]

        total_loss /= len(trainset)
        time_elapsed = time.time() - start_time
        return total_loss, time_elapsed

    def train_single(self, N, trainset, validating=False):
        """A helper function to train on a random batch."""
        batch = random.choice(trainset)
        out, hid, tgt, sel_out, sel_tgt = Engine.forward(self.model, batch, volatile=validating)
        if validating:
            loss = None
        else:
            loss = self.crit(out.view(-1, N), tgt) + \
                self.sel_crit(sel_out, sel_tgt) * self.model.args.sel_weight
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)
            self.opt.step()
        # print('-------------------------')
        # print('train_single', sel_out, sel_tgt)
        return loss, sel_out

    def valid_pass(self, N, validset, validset_stats):
        """Validation pass."""
        # put the model into the evaluation mode
        self.model.eval()

        valid_loss, select_loss = 0, 0
        for batch in validset:
            # compute forward pass
            out, hid, tgt, sel_out, sel_tgt = Engine.forward(self.model, batch, volatile=True)

            # evaluate LM and selection losses
            valid_loss += tgt.size(0) * self.crit(out.view(-1, N), tgt).data[0]
            select_loss += self.sel_crit(sel_out, sel_tgt).data[0]

        # dividing by the number of words in the input, not the tokens modeled,
        # because the latter includes padding
        return valid_loss / validset_stats['nonpadn'], select_loss / len(validset)

    def iter(self, N, epoch, lr, traindata, validdata):
        """Performs on iteration of the training.
        Runs one epoch on the training and validation datasets.
        """
        trainset, _ = traindata
        validset, validset_stats = validdata

        train_loss, train_time = self.train_pass(N, trainset)
        valid_loss, valid_select_loss = self.valid_pass(N, validset, validset_stats)

        if self.verbose:
            print('| epoch %03d | trainloss %.3f | trainppl %.3f | s/epoch %.2f | lr %0.8f' % (
                epoch, train_loss, np.exp(train_loss), train_time, lr))
            print('| epoch %03d | validloss %.3f | validppl %.3f' % (
                epoch, valid_loss, np.exp(valid_loss)))
            print('| epoch %03d | validselectloss %.3f | validselectppl %.3f' % (
                epoch, valid_select_loss, np.exp(valid_select_loss)))

        if self.args.visual:
            self.loss_plot.update('train', epoch, train_loss)
            self.loss_plot.update('valid', epoch, valid_loss)
            self.loss_plot.update('valid_select', epoch, valid_select_loss)
            self.ppl_plot.update('train', epoch, np.exp(train_loss))
            self.ppl_plot.update('valid', epoch, np.exp(valid_loss))
            self.ppl_plot.update('valid_select', epoch, np.exp(valid_select_loss))

        return train_loss, valid_loss, valid_select_loss

    def train(self, corpus):
        """Entry point."""
        N = len(corpus.word_dict)
        best_model, best_valid_select_loss = None, 1e100
        lr = self.args.lr
        last_decay_epoch = 0
        self.t = 0

        validdata = corpus.valid_dataset(self.args.bsz, device_id=self.device_id)
        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz, device_id=self.device_id)
            _, _, valid_select_loss = self.iter(N, epoch, lr, traindata, validdata)

            if valid_select_loss < best_valid_select_loss:
                best_valid_select_loss = valid_select_loss
                best_model = copy.deepcopy(self.model)

        if self.verbose:
            print('| start annealing | best validselectloss %.3f | best validselectppl %.3f' % (
                best_valid_select_loss, np.exp(best_valid_select_loss)))

        self.model = best_model
        for epoch in range(self.args.max_epoch + 1, 100):
            if epoch - last_decay_epoch >= self.args.decay_every:
                last_decay_epoch = epoch
                lr /= self.args.decay_rate
                if lr < self.args.min_lr:
                    break
                self.opt = optim.SGD(self.model.parameters(), lr=lr)

            traindata = corpus.train_dataset(self.args.bsz, device_id=self.device_id)
            train_loss, valid_loss, valid_select_loss = self.iter(
                N, epoch, lr, traindata, validdata)

        return train_loss, valid_loss, valid_select_loss
