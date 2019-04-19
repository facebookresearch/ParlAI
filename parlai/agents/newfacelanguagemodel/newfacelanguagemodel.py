from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

from parlai.agents.language_model_emb.language_model_emb import LanguageModelEmbAgent
# from parlai.core.utils import PaddingUtils, round_sigfigs
# from parlai.core.thread_utils import SharedTable
# from .modules import RNNModel

# for initializing model
from parlai.agents.language_model.modules import RNNModel
from parlai.core.build_data import modelzoo_path
from parlai.core.distributed_utils import is_primary_worker

import torch
from torch.autograd import Variable
import torch.nn as nn

import os
import math
import numpy as np
import json

from collections import Counter


class NewfacelanguagemodelAgent(LanguageModelEmbAgent):
    """ Agent which trains an RNN on a language modeling task.
    It is adapted from the language model featured in Pytorch's examples repo
    here: <https://github.com/pytorch/examples/tree/master/word_language_model>.
    """

    #     @staticmethod
    #     def dictionary_class():
    #         return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('FACE Language Model Weighted Arguments')
        agent.add_argument(
            '-emb', '--embedding-type', default='random',
            choices=['random', 'glove', 'glove-fixed', 'glove-twitter-fixed',
                     'fasttext', 'fasttext-fixed', 'fasttext_cc',
                     'fasttext_cc-fixed'],
            help='Choose between different strategies for initializing word '
                 'embeddings. Default is random, but can also preinitialize '
                 'from Glove or Fasttext. Preinitialized embeddings can also '
                 'be fixed so they are not updated during training.')

        agent.add_argument(
            '-swap', '--swap-criterion-train-eval', type='bool', default=False,
            help='Whether to swap the criterion between training and evaluation'
                 'i.e., train with idf weighting, but eval without idf in criterion')
        agent.add_argument('-ft', '--frequency-type', default='out',
                           choices=['out', 'gt', 'none'],
                           help='What to use for calculating token frequency.')
        agent.add_argument('-wt', '--weighing-time', default='pre',
                           choices=['pre', 'post', 'none'],
                           help='When to apply weight to losses.')
        agent.add_argument('-cp', '--confidence-penalty', default='none',
                           choices=['cp', 'cpf', 'cpfw', 'cpfwn', 'none'],
                           help='Which kind of confidence penalty to use: '
                                "'cp' is the confidence-penalty function reported in https://arxiv.org/abs/1809.01941. "
                                "'cpf' is the parameter-free version proposed in https://arxiv.org/abs/1902.09191. "
                                "'cpfw' means using the parameter-free version as the weight of FACE. "
                                "'cpfwn' is a new design that normalizes the weight to the range of [1, +inf], which is "
                                "more favorable as the weight of FACE.")
        agent.add_argument('-b', '--beta', type=float, default=2.5,
                           help='Penalty strength for type "cp".')
        """Copied from language_model.py"""
        agent = argparser.add_argument_group('Language Model Arguments')
        agent.add_argument('--init-model', type=str, default=None,
                           help='load dict/features/weights/opts from this file')
        agent.add_argument('-hs', '--hiddensize', type=int, default=200,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=200,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-dr', '--dropout', type=float, default=0.2,
                           help='dropout rate')
        agent.add_argument('-clip', '--gradient-clip', type=float, default=0.25,
                           help='gradient clipping')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('-rnn', '--rnn-class', default='LSTM',
                           help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
        agent.add_argument('-sl', '--seq-len', type=int, default=35,
                           help='sequence length')
        agent.add_argument('-tied', '--emb-tied', action='store_true',
                           help='tie the word embedding and softmax weights')
        agent.add_argument('-seed', '--random-seed', type=int, default=1111,
                           help='random seed')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-tr', '--truncate-pred', type=int, default=50,
                           help='truncate predictions')
        agent.add_argument('-rf', '--report-freq', type=float, default=0.1,
                           help='report frequency of prediction during eval')
        agent.add_argument('-pt', '--person-tokens', type='bool', default=True,
                           help='append person1 and person2 tokens to text')
        # learning rate parameters
        agent.add_argument('-lr', '--learningrate', type=float, default=20,
                           help='initial learning rate')
        agent.add_argument('-lrf', '--lr-factor', type=float, default=1.0,
                           help='mutliply learning rate by this factor when the \
                           validation loss does not decrease')
        agent.add_argument('-lrp', '--lr-patience', type=int, default=10,
                           help='wait before decreasing learning rate')
        agent.add_argument('-lrm', '--lr-minimum', type=float, default=0.1,
                           help='minimum learning rate')
        agent.add_argument('-sm', '--sampling-mode', type='bool', default=False,
                           help='sample when generating tokens instead of taking \
                           the max and do not produce UNK token (when bs=1)')

        NewfacelanguagemodelAgent.dictionary_class().add_cmdline_args(argparser)

        return agent

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)

        self.id = 'FACELanguageModel'

        self.id = 'NEWFACE'
        if getattr(self, 'word_freq', None) is None:
            self.word_freq = np.zeros(len(self.dict))
        self.ft = opt['frequency_type']
        self.wt = opt['weighing_time']
        self.cp = opt['confidence_penalty']
        self.beta = opt['beta']
        self.masked_entropy = HLoss(ignore_index=self.NULL_IDX)
        self.ideal_entropy = math.log(1 / len(self.dict))


        self.START_IDX = self.dict[self.dict.start_token]

        #if not shared:
         #   self.build_criterion()

    def clean_preds(self, preds):
        res = []
        # OAD:
        #         preds = preds.cpu().tolist()
        if type(preds) == tuple:
            preds = [p.cpu().tolist() for p in preds]
        else:  # should be tensor:
            preds = preds.cpu().tolist()

        for pred in preds:
            if self.END_IDX in pred:
                ind = pred.index(self.END_IDX) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == self.START_IDX:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)

        # self.word_freq *= self.opt['decay_factor']
        for k, v in curr.items():
            self.word_freq[k] += v

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        super().reset_metrics()
        self.metrics['lmloss'] = 0.0
        self.metrics['lm_num_tokens'] = 0
        self.metrics['correct_tokens'] = 0
        self.metrics['preds'] = []

    def loss_weight(self):
        RF = self.word_freq / self.word_freq.sum()  # relative frequency
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)  # normalization
        if self.use_cuda:
            return torch.FloatTensor(weight).cuda()
        else:
            return torch.FloatTensor(weight)

    def get_target_loss(self, data, hidden, targets, is_training):

        print("GETTING TARGET LOSS")
        ''' modified from language_model.py agent'''

        """Calculates the loss with respect to the targets, token by token,
           where each output token is conditioned on either the input or the
           previous target token.
        """
        loss = 0.0
        bsz = data.size(0)

        # during interactive mode, when no targets exist, we return 0
        if targets is None:
            return loss

        # feed in inputs without end token
        output, hidden = self.model(data.transpose(0, 1), hidden)
        self.hidden = self.repackage_hidden(hidden)
        # feed in end tokens
        output, hidden = self.model(Variable(self.ends[:bsz].view(1, bsz)), self.hidden)
        self.hidden = self.repackage_hidden(hidden)
        output_flat = output.view(-1, len(self.dict))
        #import pdb; pdb.set_trace()
        scores = output #batch.text_vec
        preds = targets#batch.label_vec
        score_view = scores.view(-1, scores.size(-1))
        preds_clean = self.clean_preds(preds)

        if self.ft == 'gt':
            self.update_frequency(self.clean_preds(targets))#batch.label_vec))
        elif self.ft == 'out':
            self.update_frequency(preds_clean)
        # calculate loss w/ or w/o pre-/post-weight
        if self.wt == 'pre':
            self.criterion.weight = self.loss_weight()
            # loss = self.criterion(score_view, batch.label_vec.view(-1))
        elif self.wt == 'post':
            self.criterion.reduction = 'none'
            # loss = self.criterion(score_view, batch.label_vec.view(-1))
            device = loss.device
            freq_pred = self.word_freq[preds.view(-1).cpu().numpy()]
            freq_pred = torch.FloatTensor(freq_pred).to(device)
            freq_GT = self.word_freq[targets.view(-1).cpu().numpy()]#batch.label_vec.view(-1).cpu().numpy()]
            freq_GT = torch.FloatTensor(freq_GT).to(device)
            total_freq = self.word_freq.sum()
            weight = 1 + F.relu(freq_pred - freq_GT) / total_freq
            loss = torch.matmul(loss, weight)
        else:
            loss = self.criterion(score_view, target.view(-1)) #batch.label_vec.view(-1))



        '''loss += self.compute_criterion(output_flat, targets.select(1, 0).view(-1), is_training).data

        for i in range(1, targets.size(1)):
            output, hidden = self.model(
                targets.select(1, i - 1).view(1, bsz),
                self.hidden,
                no_pack=True
            )
            self.hidden = self.repackage_hidden(hidden)
            output_flat = output.view(-1, len(self.dict))
            loss += self.compute_criterion(output_flat, targets.select(1, i).view(-1), is_training).data'''

        return loss

    def predict(self, data, hidden, targets=None, is_training=True, y_lens=None):

        ''' modified from language_model.py agent'''

        """Produce a prediction from our model."""
        output = None
        predictions = None
        if is_training:
            self.model.train()
            self.zero_grad()
            output, hidden = self.model(data, hidden)
            scores = output  # batch.text_vec
            preds = targets  # batch.label_vec
            score_view = scores.view(-1, scores.size(-1))
            preds_clean = self.clean_preds(preds)



            if self.ft == 'gt':
                self.update_frequency(self.clean_preds(targets))  # batch.label_vec))
            elif self.ft == 'out':
                self.update_frequency(preds_clean)
            # calculate loss w/ or w/o pre-/post-weight
            if self.wt == 'pre':
                self.criterion.weight = self.loss_weight()
                loss = self.criterion(score_view, targets.view(-1))#batch.label_vec.view(-1))
            elif self.wt == 'post':
                self.criterion.reduction = 'none'
                # loss = self.criterion(score_view, batch.label_vec.view(-1))
                device = loss.device
                freq_pred = self.word_freq[preds.view(-1).cpu().numpy()]
                freq_pred = torch.FloatTensor(freq_pred).to(device)
                freq_GT = self.word_freq[targets.view(-1).cpu().numpy()]  # batch.label_vec.view(-1).cpu().numpy()]
                freq_GT = torch.FloatTensor(freq_GT).to(device)
                total_freq = self.word_freq.sum()
                weight = 1 + F.relu(freq_pred - freq_GT) / total_freq
                loss = torch.matmul(loss, weight)
            else:
                loss = self.criterion(score_view, targets.view(-1))  # batch.label_vec.view(-1))
            notnull = targets.ne(self.NULL_IDX)#batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            # Use confidence penalty or not

            if self.cp != 'none':
                entropy = self.masked_entropy(score_view, targets.view(-1))#batch.label_vec.view(-1))
                mean_entropy = entropy / target_tokens
                if self.cp == 'cp':
                    loss -= self.beta * mean_entropy
                elif self.cp == 'cpf':
                    loss += 1 / mean_entropy
                elif self.cp == 'cpfw':
                    # TODO: normalize weight to [1, ++]?
                    loss *= (1 + 1 / mean_entropy)
                elif self.cp == 'cpfwn':
                    loss *= (self.ideal_entropy / mean_entropy)
            # save loss to metrics
            correct = ((targets == preds) * notnull).sum().item()#((batch.label_vec == preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['lmloss'] += loss.item()
            self.metrics['lm_num_tokens'] += target_tokens
            self.metrics['preds'].extend(preds_clean)
            loss = loss / target_tokens
            loss.backward(retain_graph=True)
            self.update_params()
        else:
            self.model.eval()
            predictions = self.get_predictions(data)
            bsz = data.size(0)
            if bsz != self.batchsize:
                self.hidden = self.model.init_hidden(bsz)
            if targets is not None:
                loss = self.get_target_loss(data, self.hidden, targets, is_training)
                self.metrics['loss'] += loss
                self.metrics['num_tokens'] += sum(y_lens)

        return output, hidden, predictions



class HLoss(nn.Module):

    def __init__(self, ignore_index=-1):
        super(HLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, x, labels):
        mask = (labels != self.ignore_index).float()
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * torch.matmul(mask, b.sum(dim=1))
        return b

