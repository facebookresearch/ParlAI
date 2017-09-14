# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent

from .upstream.engine import Engine
from .upstream.data import Dictionary
from .upstream.models.dialog_model import DialogModel
from .upstream import domain

import copy
import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional

OUTPUT_LENGTH = 6


class AttributeDict(dict):
    """A quick-n-dirty dict-like object with elements accessible as attributes.
    Similar to argparse's Namespace class."""
    def __init__(self, dict_):
        dict.__init__(self, dict_)
        self.__dict__.update(dict_)


class DictionaryAgent(Agent):
    """Wrapper for end-to-end-negotiator's internal dictionary."""

    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('Dictionary Arguments')
        group.add_argument(
            '--dict-file',
            help='file to load dictionary from')
        group.add_argument(
            '--unk-threshold', type=int, default=20,
            help='minimum word frequency to be in dictionary')
        group.add_argument(
            '--dict-maxexs', default=100000, type=int,
            help='max number of examples to build dict on')
        return group

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = copy.deepcopy(opt)
        self.last_observation = None
        self.freqs = {}

        if shared:
            self.word_dict = shared.get('word_dict')
            self.item_dict = shared.get('item_dict')
            self.context_dict = shared.get('context_dict')
            self.freqs = shared.get('freqs')
        elif os.path.isfile(opt.get('dict_file')):
            self.load(opt.get('dict_file'))
        else:
            self.word_dict = Dictionary(init=True)
            self.item_dict = Dictionary(init=False)
            self.context_dict = Dictionary(init=False)


    def share(self):
        shared = {}
        shared['word_dict'] = self.word_dict
        shared['item_dict'] = self.item_dict
        shared['context_dict'] = self.context_dict
        shared['freqs'] = self.freqs
        return shared

    def save(self, filename, sort=True):
        """Save dictionary to file as one word per line."""
        # sort isn't used for anything, but it needs to exist as one of the
        # function's arguments since it is passed to the save function by
        # build_dict
        filename = self.opt['model_file'] if filename is None else filename
        print('Dictionary: saving dictionary to {}'.format(filename))
        with open(filename, 'w') as file_:
            dicts = [self.word_dict, self.item_dict, self.context_dict]
            lengths = [len(d) for d in dicts]
            print(*lengths, file=file_)
            for d in dicts:
                words = (d.get_word(i) for i in range(len(d)))
                for word in words:
                    print(word, file=file_)

    def load(self, filename):
        """Load dictionary from file."""
        self.word_dict = Dictionary(init=False)
        self.item_dict = Dictionary(init=False)
        self.context_dict = Dictionary(init=False)
        print('Dictionary: loading dictionary from {}'.format(filename))
        with open(filename, 'r') as file_:
            dicts = [self.word_dict, self.item_dict, self.context_dict]
            counts = map(int, file_.readline().strip().split())
            for dict_, count in zip(dicts, counts):
                for _ in range(count):
                    word = file_.readline().strip()
                    dict_.add_word(word)

    def observe(self, observation):
        lines = observation['text'].split('\n')
        assert len(lines) == 2
        assert lines[0].startswith('Given the dialogue of a negotiation')
        dialogue = lines[1]
        context, output = observation['custom'], observation['labels'][0]
        for word in context.split():
            self.context_dict.add_word(word)
        for word in output.split():
            self.item_dict.add_word(word)
        for word in dialogue.split():
            # only add a word to the dict when its frequency exceeds
            # unk_threshold
            self.freqs[word] = self.freqs.get(word, 0) + 1
            if self.freqs[word] > self.opt.get('unk_threshold'):
                self.word_dict.add_word(word)
        return observation

    def act(self):
        return {'id': 'Dictionary'}


class DealnodealAgent(Agent):
    """Agent that, given a dialogue of a negotiation between himself and an
    opponent, outputs what it thinks the agreed decision was.

    For more information, see "Deal or No Deal? End-to-End Learning for
    Negotiation Dialogues"
    `(Lewis et al. 2017) <https://arxiv.org/abs/1706.05125>`_.

    Use dealnodeal.DictionaryAgent as the dictionary agent. Furthermore, use
    SelectionTeacher to train or validate the model with, e.g.

        $ examples/train_model.py \
            --dict-class parlai.agents.dealnodeal.dealnodeal:DictionaryAgent \
            --dict-file /tmp/dict.txt \
            -m dealnodeal -t dealnodeal:selection -bs 16 -mf '/tmp/model'
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent.
        Mostly copied from end-to-end-negotiator/src/train.py.
        """
        DictionaryAgent.add_cmdline_args(argparser)
        group = argparser.add_argument_group('Deal-no-deal Arguments')
        group.add_argument('--nembed_word', type=int, default=256,
            help='size of word embeddings')
        group.add_argument('--nembed_ctx', type=int, default=64,
            help='size of context embeddings')
        group.add_argument('--nhid_lang', type=int, default=256,
            help='size of the hidden state for the language module')
        group.add_argument('--nhid_ctx', type=int, default=64,
            help='size of the hidden state for the context module')
        group.add_argument('--nhid_strat', type=int, default=64,
            help='size of the hidden state for the strategy module')
        group.add_argument('--nhid_attn', type=int, default=64,
            help='size of the hidden state for the attention module')
        group.add_argument('--nhid_sel', type=int, default=64,
            help='size of the hidden state for the selection module')
        group.add_argument('--lr', type=float, default=20.0,
            help='initial learning rate')
        group.add_argument('--min_lr', type=float, default=1e-5,
            help='min threshold for learning rate annealing')
        group.add_argument('--decay_rate', type=float,  default=9.0,
            help='decrease learning rate by this factor')
        group.add_argument('--decay_every', type=int,  default=1,
            help='decrease learning rate after decay_every epochs')
        group.add_argument('--momentum', type=float, default=0.0,
            help='momentum for sgd')
        group.add_argument('--nesterov', action='store_true', default=False,
            help='enable nesterov momentum')
        group.add_argument('--clip', type=float, default=0.2,
            help='gradient clipping')
        group.add_argument('--dropout', type=float, default=0.5,
            help='dropout rate in embedding layer')
        group.add_argument('--init_range', type=float, default=0.1,
            help='initialization range')
        group.add_argument('--temperature', type=float, default=0.1,
            help='temperature')
        group.add_argument('--sel_weight', type=float, default=1.0,
            help='selection weight')
        group.add_argument('--domain', type=str, default='object_division',
            help='domain for the dialogue')
        group.add_argument('--rnn_ctx_encoder', action='store_true', default=False,
            help='wheather to use RNN for encoding the context')
        return group

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            self.dicts = DictionaryAgent(opt)

            # the "visual" argument is not being used, but is being checked by
            # end-to-end-negotiator's internals.
            opt['visual'] = False

            # opt is given as a dict, while end-to-end-negotiator's internals
            # access the arguments as attributes.
            args = AttributeDict(opt)

            # skip GPU support for now (will add later)
            self.device_id = None

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, self.model = self.load(opt['model_file'])
                # override options with stored ones
                opt = self.opt.update(new_opt)
            else:
                self.model = DialogModel(self.dicts.word_dict,
                                         self.dicts.item_dict,
                                         self.dicts.context_dict,
                                         OUTPUT_LENGTH, args, self.device_id)
            self.engine = Engine(self.model, args,
                                 self.device_id, verbose=True)
            self.domain = domain.get_domain(args.domain)
        self.reset()

    def reset(self):
        self.last_observation = None

    def observe(self, observation):
        self.last_observation = observation
        return observation

    def batch_act(self, observations):
        batch_reply = [{'id': self.getID()} for _ in range(len(observations))]
        valid_inds = [i for i, ex in enumerate(observations) if 'text' in ex]

        batch = []
        for i in valid_inds:
            observation = observations[i]
            lines = observation['text'].split('\n')
            assert len(lines) == 2
            assert lines[0].startswith('Given the dialogue of a negotiation')
            dialogue = lines[1].split()
            context = observation['custom'].split()
            if 'labels' in observation:
                output = observation['labels'][0].split()
            else:
                output = []
            batch.append((context, dialogue, output))

        if len(batch) > 0:
            model_replies = self.train(batch)
            for i, reply in enumerate(model_replies):
                ind = valid_inds[i]
                batch_reply[ind]['text'] = reply

        return batch_reply

    def _choose(self, logits, context, sample=False):
        """Choose the most probable set of items that represent a valid
        selection in the given context.

        Copied from end-to-end-negotiator/src/agent.py.
        """
        # get all the possible choices
        choices = self.domain.generate_choices(context)

        # construct probability distribution over only the valid choices
        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.from_numpy(np.array(idxs)))
            idxs = self.model.to_device(idxs)
            choices_logits.append(torch.gather(logits[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=False)
        # subtract the max to softmax more stable
        choice_logit = choice_logit.sub(choice_logit.max().data[0])
        prob = functional.softmax(choice_logit)
        if sample:
            # sample a choice
            idx = prob.multinomial().detach()
            logprob = functional.log_softmax(choice_logit).gather(0, idx)
        else:
            # take the most probably choice
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.data[0]]

        # Pick only your choice
        return choices[idx.data[0]][:self.domain.selection_length()], logprob, p_agree.data[0]

    def train(self, batch):
        # sort by dialogue word count
        batch.sort(key=lambda x: len(x[1]))

        # index of pad token
        pad = self.dicts.word_dict.get_idx('<pad>')

        # the longest dialogue in the batch
        max_len = len(batch[-1][1])

        inputs, words, items = [], [], []
        for input_tokens, dialogue_tokens, output_tokens in batch:
            # look up tokens
            input_idxs = self.dicts.context_dict.w2i(input_tokens)
            word_idxs = self.dicts.word_dict.w2i(dialogue_tokens)
            item_idxs = self.dicts.item_dict.w2i(output_tokens)

            # pad dialogues to the same length
            word_idxs.extend([pad] * (max_len - len(word_idxs)))

            inputs.append(input_idxs)
            words.append(word_idxs)
            items.append(item_idxs)

        ## construct tensors (copied from end-to-end-negotiator/src/data.py)
        # construct tensor for context
        ctx = torch.LongTensor(inputs).transpose(0, 1).contiguous()
        data = torch.LongTensor(words).transpose(0, 1).contiguous()

        if self.device_id is not None:
            ctx = ctx.cuda(self.device_id)
            data = data.cuda(self.device_id)
            sel_tgt = sel_tgt.cuda(self.device_id)

        # construct tensor for input and target
        inpt = data.narrow(0, 0, data.size(0) - 1)
        tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

        validating = (len(batch[0][2]) == 0)
        if validating:
            self.model.eval()
            sel_tgt = None
        else:
            self.model.train()
            # construct tensor for selection target
            sel_tgt = torch.LongTensor(items).transpose(0, 1).contiguous().view(-1)

        batch_tensors = (ctx, inpt, tgt, sel_tgt)
        N = len(self.dicts.word_dict)
        _, sel_out = self.engine.train_single(N, [batch_tensors], validating)

        ## generate output
        sel_out_episodes = sel_out.view(len(batch), -1, sel_out.size(1))
        replies = []
        for i, episode in enumerate(sel_out_episodes):
            context = batch[i][0]
            choice, _, _ = self._choose(episode, context)
            replies.append(' '.join(choice))
        return replies

    def act(self):
        return self.batch_act([self.last_observation])[0]

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        model = {}
        model['model'] = self.model
        model['opt'] = self.opt

        with open(path, 'wb') as write:
            torch.save(model, write)

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()

    def load(self, path):
        """Return opt and model states."""
        with open(path, 'rb') as read:
            model = torch.load(read)
        return model['opt'], model['model']
