# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import maintain_dialog_history
from .modules import Seq2seq

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

from collections import deque

import os
import random


class Seq2seqAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    several flavors of RNN. It then uses a linear layer (whose weights can
    be shared with the embedding layer) to convert RNN output states into
    output tokens. This model currently uses greedy decoding, selecting the
    highest probability token at each time step.

    For more information, see Sequence to Sequence Learning with Neural
    Networks `(Sutskever et al. 2014) <https://arxiv.org/abs/1409.3215>`_.
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
        Seq2seqAgent.dictionary_class().add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.005,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'dot', 'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                'For more details see: '
                                'https://arxiv.org/pdf/1508.04025.pdf')
        agent.add_argument('-attl', '--attention-length', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-rc', '--rank-candidates', type='bool',
                           default=False,
                           help='rank candidates if available. this is done by'
                                ' computing the mean score per token for each '
                                'candidate and selecting the highest scoring.')
        agent.add_argument('-tr', '--truncate', type=int, default=-1,
                           help='truncate input & output lengths to speed up '
                           'training (may reduce accuracy). This fixes all '
                           'input and output to have a maximum length. This '
                           'reduces the total amount '
                           'of padding in the batches.')
        agent.add_argument('-rnn', '--rnn-class', default='lstm',
                           choices=Seq2seq.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'],
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights. '
                                'Note that shared disabled some encoder '
                                'options--in particular, bidirectionality.')
        agent.add_argument('-lt', '--lookuptable', default='all',
                           choices=['unique', 'enc_dec', 'dec_out', 'all'],
                           help='The encoder, decoder, and output modules can '
                                'share weights, or not. '
                                'Unique has independent embeddings for each. '
                                'Enc_dec shares the embedding for the encoder '
                                'and decoder. '
                                'Dec_out shares decoder embedding and output '
                                'weights. '
                                'All shares all three weights.')
        agent.add_argument('-opt', '--optimizer', default='adam',
                           choices=Seq2seqAgent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        agent.add_argument('-emb', '--embedding-type', default='random',
                           choices=['random'],
                           help='Choose between different strategies '
                                'for word embeddings. Default is random, '
                                'but can also preinitialize from Glove.'
                                'Preinitialized embeddings can also be fixed '
                                'so they are not updated during training. '
                                'NOTE: glove init currently disabled.')
        agent.add_argument('-lm', '--language-model', default='none',
                           choices=['none', 'only', 'both'],
                           help='Enabled language modeling training on the '
                                'concatenated input and label data.')
        agent.add_argument('-hist', '--history-length', default=100000, type=int,
                           help='Number of past tokens to remember. '
                                'Default remembers 100000 tokens.')
        agent.add_argument('-histr', '--history-replies',
                           default='none', type=str,
                           choices=['none', 'model', 'label'],
                           help='Keep replies in the history, or not.')

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = self.opt  # there is a deepcopy in the init

        # all instances may need some params
        self.truncate = opt['truncate'] if opt['truncate'] > 0 else None
        self.history = {}

        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()


        if shared:
            # set up shared properties
            self.dict = shared['dict']
            self.START_IDX = shared['START_IDX']
            self.END_IDX = shared['END_IDX']
            self.NULL_IDX = shared['NULL_IDX']
            # answers contains a batch_size list of the last answer produced
            self.answers = shared['answers']

            if 'model' in shared:
                # model is shared during hogwild
                self.model = shared['model']
                self.states = shared['states']
        else:
            # this is not a shared instance of this class, so do full init
            # answers contains a batch_size list of the last answer produced
            self.answers = [None] * opt['batchsize']

            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            self.states = {}
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, states = self.load(opt['model_file'])
                # override model-specific options with stored ones
                opt = self.override_opt(new_opt)

            if opt['dict_file'] is None and opt.get('model_file'):
                # set default dict-file if not set
                opt['dict_file'] = opt['model_file'] + '.dict'

            # load dictionary and basic tokens & vectors
            self.dict = DictionaryAgent(opt)
            self.id = 'Seq2Seq'
            # we use START markers to start our output
            self.START_IDX = self.dict[self.dict.start_token]
            # we use END markers to end our output
            self.END_IDX = self.dict[self.dict.end_token]
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict[self.dict.null_token]

            self.model = Seq2seq(opt, len(self.dict),
                                 padding_idx=self.NULL_IDX,
                                 start_idx=self.START_IDX,
                                 end_idx=self.END_IDX,
                                 longest_label=self.states.get('longest_label', 1))

            if self.states:
                # set loaded states if applicable
                self.model.load_state_dict(self.states['model'])

            if self.use_cuda:
                self.model.cuda()

        if hasattr(self, 'model'):
            # if model was built, do more setup
            self.rank = opt['rank_candidates']
            self.lm = opt['language_model']

            # set up tensors once
            self.xs = torch.LongTensor(1, 1)
            self.ys = torch.LongTensor(1, 1)
            if self.rank:
                self.cands = torch.LongTensor(1, 1, 1)

            # set up criteria
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX)

            if self.use_cuda:
                # push to cuda
                self.xs = self.xs.cuda(async=True)
                self.ys = self.ys.cuda(async=True)
                if self.rank:
                    self.cands = self.cands.cuda(async=True)
                self.criterion.cuda()

            # set up optimizer
            lr = opt['learningrate']
            optim_class = Seq2seqAgent.OPTIM_OPTS[opt['optimizer']]
            kwargs = {'lr': lr}
            if opt['optimizer'] == 'sgd':
                kwargs['momentum'] = 0.95
                kwargs['nesterov'] = True
            self.optimizer = optim_class(self.model.parameters(), **kwargs)
            if self.states:
                if self.states['optimizer_type'] != opt['optimizer']:
                    print('WARNING: not loading optim state since optim class '
                          'changed.')
                else:
                    self.optimizer.load_state_dict(self.states['optimizer'])

        self.reset()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                      'encoder', 'decoder', 'lookuptable', 'attention',
                      'attention_length'}
        for k, v in new_opt.items():
            if k not in model_args:
                # skip non-model args
                continue
            if k not in self.opt:
                print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('Overriding option [ {k}: {old} => {v}]'.format(
                      k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        return self.opt

    def parse(self, text):
        """Convert string to token indices."""
        return self.dict.txt2vec(text)

    def v2t(self, vec):
        """Convert token indices to string of tokens."""
        if type(vec) == Variable:
            vec = vec.data
        new_vec = []
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def zero_grad(self):
        """Zero out optimizer."""
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        self.optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['answers'] = self.answers
        shared['dict'] = self.dict
        shared['START_IDX'] = self.START_IDX
        shared['END_IDX'] = self.END_IDX
        shared['NULL_IDX'] = self.NULL_IDX
        if self.opt.get('numthreads', 1) > 1:
            shared['model'] = self.model
            self.model.share_memory()
            shared['states'] = self.states
        return shared

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        self.episode_done = observation['episode_done']
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        batch_idx = self.opt.get('batchindex', 0)
        if not obs.get('preprocessed', False):
            obs['text2vec'] = maintain_dialog_history(
                self.history, obs,
                reply=self.answers[batch_idx],
                historyLength=self.opt['history_length'],
                useReplies=self.opt['history_replies'],
                dict=self.dict)
        else:
            obs['text2vec'] = deque(obs['text2vec'], self.opt['history_length'])
        self.observation = obs
        self.answers[batch_idx] = None
        return obs

    def predict(self, xs, ys=None, cands=None, valid_cands=None, lm=False):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available and param is set.
        """
        is_training = ys is not None
        text_cand_inds, loss_dict = None, None
        if is_training:
            self.model.train()
            self.zero_grad()
            loss = 0
            predictions, scores, _ = self.model(xs, ys)
            for i in range(scores.size(1)):
                # sum loss per-token
                score = scores.select(1, i)
                y = ys.select(1, i)
                loss += self.criterion(score, y)
            loss.backward()
            self.update_params()
            losskey = 'loss' if not lm else 'lmloss'
            loss_dict = {losskey: loss.mul_(len(xs)).data}
        else:
            self.model.eval()
            predictions, scores, text_cand_inds = self.model(xs, ys, cands,
                                                             valid_cands)

        return predictions, text_cand_inds, loss_dict

    def batchify(self, observations, lm=False):
        """Convert a list of observations into input & target tensors."""
        def valid(obs):
            # check if this is an example our model should actually process
            return 'text2vec' in obs and len(obs['text2vec']) > 0
        try:
            # valid examples and their indices
            valid_inds, exs = zip(*[(i, ex) for i, ex in
                                    enumerate(observations) if valid(ex)])
        except ValueError:
            # zero examples to process in this batch, so zip failed to unpack
            return None, None, None, None, None, None

        # set up the input tensors
        bsz = len(exs)

        # `x` text is already tokenized and truncated
        # sort by length so we can use pack_padded
        parsed_x = [ex['text2vec'] for ex in exs]
        x_lens = [len(x) for x in parsed_x]
        ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

        exs = [exs[k] for k in ind_sorted]
        valid_inds = [valid_inds[k] for k in ind_sorted]
        parsed_x = [parsed_x[k] for k in ind_sorted]

        labels_avail = any(['labels' in ex for ex in exs])

        if lm:
            self.xs.resize_(bsz, 1)
            self.xs.fill_(self.START_IDX)
            xs = Variable(self.xs)
        else:
            max_x_len = max([len(x) for x in parsed_x])

            # TODO: move zero padding to utility function?
            parsed_x = [x if len(x) == max_x_len else
                        x + deque((self.NULL_IDX,)) * (max_x_len - len(x))
                        for x in parsed_x]
            xs = torch.LongTensor(parsed_x)
            if self.use_cuda:
                # copy to gpu
                self.xs.resize_(xs.size())
                self.xs.copy_(xs, async=True)
                xs = Variable(self.xs)
            else:
                xs = Variable(xs)

        # set up the target tensors
        ys = None
        labels = None
        if labels_avail:
            # randomly select one of the labels to update on, if multiple
            labels = [random.choice(ex.get('labels', [''])) for ex in exs]
            # parse each label and append END
            parsed_y = [deque(maxlen=self.truncate) for _ in labels]
            for dq, y in zip(parsed_y, labels):
                dq.extendleft(reversed(self.parse(y)))
            for y in parsed_y:
                y.append(self.END_IDX)
            if lm:
                for x, y in zip(parsed_x, parsed_y):
                    if y.maxlen is not None:
                        y = deque(y, maxlen=y.maxlen * 2)
                    y.extendleft(reversed(x))

            max_y_len = max(len(y) for y in parsed_y)
            parsed_y = [y if len(y) == max_y_len else
                        y + deque((self.NULL_IDX,)) * (max_y_len - len(y))
                        for y in parsed_y]
            ys = torch.LongTensor(parsed_y)
            if self.use_cuda:
                # copy to gpu
                self.ys.resize_(ys.size())
                self.ys.copy_(ys, async=True)
                ys = Variable(self.ys)
            else:
                ys = Variable(ys)

        # set up candidates
        cands = None
        valid_cands = None
        if ys is None and self.rank:
            # only do ranking when no targets available and ranking flag set
            parsed_cs = []
            valid_cands = []
            for i, v in enumerate(valid_inds):
                if 'label_candidates' in observations[v]:
                    # each candidate tuple is a pair of the parsed version and
                    # the original full string
                    cs = list(observations[v]['label_candidates'])
                    curr_dqs = [deque(maxlen=self.truncate) for _ in cs]
                    for dq, c in zip(curr_dqs, cs):
                        dq.extendleft(reversed(self.parse(c)))
                    parsed_cs.append(curr_dqs)
                    valid_cands.append((i, v, cs))
            if len(parsed_cs) > 0:
                # TODO: store lengths of cands separately, so don't have zero
                #       padding for varying number of cands per example
                # found cands, pack them into tensor
                max_c_len = max(max(len(c) for c in cs) for cs in parsed_cs)
                max_c_cnt = max(len(cs) for cs in parsed_cs)
                for cs in parsed_cs:
                    for c in cs:
                        c += [self.NULL_IDX] * (max_c_len - len(c))
                    cs += [self.NULL_IDX] * (max_c_cnt - len(cs))
                cands = torch.LongTensor(parsed_cs)
                if self.use_cuda:
                    # copy to gpu
                    self.cands.resize_(cands.size())
                    self.cands.copy_(cands, async=True)
                    cands = Variable(self.cands)
                else:
                    cands = Variable(cands)

        return xs, ys, labels, valid_inds, cands, valid_cands

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, labels, valid_inds, cands, valid_cands = self.batchify(observations)

        if xs is None:
            # no valid examples, just return empty responses
            return batch_reply

        if self.lm != 'none' and ys is not None:
            # train on lm task: given [START], predict [x y]
            # (regular task is given [x START] produce [y])
            xs, ys, _, _, _, _ = self.batchify(observations, lm=True)
            _, _, loss = self.predict(xs, ys, lm=True)
            if loss is not None:
                batch_reply[0]['metrics'] = loss

        if self.lm != 'only' or ys is None:
            # produce predictions, train on targets if availables
            predictions, text_cand_inds, loss = self.predict(xs, ys, cands, valid_cands)
            if loss is not None:
                if 'metrics' in batch_reply[0]:
                    for k, v in loss.items():
                        batch_reply[0]['metrics'][k] = v
                else:
                    batch_reply[0]['metrics'] = loss

            predictions = predictions.cpu()
            for i in range(len(predictions)):
                # map the predictions back to non-empty examples in the batch
                # we join with spaces since we produce tokens one at a time
                curr = batch_reply[valid_inds[i]]
                output_tokens = []
                for c in predictions.data[i]:
                    if c == self.END_IDX:
                        break
                    else:
                        output_tokens.append(c)
                curr_pred = self.v2t(output_tokens)
                curr['text'] = curr_pred
                if labels is not None:
                    y = []
                    for c in ys.data[i]:
                        if c == self.END_IDX:
                            break
                        else:
                            y.append(c)
                    self.answers[valid_inds[i]] = y
                else:
                    self.answers[valid_inds[i]] = output_tokens
            if labels is None and random.random() > 0.2:
                print('prediction: ', curr_pred)

            if text_cand_inds is not None:
                text_cand_inds = text_cand_inds.cpu().data
                for i in range(len(valid_cands)):
                    order = text_cand_inds[i]
                    _, batch_idx, curr_cands = valid_cands[i]
                    curr = batch_reply[batch_idx]
                    curr['text_candidates'] = [curr_cands[idx] for idx in order
                                               if idx < len(curr_cands)]

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            model['model'] = self.model.state_dict()
            model['longest_label'] = self.model.longest_label
            model['optimizer'] = self.optimizer.state_dict()
            model['optimizer_type'] = self.opt['optimizer']
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

        return model['opt'], model
