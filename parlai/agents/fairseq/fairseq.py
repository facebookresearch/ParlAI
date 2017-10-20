# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

from fairseq import models
from fairseq.models import fconv
from fairseq.multiprocessing_trainer import MultiprocessingTrainer
from fairseq import criterions
from fairseq import dictionary
from fairseq.sequence_generator import SequenceGenerator

from torch.autograd import Variable
import torch
import random
import argparse
import os


def OptWrapper(opt):
    args = argparse.Namespace()
    for key in opt:
        if opt[key] is not None:
            setattr(args, key, opt[key])
    args.model = models.arch_model_map[args.arch]
    getattr(models, args.model).parse_arch(args)
    return args


def _make_fairseq_dict(parlai_dict):
    fairseq_dict = dictionary.Dictionary()
    for i in range(len(parlai_dict)):
        fairseq_dict.add_symbol(parlai_dict[i])
    return fairseq_dict


class FairseqAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    For more information, see Convolutional Sequence to Sequence Learning
     `(Gehring et al. 2017) <https://arxiv.org/abs/1705.03122>`_.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Fairseq Arguments')

        agent.add_argument(
            '--max-positions',
            default=1024,
            type=int,
            metavar='N',
            help='max number of tokens in the sequence')
        agent.add_argument(
            '--seed',
            default=1,
            type=int,
            metavar='N',
            help='pseudo random number generator seed')
        agent.add_argument(
            '--lr',
            '--learning-rate',
            default=0.25,
            type=float,
            metavar='LR',
            help='initial learning rate')
        agent.add_argument(
            '--momentum',
            default=0.99,
            type=float,
            metavar='M',
            help='momentum factor')
        agent.add_argument(
            '--weight-decay',
            '--wd',
            default=0.0,
            type=float,
            metavar='WD',
            help='weight decay')
        agent.add_argument(
            '--force-anneal',
            '--fa',
            default=0,
            type=int,
            metavar='N',
            help='force annealing at specified epoch')
        agent.add_argument(
            '--beam', default=5, type=int, metavar='N', help='beam size')
        agent.add_argument(
            '--no-early-stop',
            action='store_true',
            help=('continue searching even after finalizing k=beam '
                  'hypotheses; this is more correct, but increases '
                  'generation time by 50%%'))
        agent.add_argument(
            '--unnormalized',
            action='store_true',
            help='compare unnormalized hypothesis scores')

        agent.add_argument(
            '--lenpen',
            default=1,
            type=float,
            help=
            'length penalty: <1.0 favors shorter, >1.0 favors longer sentences')

        agent.add_argument(
            '--clip-norm',
            default=25,
            type=float,
            metavar='NORM',
            help='clip threshold of gradients')

        agent.add_argument(
            '--arch',
            '-a',
            default='fconv',
            metavar='ARCH',
            choices=models.arch_model_map.keys(),
            help='model architecture ({})'.format(
                ', '.join(models.arch_model_map.keys())))
        agent.add_argument(
            '--encoder-embed-dim',
            type=int,
            metavar='N',
            help='encoder embedding dimension')
        agent.add_argument(
            '--encoder-layers',
            type=str,
            metavar='EXPR',
            help='encoder layers [(dim, kernel_size), ...]')
        agent.add_argument(
            '--decoder-embed-dim',
            type=int,
            metavar='N',
            help='decoder embedding dimension')
        agent.add_argument(
            '--decoder-layers',
            type=str,
            metavar='EXPR',
            help='decoder layers [(dim, kernel_size), ...]')
        agent.add_argument(
            '--decoder-out-embed-dim',
            type=int,
            metavar='N',
            help='decoder output embedding dimension')
        agent.add_argument(
            '--decoder-attention',
            type=str,
            metavar='EXPR',
            help='decoder attention [True, ...]')

        # These arguments have default values independent of the model:
        agent.add_argument(
            '--dropout',
            default=0.1,
            type=float,
            metavar='D',
            help='dropout probability')
        agent.add_argument(
            '--label-smoothing',
            default=0,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing')

    def __init__(self, opt, shared=None):
        # initialize defaults first
        super().__init__(opt, shared)
        if not shared:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' +
                      opt['model_file'])
                new_opt, self.saved_state = self.load(opt['model_file'])
                # override options with stored ones
                opt = self._override_opt(new_opt)

            self.args = OptWrapper(opt)
            self.fairseq_dict = _make_fairseq_dict(DictionaryAgent(opt))
            self.id = 'Fairseq'

            self.EOS = self.fairseq_dict[self.fairseq_dict.eos()]
            self.NULL_IDX = self.fairseq_dict.pad()

            encoder = fconv.Encoder(
                len(self.fairseq_dict),
                embed_dim=self.args.encoder_embed_dim,
                convolutions=eval(self.args.encoder_layers),
                dropout=self.args.dropout,
                padding_idx=self.NULL_IDX,
                max_positions=self.args.max_positions)
            decoder = fconv.Decoder(
                len(self.fairseq_dict),
                embed_dim=self.args.decoder_embed_dim,
                convolutions=eval(self.args.decoder_layers),
                out_embed_dim=self.args.decoder_out_embed_dim,
                attention=eval(self.args.decoder_attention),
                dropout=self.args.dropout,
                padding_idx=self.NULL_IDX,
                max_positions=self.args.max_positions)
            self.model = fconv.FConvModel(encoder, decoder, self.NULL_IDX)

            # from fairseq's build_criterion()
            if self.args.label_smoothing > 0:
                self.criterion = criterions.LabelSmoothedCrossEntropyCriterion(
                    self.args.label_smoothing, self.NULL_IDX)
            else:
                self.criterion = criterions.CrossEntropyCriterion(
                    self.NULL_IDX)

            self.trainer = MultiprocessingTrainer(self.args, self.model)
            if hasattr(self, 'saved_state'):
                self.set_states(self.saved_state)

        self.reset()

    def _override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {
            'arch',
            'encoder-embed-dim',
            'encoder-layers',
            'decoder-embed-dim',
            'decoder-layers',
            'decoder-out-embed-dim',
            'decoder-attention',
        }

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

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def observe(self, observation):
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, valid_inds = self.batchify(observations)

        if len(xs) == 0:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions if testing; otherwise, train

        if ys is None:
            predictions = self._generate(self.args, xs)
            for i in range(len(predictions)):
                # map the predictions back to non-empty examples in the batch
                batch_reply[valid_inds[i]]['text'] = predictions[i]
        else:
            self._train(xs, ys)

        return batch_reply

    def parse(self, string):
        return [self.fairseq_dict.index(word) for word in string.split(' ')]

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        # valid examples
        exs = [ex for ex in observations if 'text' in ex]
        # the indices of the valid (non-empty) tensors
        valid_inds = [i for i, ex in enumerate(observations) if 'text' in ex]

        # set up the input tensors
        batchsize = len(exs)
        # tokenize the text
        parsed = [self.parse(ex['text']) for ex in exs]
        max_x_len = max([len(x) for x in parsed])
        xs = torch.LongTensor(batchsize,
                              max_x_len).fill_(self.fairseq_dict.pad())
        # pack the data to the right side of the tensor for this model
        for i, x in enumerate(parsed):
            offset = max_x_len - len(x)
            for j, idx in enumerate(x):
                xs[i][j + offset] = idx
        xs = xs.cuda(async=True)
        # set up the target tensors
        ys = None
        if 'labels' in exs[0]:
            # randomly select one of the labels to update on, if multiple
            # append EOS to each label
            labels = [
                random.choice(ex['labels']) + ' ' + self.EOS for ex in exs
            ]
            parsed = [self.parse(y) for y in labels]
            max_y_len = max(len(y) for y in parsed)
            ys = torch.LongTensor(batchsize,
                                  max_y_len).fill_(self.fairseq_dict.pad())
            for i, y in enumerate(parsed):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            ys = ys.cuda(async=True)
        return xs, ys, valid_inds

    def _positions_for_tokens(self, tokens):
        start = self.fairseq_dict.pad() + 1
        size = tokens.size()
        positions = torch.LongTensor(size).fill_(self.fairseq_dict.pad())
        for i in range(size[0]):
            nonpad = 0
            for j in range(size[1]):
                if (tokens[i][j] != self.fairseq_dict.pad()):
                    positions[i][j] = start + nonpad
                    nonpad += 1
        positions = positions.cuda(async=True)
        return positions

    def _right_shifted_ys(self, ys):
        result = torch.LongTensor(ys.size())
        result[:, 0] = self.fairseq_dict.index(self.EOS)
        result[:, 1:] = ys[:, :-1]
        return result

    def _generate(self, opt, src_tokens):
        translator = SequenceGenerator(
            [self.trainer.get_model()],
            self.fairseq_dict,
            beam_size=opt.beam,
            stop_early=(not opt.no_early_stop),
            normalize_scores=(not opt.unnormalized),
            len_penalty=opt.lenpen)
        translator.cuda()
        tokens = src_tokens
        translations = translator.generate(
            Variable(tokens), Variable(self._positions_for_tokens(tokens)))
        results = [t[0] for t in translations]
        output_lines = [[] for _ in range(len(results))]
        for i in range(len(results)):
            output_lines[i] = ' '.join(self.fairseq_dict[idx]
                                       for idx in results[i]['tokens'][:-1])
        return output_lines

    def _train(self, xs, ys=None):
        """Produce a prediction from our model. Update the model using the
        targets if available.
        """
        if ys is not None:
            sample = {
                'src_tokens': xs,
                'input_tokens': self._right_shifted_ys(ys),
                'target': ys,
                'id': None
            }
            sample['ntokens'] = sum(len(t) for t in sample['target'])
            sample['src_positions'] = self._positions_for_tokens(
                sample['src_tokens'])
            sample['input_positions'] = self._positions_for_tokens(
                sample['input_tokens'])
            self.trainer.train_step([sample], self.criterion)

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path
        if path and hasattr(self, 'trainer'):
            model = {}
            model['state_dict'] = self.trainer.get_model().state_dict()
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
        return model['opt'], model['state_dict']

    def set_states(self, state_dict):
        """Set the state dict of the model from saved states."""
        self.trainer.get_model().load_state_dict(state_dict)
