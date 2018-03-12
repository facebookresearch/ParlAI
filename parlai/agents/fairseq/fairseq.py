# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

try:
    from .fairseq_py.fairseq import models
except ImportError:
    raise RuntimeError("Please run 'python setup.py build' and "
                       "'python setup.py develop' from the fairseq_py directory")
from .fairseq_py.fairseq.models import fconv
from .fairseq_py.fairseq.multiprocessing_trainer import MultiprocessingTrainer
from .fairseq_py.fairseq import criterions
from .fairseq_py.fairseq import dictionary
from .fairseq_py.fairseq.sequence_generator import SequenceGenerator
from .fairseq_py.fairseq import options

from torch.autograd import Variable

from collections import deque
import argparse
import numpy as np
import random
import torch
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
            '-tr', '--truncate',
            type=int, default=-1,
            help='truncate input & output lengths to speed up training (may '
                 'reduce accuracy). This fixes all input and output to have a '
                 'maximum length. This reduces the total amount of padding in '
                 'the batches.')
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
        options.add_optimization_args(argparser)
        options.add_generation_args(argparser)
        options.add_model_args(argparser)

    def __init__(self, opt, shared=None):
        # initialize defaults first
        super().__init__(opt, shared)
        if not shared:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.
            saved_state = None
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' +
                      opt['model_file'])
                new_opt, saved_state = self.load(opt['model_file'])
                # override options with stored ones
                opt = self._override_opt(new_opt)

            self.args = OptWrapper(opt)
            self.parlai_dict = DictionaryAgent(opt)
            self.fairseq_dict = _make_fairseq_dict(self.parlai_dict)
            self.id = 'Fairseq'
            self.truncate = opt['truncate'] if opt['truncate'] > 0 else None

            self.EOS = self.fairseq_dict[self.fairseq_dict.eos()]
            self.EOS_TENSOR = (torch.LongTensor(1, 1)
                               .fill_(self.fairseq_dict.eos()))
            self.NULL_IDX = self.fairseq_dict.pad()

            encoder = fconv.FConvEncoder(
                self.fairseq_dict,
                embed_dim=self.args.encoder_embed_dim,
                convolutions=eval(self.args.encoder_layers),
                dropout=self.args.dropout,
                max_positions=self.args.max_positions)
            decoder = fconv.FConvDecoder(
                self.fairseq_dict,
                embed_dim=self.args.decoder_embed_dim,
                convolutions=eval(self.args.decoder_layers),
                out_embed_dim=self.args.decoder_out_embed_dim,
                attention=eval(self.args.decoder_attention),
                dropout=self.args.dropout,
                max_positions=self.args.max_positions)
            self.model = fconv.FConvModel(encoder, decoder)

            # from fairseq's build_criterion()
            if self.args.label_smoothing > 0:
                self.criterion = criterions.LabelSmoothedCrossEntropyCriterion(
                    self.args.label_smoothing, self.NULL_IDX)
            else:
                self.criterion = criterions.CrossEntropyCriterion(
                    self.args, self.fairseq_dict)

            self.trainer = MultiprocessingTrainer(self.args, self.model, self.criterion)
            if saved_state is not None:
                self.set_states(saved_state)
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
        if not self.episode_done and not observation.get('preprocessed', False):
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
        bsz = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(bsz)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field

        # also, split observations into sub-batches based on number of gpus
        obs_split = np.array_split(observations, self.trainer.num_replicas)
        samples = [self.batchify(obs) for obs in obs_split]
        samples = [s for s in samples if s[0] is not None]
        any_valid = any(len(s[0]) > 0 for s in samples)

        if not any_valid:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions if testing; otherwise, train
        has_targets = any(s[1] is not None for s in samples)
        if not has_targets:
            offset = 0
            for s in samples:
                xs = s[0]
                valid_inds = s[2]

                predictions = self._generate(self.args, xs)
                for i in range(len(predictions)):
                    # map the predictions back to non-empty examples in the batch
                    batch_reply[valid_inds[i] + offset]['text'] = predictions[i]
                    if i == 0:
                        print('prediction:', predictions[i])
                offset += len(valid_inds)
        else:
            loss = self._train(samples)

            batch_reply[0]['metrics'] = {}
            for k, v in loss.items():
                batch_reply[0]['metrics'][k] = v * bsz

        return batch_reply

    def parse(self, string):
        return [self.fairseq_dict.index(word)
                for word in self.parlai_dict.tokenize(string)]

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        # valid examples
        exs = [ex for ex in observations if 'text' in ex]
        # the indices of the valid (non-empty) tensors
        valid_inds = [i for i, ex in enumerate(observations) if 'text' in ex]

        # set up the input tensors
        batchsize = len(exs)
        if batchsize == 0:
            return None, None, None
        # tokenize the text
        parsed_x = [deque(maxlen=self.truncate) for _ in exs]
        for dq, ex in zip(parsed_x, exs):
            dq += self.parse(ex['text'])
        # parsed = [self.parse(ex['text']) for ex in exs]
        max_x_len = max((len(x) for x in parsed_x))
        for x in parsed_x:
            # left pad with zeros
            x.extendleft([self.fairseq_dict.pad()] * (max_x_len - len(x)))
        xs = torch.LongTensor(parsed_x)

        # set up the target tensors
        ys = None
        if 'labels' in exs[0]:
            # randomly select one of the labels to update on, if multiple
            labels = [random.choice(ex.get('labels', [''])) for ex in exs]
            parsed_y = [deque(maxlen=self.truncate) for _ in labels]
            for dq, y in zip(parsed_y, labels):
                dq.extendleft(reversed(self.parse(y)))
            for y in parsed_y:
                y.append(self.fairseq_dict.eos())
            # append EOS to each label
            max_y_len = max(len(y) for y in parsed_y)
            for y in parsed_y:
                y += [self.fairseq_dict.pad()] * (max_y_len - len(y))
            ys = torch.LongTensor(parsed_y)
        return xs, ys, valid_inds

    def _positions_for_tokens(self, tokens):
        size = tokens.size()
        not_pad = tokens.ne(self.fairseq_dict.pad()).long()
        new_pos = tokens.new(size).fill_(self.fairseq_dict.pad())
        new_pos += not_pad
        for i in range(1, size[1]):
            new_pos[:, i] += new_pos[:, i-1] - 1
        return new_pos

    def _right_shifted_ys(self, ys):
        result = torch.LongTensor(ys.size())
        result[:, 0] = self.fairseq_dict.index(self.EOS)
        result[:, 1:] = ys[:, :-1]
        return result

    def _generate(self, opt, src_tokens):
        if not hasattr(self, 'translator'):
            self.translator = SequenceGenerator(
                [self.trainer.get_model()],
                beam_size=opt.beam,
                stop_early=(not opt.no_early_stop),
                normalize_scores=(not opt.unnormalized),
                len_penalty=opt.lenpen)
            self.translator.cuda()
        tokens = src_tokens.cuda(async=True)
        translations = self.translator.generate(Variable(tokens))
        results = [t[0] for t in translations]
        output_lines = [[] for _ in range(len(results))]
        for i in range(len(results)):
            output_lines[i] = ' '.join(self.fairseq_dict[idx]
                                       for idx in results[i]['tokens'][:-1])
        return output_lines

    def _train(self, samples):
        """Update the model using the targets."""
        for i, sample in enumerate(samples):
            # add extra info to samples
            sample = {
                'src_tokens': sample[0],
                'input_tokens': self._right_shifted_ys(sample[1]),
                'target': sample[1],
                'id': None
            }
            sample['ntokens'] = sum(len(t) for t in sample['target'])
            sample['src_positions'] = self._positions_for_tokens(
                sample['src_tokens'])
            sample['input_positions'] = self._positions_for_tokens(
                sample['input_tokens'])
            samples[i] = sample
        return self.trainer.train_step(samples)

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
