# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.dict import DictionaryAgent

try:
    from fairseq import models
except ImportError:
    raise RuntimeError(
        "Please run \"pip install 'git+https://github.com/pytorch/"
        "fairseq.git@v0.5.0#egg=fairseq'\""
    )
from fairseq import trainer
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.sequence_generator import SequenceGenerator
from fairseq import options
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.utils import convert_padding_direction

from parlai.core.torch_agent import TorchAgent

import argparse
import torch
import os
import numpy as np


# these are the fairseq metrics we want to pass up along through ParlAI
# Key is FairSeq metric, value is ParlAI name
METRIC_MAPPER = {
    # train loss
    "train_loss": "loss",
    "valid_loss": "loss",
    # updates per second
    "ups": "ups",
    # words per second
    "wps": "wps",
}


def _fairseq_opt_wrapper(opt):
    """
    Marshalls from a dict to a argparse.Namespace object for API compatibility.
    Also does some necessary post-processing needed for fairseq-py.

    :param opt: dict. ParlAI options passed around from everywhere.
    :return: an argparse.Namespace object for use in fairseq-py.
    """
    args = argparse.Namespace()

    # set default options for the given opt
    models.ARCH_CONFIG_REGISTRY[opt["arch"]](args)

    # post processing of args. See
    # https://github.com/pytorch/fairseq/blob/v0.5.0/fairseq/options.py#L95
    if "lr" in opt:
        opt["lr"] = options.eval_str_list(opt["lr"], type=float)
    if "update_freq" in opt:
        opt["update_freq"] = options.eval_str_list(opt["update_freq"], int)
    if opt.get("max_sentences_valid") is not None:
        opt["max_sentences_valid"] = opt["max_sentences"]

    # hardcode turn off distributed training. May need to revisit this later
    opt["distributed_world_size"] = 1

    for key in opt:
        if opt[key] is None:
            continue
        setattr(args, key, opt[key])

    return args


class _FairseqDictionary(DictionaryAgent):
    """Skeleton dictionary class needed for interaction with fairseq-py"""

    def pad(self):
        return self.pad_index

    def eos(self):
        return self[self.end_token]

    def unk(self):
        return self[self.unk_token]

    @property
    def pad_index(self):
        return self[self.null_token]

    @property
    def eos_index(self):
        return self[self.end_token]

    @property
    def unk_index(self):
        return self[self.unk_token]

    def add_symbol(self):
        raise NotImplementedError("This is a fake class")

    @property
    def symbols(self):
        return self.tok2ind.keys()


class _ParlaiTask(FairseqTask):
    """Skeleton task class needed for interaction with fairseq-py."""

    def __init__(self, dictionary):
        self.dict = dictionary

    @property
    def target_dictionary(self):
        return self.dict

    @property
    def source_dictionary(self):
        return self.dict


class FairseqAgent(TorchAgent):
    """Generic wrapper around fairseq for use in ParlAI"""

    metrics = {}

    # TODO: merge with TorchAgent.add_cmdline_args
    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        # first we need to add the general torch agent operations
        TorchAgent.add_cmdline_args(argparser)

        agent = argparser.add_argument_group('Fairseq Arguments')
        # TODO maybe drop these?
        agent.add_argument(
            '--max-source-positions',
            default=1024,
            type=int,
            metavar='N',
            help='max number of tokens in the sequence'
        )
        agent.add_argument(
            '--max-target-positions',
            default=1024,
            type=int,
            metavar='N',
            help='max number of tokens in the sequence'
        )
        agent.add_argument(
            '--seed',
            default=1,
            type=int,
            metavar='N',
            help='pseudo random number generator seed'
        )

        # TODO: is this necessary?
        _FairseqDictionary.add_cmdline_args(argparser)

        # TODO: opt suboptions isn't set up yet
        options.add_optimization_args(argparser)
        # TODO: check for generation sub-args
        options.add_generation_args(argparser)
        options.add_model_args(argparser)

        # We need to find out the fairseq model-specific options, so grab the
        # architecture stuff and look up its options
        known_args = argparser.parse_known_args(nohelp=True)[0]
        if hasattr(known_args, "arch"):
            arch = known_args.arch
            arch_group = argparser.add_argument_group(
                "{} specific Arguments".format(arch)
            )
            models.ARCH_MODEL_REGISTRY[arch].add_args(arch_group)

    def __init__(self, opt, shared=None):
        # In general use a basic TorchAgent wherever possible
        super().__init__(opt, shared)
        if not shared:
            # TODO: should this block come from TorchAgent?
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

            # Begin real fairseq stuff
            # copy over all the args we can
            self.args = _fairseq_opt_wrapper(opt)
            # Just some identifying info
            self.id = "fairseq:{}".format(self.args.arch)
            # construct dictionaries for parlai frontend and fairseq backend
            self.dict = _FairseqDictionary(opt)

            # We need a placeholder task for fairseq
            self.task = _ParlaiTask(self.dict)

            # actually construct the model and generator
            model_class = models.ARCH_MODEL_REGISTRY[self.args.arch]
            self.model = model_class.build_model(self.args, self.task)
            self.generator = SequenceGenerator(
                [self.model],
                tgt_dict=self.dict,
                beam_size=self.args.beam,
                stop_early=(not self.args.no_early_stop),
                normalize_scores=(not self.args.unnormalized),
                len_penalty=self.args.lenpen,
            )
            # set up the grader and the trainer
            # TODO: maybe support label smoothing here
            self.criterion = CrossEntropyCriterion(self.args, self.task)
            self.trainer = trainer.Trainer(
                self.args, self.task, self.model, self.criterion
            )

            # move things to the GPU if possible
            if self.use_cuda:
                self.model = self.model.cuda()
                self.generator = self.generator.cuda()

        # Start things off clean
        self.reset()

    # TODO: verify this is needed
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
        super().reset()
        self.reset_metrics()

    def batch_act(self, observations):
        bsz = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{"id": self.getID()} for _ in range(bsz)]

        # torchagent boilerplate
        # TODO: is this really the right way to check is_training?
        self.is_training = any(["labels" in obs for obs in observations])
        vec_obs = [self.vectorize(obs) for obs in observations]
        xs, _, ys, _, valid_inds = self.map_valid(vec_obs)
        if xs is None:
            return batch_reply

        # here begins fairseq specific stuff
        samples = self._make_sample(xs, ys)

        if self.is_training:
            self.model.train()
            self.trainer.train_step(samples)
        else:
            # grade the evaluation label
            self.model.eval()
            self.trainer.valid_step(samples)

            # Grade each of the candidate sequences
            # TODO: grade everything in observations[i]['label_candidates']

            # Next generate freely to create our response
            for i, response in zip(valid_inds, self._generate(samples)):
                batch_reply[i]["text"] = response

        return batch_reply

    def _generate(self, samples):
        src_tokens = samples["net_input"]["src_tokens"]
        src_lengths = samples["net_input"]["src_lengths"]
        gens = self.generator.generate(src_tokens, src_lengths, maxlen=64)
        responses = []
        for i in range(len(src_tokens)):
            beams = gens[i]
            selected = max(beams, key=lambda x: x["score"])
            # TODO: we can get the attention here actually :)
            response = []
            for t in selected["tokens"]:
                t = t.item()
                if t == self.dict.eos:
                    break
                response.append(self.dict[t])
            responses.append(" ".join(response))
        return responses

    def report(self):
        return {k: v.avg for k, v in self.trainer.meters.items()}

    # TODO: document
    # TODO: make sure we're only passing along some metrics
    # TODO: put in PPL
    def reset_metrics(self):
        if not hasattr(self, "trainer"):
            # We haven't initialized the trainer yet, so we don't have any metrics
            return
        for k in self.trainer.meters:
            self.trainer.meters[k].reset()

    # Helper functions
    def _seq_length(self, xs):
        """Computes length of the sequence (non-padded size)"""
        return xs.ne(self.dict.pad_index).long().sum(dim=-1)

    def _right_shifted_ys(self, ys):
        """Replaces first token with EOS and shifts the remaining tokens right one."""
        result = torch.LongTensor(ys.size())
        result[:, 0] = self.dict.eos_index
        result[:, 1:] = ys[:, :-1]
        return result

    def _make_sample(self, xs, ys):
        """Generates a sample object that Fairseq expects."""
        # add extra info to samples
        # TODO: should the right/left padding thing be in torch agent?
        repadded = convert_padding_direction(xs, self.dict.pad(), right_to_left=True)
        sample = {"target": ys, "ntokens": sum(self._seq_length(ys)).item()}
        sample["net_input"] = {
            "src_tokens": repadded,
            "src_lengths": self._seq_length(xs),
            "prev_output_tokens": self._right_shifted_ys(ys),
        }
        return sample
