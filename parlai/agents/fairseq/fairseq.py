# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.dict import DictionaryAgent

try:
    from fairseq import models, optim
except ImportError:
    raise RuntimeError(
        "Please run \"pip install -U 'git+https://github.com/pytorch/"
        "fairseq.git@v0.5.0#egg=fairseq'\""
    )
from fairseq import trainer, fp16_trainer
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.sequence_generator import SequenceGenerator
from fairseq import options
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.utils import convert_padding_direction

from parlai.core.torch_agent import TorchAgent
from parlai.core.build_data import modelzoo_path

import argparse
import torch
import os
import numpy as np
import pickle


# If a model file is loaded, these arguments may NOT be overridden in the
# command line:
NON_OVERRIDABLE_ARGS = {
    'arch',
    'encoder_embed_dim',
    'encoder_layers',
    'decoder_embed_dim',
    'decoder_layers',
    'decoder_out_embed_dim',
    'decoder_attention',
}


def _fairseq_opt_wrapper(opt):
    """
    Marshalls from a dict to a argparse.Namespace object for API compatibility.
    Also does some necessary post-processing needed for fairseq-py.

    :param opt: dict. ParlAI options passed around from everywhere.
    :return: an argparse.Namespace object for use in fairseq-py.
    """
    args = argparse.Namespace()

    # first set args according to ParlAI options
    for key in opt:
        if opt[key] is not None:
            setattr(args, key, opt[key])

    # at this point the user *must* have specified an arch
    if not hasattr(args, "arch"):
        raise ValueError("--arch/-a must be specified")
    # fill in default options from the model
    models.ARCH_CONFIG_REGISTRY[args.arch](args)

    # post processing of args. See
    # https://github.com/pytorch/fairseq/blob/v0.5.0/fairseq/options.py#L95
    if hasattr(args, "lr"):
        args.lr = options.eval_str_list(args.lr, type=float)
    if hasattr(args, "update_freq"):
        args.update_freq = options.eval_str_list(args.update_freq, int)
    if hasattr(args, "max_sentences_valid"):
        args.max_sentences_valid = args.max_sentences
    if getattr(args, "truncate") == -1:
        # some torch agents use positional embeddings, which must have a max length
        setattr(args, "truncate", 1024)
    if not hasattr(args, "max_source_positions"):
        # fairseq uses a different name for this CLI parameter
        # Sometimes it's set in model defaults, but not for all models
        setattr(args, "max_source_positions", getattr(args, "truncate"))
        # if we don't have source lengths, we don't have target lengths
        setattr(args, "max_target_positions", getattr(args, "truncate"))

    # handle modelzoo if possible
    for k in ("encoder_embed_path", "decoder_embed_path"):
        if hasattr(args, k) and getattr(args, k) is not None:
            setattr(args, k, modelzoo_path(opt.get("datapath"), getattr(args, k)))

    # Here we hardcode a few options that we currently do not support
    # turn off distributed training
    args.distributed_world_size = 1
    args.distributed_rank = 0

    return args, vars(args)


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
    def bos_index(self):
        return self[self.start_token]

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
        agent.add_argument(
            '--seed',
            default=1,
            type=int,
            metavar='N',
            help='pseudo random number generator seed'
        )
        agent.add_argument(
            '--skip-generation',
            default=False,
            type=bool,
            metavar='BOOL',
            help='Skips test time beam search. Much faster if you only need PPL',
        )

        # Dictionary construction stuff. Using the subclass in case we end up
        # needing any fairseq specific things
        _FairseqDictionary.add_cmdline_args(argparser)

        # Optimization and learning rate schedule specific arguments
        options.add_optimization_args(argparser)
        known_args = argparser.parse_known_args(nohelp=True)[0]
        if hasattr(known_args, "optimizer"):
            optimizer = known_args.optimizer
            opt_group = argparser.add_argument_group(
                '{} optimizer arguments'.format(optimizer)
            )
            optim.OPTIMIZER_REGISTRY[optimizer].add_args(opt_group)
        if hasattr(known_args, "lr_scheduler"):
            lr_scheduler = known_args.lr_scheduler
            lr_group = argparser.add_argument_group(
                '{} scheduler arguments'.format(lr_scheduler)
            )
            optim.lr_scheduler.LR_SCHEDULER_REGISTRY[lr_scheduler].add_args(lr_group)

        # Generation arguments
        options.add_generation_args(argparser)

        # We need to find out the fairseq model-specific options, so grab the
        # architecture stuff and look up its options
        arch_group = options.add_model_args(argparser)
        # Fairseq marks the arch flag as required, but it may be specified
        # by a saved model cache, so we do some weird stuff to undo that
        for a in arch_group._actions:
            if a.dest == "arch":
                a.required = False
                a.default = None
                break
        known_args = argparser.parse_known_args(nohelp=True)[0]
        if hasattr(known_args, "arch") and known_args.arch is not None:
            arch = known_args.arch
            arch_group = argparser.add_argument_group(
                "{} architecture arguments".format(arch)
            )
            models.ARCH_MODEL_REGISTRY[arch].add_args(arch_group)

        # Override a few defaults from within fairseq to more sensible defaults
        argparser.set_defaults(
            clip_norm=0.1,
            adam_betas="(0.9,0.98)"
        )

    def __init__(self, opt, shared=None):
        # In general use a basic TorchAgent wherever possible
        super().__init__(opt, shared)
        if not shared:
            # this is not a shared instance of this class, so do full initialization

            # fairseq expects options to be in argparse format, instead of a dict
            # We also need to do some argument postprocessing and whatnot
            self.args, self.opt = _fairseq_opt_wrapper(opt)

            # seed the RNG
            torch.manual_seed(self.args.seed)

            # Just some identifying info
            self.id = "fairseq:{}".format(self.args.arch)

            # construct dictionaries for parlai frontend and fairseq backend
            self.dict = _FairseqDictionary(self.opt)

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

            if self.args.fp16:
                self.trainer = fp16_trainer.FP16Trainer(
                    self.args, self.task, self.model, self.criterion
                )
            else:
                # TODO: we might choose to add a --no-fp16 opt in the future to
                # explicitly disable fp16 instead
                if torch.cuda.get_device_capability(0)[0] >= 7:
                    print("Heads up: using --fp16 could be a lot faster!")
                self.trainer = trainer.Trainer(
                    self.args, self.task, self.model, self.criterion
                )

            # if the model already existed, let's preload it and the trainer
            if self.opt.get('model_file') and os.path.isfile(self.opt['model_file']):
                print('Loading existing model params from ' + self.opt['model_file'])
                self.load(self.opt.get('model_file'))

            # move things to the GPU if possible
            if self.use_cuda:
                self.model = self.model.cuda()
                self.generator = self.generator.cuda()

        # Start things off clean
        self.reset()

    def _check_opts_unchanged(self, saved_opts, current_opts):
        """Verify that critical options do not differ in command line vs saved model"""
        for k in NON_OVERRIDABLE_ARGS:
            if k not in saved_opts or k not in current_opts:
                # if it's not an option needed by this fairseq model, don't stress
                continue
            if saved_opts[k] != current_opts[k]:
                raise ValueError(
                    '{} cannot be overridden when --model-file is specified'.format(k)
                )

    def save(self, path):
        """Save using fairseq's checkpointing."""
        if not path:
            return
        self.trainer.save_checkpoint(path, {'opt': self.opt, 'epoch': 0})
        # Parlai expects options to also be saved
        with open(path + ".opt", 'wb') as handle:
            # overridden options shouldn't be stored, only the main ones
            if 'override' in self.opt:
                del self.opt['override']
            pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """Load using fairseq's checkpointing."""
        old_options = self.trainer.load_checkpoint(path)
        self._check_opts_unchanged(old_options, self.opt)

    def shutdown(self):
        if not hasattr(self, 'trainer'):
            # looks like this is a "fake" model that isn't actually used for batch_act.
            # we don't need to save this one.
            return
        super().shutdown()

    def reset(self):
        """Reset observation and episode_done."""
        super().reset()
        self.reset_metrics()

    def batch_act(self, observations):
        bsz = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{"id": self.getID()} for _ in range(bsz)]

        # torchagent boilerplate
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
            if ys is not None:
                # Interactive mode won't have a gold label
                self.trainer.valid_step(samples)

            # Grade each of the candidate sequences
            # TODO: grade everything in observations[i]['label_candidates']

            # Next generate freely to create our response
            if self.args.skip_generation:
                # skip the generation step
                for i in valid_inds:
                    batch_reply[i]["text"] = ""
            else:
                # actually do the generation
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
            response = []
            for t in selected["tokens"]:
                t = t.item()
                if t == self.dict.bos_index:
                    # don't include <s> token
                    continue
                if t == self.dict.eos_index:
                    # stop (and don't include) </s> token
                    break
                response.append(self.dict[t])
            responses.append(" ".join(response))
        return responses

    def report(self):
        # if we haven't initialized yet, just return a dummy object
        if not hasattr(self, "trainer"):
            return {}

        # These are the metrics we'll pass up the way, and their new names
        train_metrics = {"train_loss", "ups", "wps", "gnorm", "clip"}
        valid_metrics = {"valid_loss"}

        metrics = train_metrics if self.is_training else valid_metrics

        output = {k: self.trainer.meters[k].avg for k in metrics}

        # additionally output perplexity. note that fairseq models use base 2
        # in cross_entropy:
        # github.com/pytorch/fairseq/blob/master/fairseq/criterions/cross_entropy.py#L55
        if "train_loss" in output:
            output["train_ppl"] = np.exp2(output["train_loss"])
        if "valid_loss" in output:
            output["ppl"] = np.exp2(output["valid_loss"])

        return output

    def reset_metrics(self):
        if not hasattr(self, "trainer"):
            # We haven't initialized the trainer yet, so we don't have any metrics
            return
        # We need to reset everything
        for k in self.trainer.meters:
            self.trainer.meters[k].reset()

    def receive_metrics(self, metrics_dict):
        """Used to update lr scheduler."""
        self.trainer.lr_step(-1, metrics_dict["valid_loss"])

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
        sample = {}
        sample["net_input"] = {
            "src_tokens": repadded,
            "src_lengths": self._seq_length(xs),
        }
        if ys is not None:
            sample["target"] = ys
            sample["ntokens"] = sum(self._seq_length(ys)).item()
            sample["net_input"]["prev_output_tokens"] = self._right_shifted_ys(ys)
        return sample
