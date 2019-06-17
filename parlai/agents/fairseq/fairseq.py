#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ParlAI has limited support for using models from
`Fairseq <https://github.com/pytorch/fairseq>`_. Fairseq often supports more
experimental seq2seq architectures with fast fp16 training.

Fairseq models can be used for many default tasks by combining a
``--arch`` flag. For example:

`python -m parlai.scripts.train -t convai2 -m fairseq -a transformer`
"""


from parlai.core.dict import DictionaryAgent
from parlai.core.utils import argsort, padded_tensor

try:
    from fairseq import models, optim, criterions
    # this is a hack around versioning check because fairseq doesn't
    # announce version numbers yet
    # fairseq 0.5.0 has fp16_trainer, 0.6.0 does not
    try:
        from fairseq import fp16_trainer  # noqa: F401
    except ImportError:
        pass
    else:
        raise ImportError
except ImportError:
    raise ImportError(
        "Please run \"pip install -U 'git+https://github.com/pytorch/"
        "fairseq.git@v0.6.0#egg=fairseq'\""
    )
from fairseq import trainer
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer
from fairseq import options
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.utils import convert_padding_direction, load_model_state
from fairseq.meters import AverageMeter

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.build_data import modelzoo_path
from parlai.core.utils import round_sigfigs

import argparse
import torch
import os
import numpy as np
import json
from collections import defaultdict


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


def _fairseq_opt_wrapper(opt, skip_pretrained_embedding_loading=False):
    """
    Marshalls from a dict to a argparse.Namespace object for API compatibility.

    Also does some necessary post-processing needed for fairseq. Optionally can
    override pretrained embedding options, which is useful if we're just loading
    a model from a checkpoint.

    :param opt: dict. ParlAI options passed around from everywhere.
    :param skip_pretrained_embedding_loading: bool. Don't preload word embeddings.
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
    if args.truncate == -1:
        # some torch agents use positional embeddings, which must have a max length
        args.truncate = 1024
    if not hasattr(args, "max_source_positions"):
        # fairseq uses a different name for this CLI parameter
        # Sometimes it's set in model defaults, but not for all models
        args.max_source_positions = args.truncate
        # if we don't have source lengths, we don't have target lengths
        args.max_target_positions = args.truncate

    # handle modelzoo if possible
    for k in ("encoder_embed_path", "decoder_embed_path"):
        if getattr(args, k, None) is None:
            # not an argument for this model, pretrained embeddings don't matter
            continue
        elif skip_pretrained_embedding_loading:
            # if we want to skip pretrained, then hide the option from fairseq
            setattr(args, k, None)
        else:
            # otherwise we may need to modelzoo adjust the path for fairseq
            import warnings
            warnings.warn("We recommend using --embedding-type instead")
            setattr(args, k, modelzoo_path(opt.get("datapath"), getattr(args, k)))

    # Here we hardcode a few options that we currently do not support
    # turn off distributed training
    args.distributed_world_size = 1
    args.distributed_rank = 0

    return args, vars(args)


class _FairseqDictionary(DictionaryAgent):
    """
    Skeleton dictionary class needed for interaction with fairseq-py.

    This class mostly just adds some basic API behavior that Fairseq internally
    expects from dictionaries.

    It also inserts a fake token at the 0th index of the dictionary, as
    fairseq-py maintains backwards compatibility with fairseq-lua, which uses
    1 indexing.
    """
    # Name of our fake lua compatibility token
    _LUA = '__LUACOMPAT__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # insert the fairseq-lua compatibility token to emulate 1-indexing.
        # This 1-indexing assumption is baked into a couple of places in fairseq-py,
        # and is unavoidable at the moment.
        #
        # Because of the structure of DictionaryAgent, it's difficult to force
        # a token in the 0th position without breaking load()ing. I've found
        # this to be the best way.

        # add the token to the dictionary
        self.add_token(_FairseqDictionary._LUA)
        # force it to be the "most frequent" token
        self.freq[_FairseqDictionary._LUA] = self.freq[self.null_token] + 1
        # sort the list to ensure the lua token is placed first. trim=False to
        # ensure shuffle is non-destructive.
        self.sort(trim=False)

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

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        # first we need to add the general torch agent operations
        super(FairseqAgent, cls).add_cmdline_args(argparser)

        # let's store any defaults that were overridden
        old_defaults = argparser._defaults
        if 'clip_norm' not in old_defaults:
            # fairseq has a few awful defaults
            old_defaults['clip_norm'] = 1.0
        if 'optimizer' not in old_defaults:
            old_defaults['optimizer'] = 'adam'
            old_defaults['adam_betas'] = '(0.9,0.98)'

        agent = argparser.add_argument_group('Fairseq Arguments')
        agent.add_argument(
            '--fp16',
            default=False,
            type='bool',
            help='Use fp16 training'
        )
        agent.add_argument(
            '--fp16-init-scale',
            default=2**7,
            type=int,
            help='default FP16 loss scale'
        )
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
            type='bool',
            metavar='BOOL',
            help='Skips test time beam search. Much faster if you only need PPL',
        )

        # Check subargs for generation, optimizers, criterions, archs, etc
        options.add_generation_args(argparser)
        options.add_optimization_args(argparser)
        options.add_checkpoint_args(argparser)

        # restore any user set defaults that fairseq possibly overrode
        argparser.set_defaults(**old_defaults)
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

        # once again restore any user-set defaults
        argparser.set_defaults(**old_defaults)
        known_args = argparser.parse_known_args(nohelp=True)[0]

        if hasattr(known_args, "arch") and known_args.arch is not None:
            arch = known_args.arch
            arch_group = argparser.add_argument_group(
                "{} architecture arguments".format(arch)
            )
            models.ARCH_MODEL_REGISTRY[arch].add_args(arch_group)

        if hasattr(known_args, "criterion"):
            crit_group = argparser.add_argument_group(
                '{} criterion arguments'.format(known_args.criterion)
            )
            criterions.CRITERION_REGISTRY[known_args.criterion].add_args(crit_group)

        # one last time, restore any user set defaults
        argparser.set_defaults(**old_defaults)

    @staticmethod
    def dictionary_class():
        # Force use of the Fairseq Dictionary
        return _FairseqDictionary

    def __init__(self, opt, shared=None):
        # In general use a basic TorchAgent wherever possible
        super().__init__(opt, shared)
        if not shared:
            # this is not a shared instance of this class, so do full initialization

            # check early if we're going to be loading the model from a checkpoint
            model_file_exists = (
                self.opt.get('model_file') and os.path.isfile(self.opt['model_file'])
            )

            # fairseq expects options to be in argparse format, instead of a dict
            # We also need to do some argument postprocessing and whatnot
            # We'll skip pretrained embeddings if we're going to override them with
            # a model checkpoint anyway
            self.args, self.opt = _fairseq_opt_wrapper(opt, model_file_exists)

            # seed the RNG
            torch.manual_seed(self.args.seed)

            # Just some identifying info
            self.id = "fairseq:{}".format(self.args.arch)

            # We need a placeholder task for fairseq
            self.task = _ParlaiTask(self.dict)

            # meters for keeping track of loss, ppl, etc.
            self.meters = defaultdict(AverageMeter)

            # actually construct the model and generator
            self.model = self.build_model()

            # Construct the generator and scorer
            self.generator = SequenceGenerator(
                [self.model],
                tgt_dict=self.dict,
                beam_size=self.args.beam,
                stop_early=(not self.args.no_early_stop),
                normalize_scores=(not self.args.unnormalized),
                len_penalty=self.args.lenpen,
                unk_penalty=self.args.unkpen,
                sampling=self.args.sampling,
                sampling_topk=self.args.sampling_topk,
                sampling_temperature=self.args.sampling_temperature,
            )
            self.scorer = SequenceScorer([self.model], self.dict)

            # set up the grader and the trainer
            self.criterion = criterions.build_criterion(self.args, self.task)

            # TODO: we might choose to add a --no-fp16 opt in the future to
            # explicitly disable fp16 instead
            if not self.args.fp16 and torch.cuda.get_device_capability(0)[0] >= 7:
                print("Heads up: using --fp16 could be a lot faster!")
            if self.use_cuda:
                self.trainer = trainer.Trainer(
                    self.args, self.task, self.model, self.criterion, None,
                )
                self.trainer._build_optimizer()
            else:
                self.trainer = None

            # if the model already existed, let's preload it and the trainer
            if model_file_exists:
                print('Loading existing model params from ' + self.opt['model_file'])
                self.load(self.opt.get('model_file'))

            # move things to the GPU if possible
            if self.use_cuda:
                self.model = self.model.cuda()
                self.generator = self.generator.cuda()
        else:
            self.model = shared['model']
            self.trainer = shared['trainer']
            self.generator = shared['generator']
            self.dict = shared['dict']
            self.args = shared['args']
            self.meters = shared['meters']

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

    def build_model(self):
        """
        Construct the actual Fairseq model. Default implementation is to use
        Fairseq's arch builder, but this method may be overridden to build custom
        models.
        """
        model_class = models.ARCH_MODEL_REGISTRY[self.args.arch]
        model = model_class.build_model(self.args, self.task)
        if self.args.embedding_type != 'random':
            self._copy_embeddings(
                model.encoder.embed_tokens.weight, self.args.embedding_type
            )
        return model

    def share(self):
        shared = super().share()
        shared['model'] = self.model
        shared['trainer'] = self.trainer
        shared['generator'] = self.generator
        shared['dict'] = self.dict
        shared['args'] = self.args
        shared['meters'] = self.meters
        return shared

    def save(self, path):
        """Save using fairseq's checkpointing."""
        if not path:
            return
        self.trainer.save_checkpoint(path, {'opt': self.opt, 'epoch': 0})
        # Parlai expects options to also be saved
        with open(path + '.opt', 'w') as handle:
            # overridden options shouldn't be stored, only the main ones
            if 'override' in self.opt:
                del self.opt['override']
            json.dump(self.opt, handle)

        # force save the dict
        self.dict.save(path + '.dict', sort=False)

    def load(self, path):
        """Load using fairseq's checkpointing."""
        if self.trainer:
            old_options = self.trainer.load_checkpoint(path, self.args.reset_optimizer)
            self._check_opts_unchanged(old_options, self.opt)
        else:
            load_model_state(path, self.model)

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

    def is_valid(self, obs):
        """Override from TorchAgent.
        Check if an observation has no tokens in it."""
        return len(obs.get('text_vec', [])) > 0

    def batchify(self, obs_batch):
        """
        Override parent batchify to set requirements for fairseq.

        Fairseq depends on sorted batch inputs for a call to rnn.pad_packed_sequence.
        Fairseq models cannot handle zero length sentences
        """
        return super().batchify(obs_batch, sort=True)

    def _update_metrics(self, metrics, sample):
        if metrics is None:
            # probably got an overflow in fp16 mode. don't count this sample
            return

        bsz = len(sample['target'])
        ntok = sample['ntokens']
        ssize = metrics['sample_size']

        for k, v in metrics.items():
            if k in {'ntokens', 'nsentences', 'sample_size'}:
                # don't need these
                continue
            elif k == "nll_loss":
                # nll loss is always normalized by ntokens
                self.meters[k].update(v, ntok)
            elif k == "loss":
                # loss is explicitly normalized by passed up sample size
                self.meters[k].update(v, ssize)
            else:
                # assume everything else it's averaged over bsz
                self.meters[k].update(v, bsz)

    def train_step(self, batch):
        """Process batch of inputs and targets and train on them.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.
        """
        if batch.text_vec is None:
            return
        self.is_training = True
        sample = self._make_sample(batch)
        self.model.train()
        metrics = self.trainer.train_step([sample])
        self._update_metrics(metrics, sample)

    def eval_step(self, batch):
        """Process batch of inputs.

        If the batch includes labels, calculate validation metrics as well.
        If --skip-generation is not set, return a prediction for each input.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.
        """
        if batch.text_vec is None:
            return
        self.is_training = False
        samples = self._make_sample(batch)
        self.model.eval()
        if batch.label_vec is not None and self.trainer is not None:
            # Interactive mode won't have a gold label
            metrics = self.trainer.valid_step(samples)
            self._update_metrics(metrics, samples)

        # Output placeholders
        reranked_cands = None
        generated_output = None

        # Grade each of the candidate sequences
        if batch.candidate_vecs is not None:
            bsz = len(batch.text_vec)
            reranked_cands = []
            # score the candidates for each item in the batch separately, so that
            # we can support variable number of candidates
            for i in range(bsz):
                cands = batch.candidate_vecs[i]
                if not cands:
                    reranked_cands.append(None)
                    continue
                ncand = len(cands)
                # repeat the input many times
                xs = batch.text_vec[i].unsqueeze(0).expand(ncand, -1)
                # some models crash if there's leading padding on every example
                xs = xs[:, :batch.text_lengths[i]]
                # and appropriately pack the outputs
                ys, _ = padded_tensor(cands, self.NULL_IDX, self.use_cuda)
                s = self._make_sample(xs=xs, ys=ys)
                # perform the actual grading, extract the scores
                scored = list(self.scorer.score_batched_itr([s], cuda=self.use_cuda))
                scores = [s[3][0]['score'].item() for s in scored]
                # intentional hanging comma here; argsort returns a list
                ranked, = argsort(scores, batch.candidates[i], descending=True)
                reranked_cands.append(ranked)

        # Next generate freely to create our response
        if not self.args.skip_generation:
            generated_output = self._generate(samples)
        elif reranked_cands:
            # we're skiping generation, but we're also grading candidates
            # so output the highest ranked candidate
            # In the case of zero candidates, we don't have something to rank,
            # so we may need to pass on that None
            generated_output = [
                ranked and ranked[0] or None for ranked in reranked_cands
            ]
        else:
            # no output at all
            pass

        return Output(generated_output, reranked_cands)

    def _generate(self, samples):
        no_prev_token = {
            k: v for k, v in samples['net_input'].items() if k != 'prev_output_tokens'
        }
        gens = self.generator.generate(no_prev_token, maxlen=64)
        bsz = samples['net_input']['src_tokens'].size(0)
        responses = []
        for i in range(bsz):
            beams = gens[i]
            selected = max(beams, key=lambda x: x["score"])
            tokens = selected["tokens"]
            start = 0
            end = -1
            for i, t in enumerate(tokens):
                t = t.item()
                if t == self.dict.bos_index:
                    # don't include <s> token
                    start = i + 1
                    continue
                if t == self.dict.eos_index:
                    # stop (and don't include) </s> token
                    end = i
                    break
            responses.append(self.dict.vec2txt(tokens[start:end]))
        return responses

    def report(self):
        """Return metrics calculated by the model."""
        # if we haven't initialized yet, just return a dummy object
        if not hasattr(self, "trainer"):
            return {}

        output = {k: v.avg for k, v in self.meters.items()}

        if "nll_loss" in self.meters:
            # special case, we used sentence averaging so ppl comes from nll_loss
            output["ppl"] = np.exp2(self.meters["nll_loss"].avg)
        else:
            # normal case, just use loss
            output["ppl"] = np.exp2(self.meters["loss"].avg)

        # Fairseq trainer metrics we'll pass up the way
        trainer_metrics = {"ups", "wps", "gnorm", "clip"}
        if self.is_training:
            for k in trainer_metrics:
                output[k] = self.trainer.meters[k].avg

        # for display purposes
        output = {k: round_sigfigs(v, 4) for k, v in output.items()}
        return output

    def reset_metrics(self):
        """Reset metrics calculated by the model back to zero."""
        if not hasattr(self, "trainer"):
            # We haven't set up the trainer yet, so we don't have any metrics
            return
        # We need to reset everything
        self.meters.clear()
        if self.trainer:
            for k in self.trainer.meters:
                self.trainer.meters[k].reset()

    def receive_metrics(self, metrics_dict):
        """Update lr scheduler with validation loss."""
        # TODO: this should be smarter
        self.trainer.lr_step(-1, metrics_dict["loss"])

    # Helper functions
    def _seq_length(self, xs):
        """Compute length of the sequence (non-padded size)."""
        return xs.ne(self.dict.pad_index).long().sum(dim=-1)

    def _right_shifted_ys(self, ys):
        """Replace first token with EOS and shift remaining tokens right 1."""
        result = torch.LongTensor(ys.size())
        result[:, 0] = self.dict.eos_index
        result[:, 1:] = ys[:, :-1]
        return result

    def _make_sample(self, batch=None, xs=None, ys=None):
        """Generate a sample object that Fairseq expects."""
        # add extra info to samples
        if batch is None and xs is None:
            raise ValueError("Must supply either batch or xs")
        if batch is None and ys is None:
            raise ValueError("Must supply either batch or ys")
        if xs is None:
            xs = batch.text_vec
        if ys is None:
            ys = batch.label_vec
        repadded = convert_padding_direction(xs, self.dict.pad(), right_to_left=True)
        sample = {}
        sample["id"] = torch.arange(len(xs) - 1)
        sample["net_input"] = {
            "src_tokens": repadded,
            "src_lengths": self._seq_length(xs),
        }
        if ys is not None:
            sample["target"] = ys
            sample["ntokens"] = sum(self._seq_length(ys)).item()
            sample["net_input"]["prev_output_tokens"] = self._right_shifted_ys(ys)
        return sample
