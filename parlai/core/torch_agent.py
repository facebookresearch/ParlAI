#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
General utility code for building PyTorch-based agents in ParlAI.

Contains the following main utilities:

* TorchAgent class which serves as a useful parent class for other model agents
* Batch namedtuple which is the input type of the main abstract methods of
  the TorchAgent class
* Output namedtuple which is the expected output type of the main abstract
  methods of the TorchAgent class

See below for documentation on each specific tool.
"""

from abc import ABC, abstractmethod
from collections import deque
import json
import random
import numpy as np
import os
from torch import optim

from parlai.core.agents import Agent
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import (
    AttrDict,
    argsort,
    padded_tensor,
    warn_once,
    round_sigfigs,
    fp16_optimizer_wrapper,
)
from parlai.core.distributed_utils import is_primary_worker

try:
    import torch
except ImportError:
    raise ImportError('Need to install Pytorch: go to pytorch.org')


class Batch(AttrDict):
    """
    Batch is a namedtuple containing data being sent to an agent.

    This is the input type of the train_step and eval_step functions.
    Agents can override the batchify function to return an extended namedtuple
    with additional fields if they would like, though we recommend calling the
    parent function to set up these fields as a base.

    :param text_vec:
        bsz x seqlen tensor containing the parsed text data.

    :param text_lengths:
        list of length bsz containing the lengths of the text in same order as
        text_vec; necessary for pack_padded_sequence.

    :param label_vec:
        bsz x seqlen tensor containing the parsed label (one per batch row).

    :param label_lengths:
        list of length bsz containing the lengths of the labels in same order as
        label_vec.

    :param labels:
        list of length bsz containing the selected label for each batch row (some
        datasets have multiple labels per input example).

    :param valid_indices:
        list of length bsz containing the original indices of each example in the
        batch. we use these to map predictions back to their proper row, since e.g.
        we may sort examples by their length or some examples may be invalid.

    :param candidates:
        list of lists of text. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param candidate_vecs:
        list of lists of tensors. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param image:
        list of image features in the format specified by the --image-mode arg.

    :param observations:
        the original observations in the batched order
    """

    def __init__(
        self,
        text_vec=None,
        text_lengths=None,
        label_vec=None,
        label_lengths=None,
        labels=None,
        valid_indices=None,
        candidates=None,
        candidate_vecs=None,
        image=None,
        observations=None,
        **kwargs
    ):
        super().__init__(
            text_vec=text_vec,
            text_lengths=text_lengths,
            label_vec=label_vec,
            label_lengths=label_lengths,
            labels=labels,
            valid_indices=valid_indices,
            candidates=candidates,
            candidate_vecs=candidate_vecs,
            image=image,
            observations=observations,
            **kwargs
        )


class Output(AttrDict):
    """
    Output is an object containing agent predictions.

    This is the expected return type of the train_step and eval_step functions,
    though agents can choose to return None if they do not want to answer.

    :param List[str] text:
        list of strings of length bsz containing the predictions of the model

    :param List[List[str]] text_candidates:
        list of lists of length bsz containing ranked predictions of the model.
        each sub-list is an ordered ranking of strings, of variable length.
    """

    def __init__(self, text=None, text_candidates=None, **kwargs):
        super().__init__(text=text, text_candidates=text_candidates, **kwargs)


class History(object):
    """
    History handles tracking the dialogue state over the course of an episode.

    History may also be used to track the history of any field.

    :param field:
        field in the observation to track over the course of the episode
        (defaults to 'text')

    :param vec_type:
        specify a 'list' or 'deque' to save the history in this object

    :param maxlen:
        if `vec_type` is 'deque', this sets the maximum length of that object

    :param p1_token:
        token indicating 'person 1'; opt must have 'person_tokens' set to True
        for this to be added

    :param p1_token:
        token indicating 'person 2'; opt must have 'person_tokens' set to True
        for this to be added

    :param dict_agent:
        DictionaryAgent object for tokenizing the history
    """

    def __init__(
        self,
        opt,
        field='text',
        vec_type='deque',
        maxlen=None,
        size=-1,
        p1_token='__p1__',
        p2_token='__p2__',
        dict_agent=None,
    ):
        self.field = field
        self.dict = dict_agent
        self.delimiter = opt.get('delimiter', '\n')
        self.delimiter_tok = self.parse(self.delimiter)
        self.size = size
        self.split_on_newln = opt.get('split_lines', False)

        # set up history objects
        if vec_type != 'deque' and vec_type != 'list':
            raise RuntimeError('Type {} is not supported for history'.format(vec_type))
        self.vec_type = vec_type
        self.max_len = maxlen

        self.history_strings = []
        self.history_raw_strings = []
        self.history_vecs = []

        # person token args
        self.add_person_tokens = opt.get('person_tokens', False)
        self.add_p1_after_newln = opt.get('add_p1_after_newln', False)
        self.p1_token = p1_token
        self.p2_token = p2_token

        # tracking when to clear history
        self.reset_on_next_update = False

    def parse(self, text):
        """Tokenize text with the given dictionary."""
        return self.dict.txt2vec(text)

    def reset(self):
        """Clear the history."""
        self.history_raw_strings = []
        self.history_strings = []
        self.history_vecs = []

    def _update_strings(self, text):
        if self.size > 0:
            while len(self.history_strings) >= self.size:
                self.history_strings.pop(0)
        self.history_strings.append(text)

    def _update_raw_strings(self, text):
        if self.size > 0:
            while len(self.history_raw_strings) >= self.size:
                self.history_raw_strings.pop(0)
        self.history_raw_strings.append(text)

    def _update_vecs(self, text):
        if self.size > 0:
            while len(self.history_vecs) >= self.size:
                self.history_vecs.pop(0)
        self.history_vecs.append(self.parse(text))

    def update_history(self, obs, add_next=None):
        """
        Update the history with the given observation.

        :param add_next:
            string to append to history prior to updating it with the
            observation
        """
        if self.reset_on_next_update:
            # this is the first example in a new episode, clear the previous
            # history
            self.reset()
            self.reset_on_next_update = False

        if add_next is not None:
            self._update_raw_strings(add_next)
            if self.add_person_tokens:
                add_next = self._add_person_tokens(add_next, self.p2_token)
            # update history string
            self._update_strings(add_next)
            # update history vecs
            self._update_vecs(add_next)

        if self.field in obs and obs[self.field] is not None:
            if self.split_on_newln:
                next_texts = obs[self.field].split('\n')
            else:
                next_texts = [obs[self.field]]
            for text in next_texts:
                self._update_raw_strings(text)
                if self.add_person_tokens:
                    text = self._add_person_tokens(
                        obs[self.field], self.p1_token, self.add_p1_after_newln
                    )
                # update history string
                self._update_strings(text)
                # update history vecs
                self._update_vecs(text)

        if obs['episode_done']:
            # end of this episode, clear the history when we see a new example
            self.reset_on_next_update = True

    def get_history_str(self):
        """Return the string version of the history."""
        if len(self.history_strings) > 0:
            return self.delimiter.join(self.history_strings)
        return None

    def get_history_vec(self):
        """Return a vectorized version of the history."""
        if len(self.history_vecs) == 0:
            return None

        if self.vec_type == 'deque':
            history = deque(maxlen=self.max_len)
            for vec in self.history_vecs[:-1]:
                history.extend(vec)
                history.extend(self.delimiter_tok)
            history.extend(self.history_vecs[-1])
        else:
            # vec type is a list
            history = []
            for vec in self.history_vecs[:-1]:
                history += vec
                history += self.delimiter_tok
            history += self.history_vecs[-1]

        return history

    def get_history_vec_list(self):
        """Return a list of history vecs."""
        return self.history_vecs

    def _add_person_tokens(self, text, token, add_after_newln=False):
        if add_after_newln:
            split = text.split('\n')
            split[-1] = token + ' ' + split[-1]
            return '\n'.join(split)
        else:
            return token + ' ' + text


class TorchAgent(ABC, Agent):
    """
    A provided abstract base agent for any model that wants to use Torch.

    Exists to make it easier to implement a new agent.
    Not necessary, but reduces duplicated code.

    Many methods are intended to be either used as is when the default is
    acceptable, or to be overriden and called with super(), with the extra
    functionality added to the initial result. See the method comment for
    recommended behavior.

    This agent serves as a common framework for all ParlAI models which want
    to use PyTorch.
    """

    P1_TOKEN = '__p1__'
    P2_TOKEN = '__p2__'

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {
            k.lower(): v
            for k, v in optim.__dict__.items()
            if not k.startswith('__') and k[0].isupper()
        }
        try:
            import apex.optimizers.fused_adam as fused_adam

            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam

            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return DictionaryAgent

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return History

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add the default commandline args we expect most agents to want."""
        agent = argparser.add_argument_group('TorchAgent Arguments')
        # pretrained embedding arguments
        agent.add_argument(
            '-emb',
            '--embedding-type',
            default='random',
            choices=[
                'random',
                'glove',
                'glove-fixed',
                'glove-twitter-fixed',
                'fasttext',
                'fasttext-fixed',
                'fasttext_cc',
                'fasttext_cc-fixed',
            ],
            help='Choose between different strategies for initializing word '
            'embeddings. Default is random, but can also preinitialize '
            'from Glove or Fasttext. Preinitialized embeddings can also '
            'be fixed so they are not updated during training.',
        )
        agent.add_argument(
            '-embp',
            '--embedding-projection',
            default='random',
            help='If pretrained embeddings have a different dimensionality '
            'than your embedding size, strategy for projecting to the '
            'correct size. If the dimensions are the same, this is '
            'ignored unless you append "-force" to your choice.',
        )
        # optimizer arguments
        agent.add_argument(
            '--fp16', type='bool', default=False, help='Use fp16 computations.'
        )
        agent.add_argument(
            '-opt',
            '--optimizer',
            default='sgd',
            choices=cls.optim_opts(),
            help='Choose between pytorch optimizers. Any member of torch.optim'
            ' should be valid.',
        )
        agent.add_argument(
            '-lr', '--learningrate', type=float, default=1, help='learning rate'
        )
        agent.add_argument(
            '-clip',
            '--gradient-clip',
            type=float,
            default=0.1,
            help='gradient clipping using l2 norm',
        )
        agent.add_argument(
            '-mom',
            '--momentum',
            default=0,
            type=float,
            help='if applicable, momentum value for optimizer.',
        )
        agent.add_argument(
            '--nesterov',
            default=True,
            type='bool',
            help='if applicable, whether to use nesterov momentum.',
        )
        agent.add_argument(
            '-nu',
            '--nus',
            default='0.7',
            type='floats',
            help='if applicable, nu value(s) for optimizer. can use a single '
            'value like 0.7 or a comma-separated tuple like 0.7,1.0',
        )
        agent.add_argument(
            '-beta',
            '--betas',
            default='0.9,0.999',
            type='floats',
            help='if applicable, beta value(s) for optimizer. can use a single '
            'value like 0.9 or a comma-separated tuple like 0.9,0.999',
        )
        # lr scheduler
        agent.add_argument(
            '--lr-scheduler',
            type=str,
            default='reduceonplateau',
            choices=['reduceonplateau', 'none', 'fixed', 'invsqrt'],
            help='Learning rate scheduler.',
        )
        agent.add_argument(
            '--lr-scheduler-patience',
            type=int,
            default=3,
            help='LR scheduler patience. In number of validation runs. If using '
            'fixed scheduler, LR is decayed every <patience> validations.',
        )
        agent.add_argument(
            '--lr-scheduler-decay',
            type=float,
            default=0.5,
            help='Decay factor for LR scheduler, or how much LR is multiplied by '
            'when it is lowered.',
        )
        agent.add_argument(
            '--warmup-updates',
            type=int,
            default=-1,
            hidden=True,
            help='Learning rate warmup period, in number of SGD updates. '
            'Linearly scales up LR over period. Only enabled if > 0.',
        )
        agent.add_argument(
            '--warmup-rate',
            type=float,
            default=1e-4,
            hidden=True,
            help='Warmup learning rate *multiplier*. Initial LR is multiplied by '
            'this value. Linearly adjusted up to 1.0 across --warmup-updates '
            'steps.',
        )
        agent.add_argument(
            '--update-freq',
            type=int,
            default=-1,
            hidden=True,
            help='Accumulate gradients N times before performing an optimizer.step().',
        )
        # preprocessing arguments
        agent.add_argument(
            '-rc',
            '--rank-candidates',
            type='bool',
            default=False,
            help='Whether the model should parse candidates for ranking.',
        )
        agent.add_argument(
            '-tr',
            '--truncate',
            default=-1,
            type=int,
            help='Truncate input lengths to increase speed / use less memory.',
        )
        agent.add_argument(
            '--text-truncate',
            type=int,
            help='Text input truncation length: if not specified, this will '
            'default to `truncate`',
        )
        agent.add_argument(
            '--label-truncate',
            type=int,
            help='Label truncation length: if not specified, this will default '
            'to `truncate`',
        )
        agent.add_argument(
            '-histsz',
            '--history-size',
            default=-1,
            type=int,
            help='Number of past dialog utterances to remember.',
        )
        agent.add_argument(
            '-pt',
            '--person-tokens',
            type='bool',
            default=False,
            help='add person tokens to history. adds __p1__ in front of input '
            'text and __p2__ in front of past labels when available or '
            'past utterances generated by the model. these are added to '
            'the dictionary during initialization.',
        )
        agent.add_argument(
            '--split-lines',
            type='bool',
            default=False,
            help='split the dialogue history on newlines and save in separate '
            'vectors',
        )
        agent.add_argument(
            '--use-reply',
            default='label',
            hidden=True,
            choices=['label', 'model', 'none'],
            help='Which previous replies to use as history. If label, use '
            'gold dataset replies. If model, use model\'s own replies. '
            'If none, do not track replies in history.',
        )
        agent.add_argument(
            '--add-p1-after-newln',
            type='bool',
            default=False,
            hidden=True,
            help='Add the other speaker token before the last newline in the '
            'input instead of at the beginning of the input. this is '
            'useful for tasks that include some kind of context before '
            'the actual utterance (e.g. squad, babi, personachat).',
        )
        agent.add_argument(
            '--delimiter',
            type=str,
            default='\n',
            help='Join history lines with this token, defaults to newline',
        )
        # GPU arguments
        # these gpu options are all mutually exclusive, and should error if the
        # user tries to present multiple of them
        gpugroup = agent.add_mutually_exclusive_group()
        gpugroup.add_argument(
            '-gpu', '--gpu', type=int, default=-1, help='which GPU to use'
        )
        gpugroup.add_argument(
            '--no-cuda',
            default=False,
            action='store_true',
            dest='no_cuda',
            help='disable GPUs even if available. otherwise, will use GPUs if '
            'available on the device.',
        )

        cls.dictionary_class().add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        """Initialize agent."""
        super().__init__(opt, shared)
        opt = self.opt
        if not shared:
            # intitialize any important structures from scratch
            self.replies = {}  # past replies
            self.dict = self.build_dictionary()
            if opt.get('fp16'):
                # Volta cores revert to FP32 hardware if tensors are not multiples
                # of 8 in all dimensions. This INCLUDES the embeddings layer! As
                # such, we need some extra magic to ensure the dictionary is padded
                # with extra tokens to make it a multiple of 8.
                if len(self.dict) % 8 != 0:
                    for i in range(8 - len(self.dict) % 8):
                        self.dict['__FP16_PAD_{}__'.format(i)] = 1
        else:
            # copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']
            if self.opt['batchsize'] == 1:
                # if we're not using batching (e.g. mturk), then replies really need
                # to stay separated
                self.replies = {}
            else:
                self.replies = shared['replies']

        if opt.get('numthreads', 1) > 1:
            torch.set_num_threads(1)

        # check for cuda
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        if self.use_cuda:
            if not shared:
                print('[ Using CUDA ]')
            if not shared and opt['gpu'] != -1:
                torch.cuda.set_device(opt['gpu'])
        # indicate whether using fp16
        self.fp16 = self.use_cuda and self.opt.get('fp16', False)

        # Default to the class name, sans "Agent". child can override
        self.id = type(self).__name__.replace("Agent", "")

        # now set up any fields that all instances may need
        self.EMPTY = torch.LongTensor([])
        self.NULL_IDX = self.dict[self.dict.null_token]
        self.START_IDX = self.dict[self.dict.start_token]
        self.END_IDX = self.dict[self.dict.end_token]

        # for gradient acumulation
        self._number_grad_accum = 0
        # for the LR scheduler
        self._number_training_updates = 0
        # fixed random seed
        self.random = random.Random(42)
        # which row in the batch this instance is
        self.batch_idx = shared and shared.get('batchindex') or 0
        # can remember as few as zero utterances if desired
        self.histsz = opt['history_size']
        # truncate == 0 might give funny behavior
        self.truncate = opt['truncate'] if opt['truncate'] >= 0 else None
        text_truncate = opt.get('text_truncate') or opt['truncate']
        self.text_truncate = text_truncate if text_truncate >= 0 else None
        label_truncate = opt.get('label_truncate') or opt['truncate']
        self.label_truncate = label_truncate if label_truncate >= 0 else None
        # stores up to hist_utt past observations within current dialog
        self.history = self.history_class()(
            opt,
            maxlen=self.text_truncate,
            size=self.histsz,
            p1_token=self.P1_TOKEN,
            p2_token=self.P2_TOKEN,
            dict_agent=self.dict,
        )

        self.is_training = False  # track whether model is training
        self.rank_candidates = opt['rank_candidates']
        self.add_person_tokens = opt.get('person_tokens', False)

    def build_dictionary(self):
        """
        Return the constructed dictionary, which will be set to self.dict.

        If you need to add additional tokens to the dictionary, this is likely
        the right place to do it.
        """
        d = self.dictionary_class()(self.opt)
        if self.opt.get('person_tokens'):
            d[self.P1_TOKEN] = 999999999
            d[self.P2_TOKEN] = 999999998
        return d

    def _get_init_model(self, opt, shared):
        """
        Get model file to initialize with.

        If `init_model` exits, we will return the path to that file and maybe
        load dict file from that path. Otherwise, use `model_file.`

        :return:  path to load model from, whether we loaded from `init_model`
                  or not
        """
        init_model = None
        is_finetune = False
        if not shared:  # only do this on first setup
            # first check load path in case we need to override paths
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                # check first for 'init_model' for loading model from file
                init_model = opt['init_model']
                is_finetune = True
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # next check for 'model_file', this would override init_model
                init_model = opt['model_file']
                is_finetune = False

            if init_model is not None:
                # if we are loading a model, should load its dict too
                if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                    opt['dict_file'] = init_model + '.dict'

        return init_model, is_finetune

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """
        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        if opt.get('momentum') > 0 and opt['optimizer'] in ['sgd', 'rmsprop', 'qhm']:
            # turn on momentum for optimizers that use it
            kwargs['momentum'] = opt['momentum']
            if opt['optimizer'] == 'sgd' and opt.get('nesterov', True):
                # for sgd, maybe nesterov
                kwargs['nesterov'] = opt.get('nesterov', True)
            elif opt['optimizer'] == 'qhm':
                # qhm needs a nu
                kwargs['nu'] = opt.get('nus', (0.7,))[0]
        elif opt['optimizer'] == 'adam':
            # turn on amsgrad for adam
            # amsgrad paper: https://openreview.net/forum?id=ryQu7f-RZ
            kwargs['amsgrad'] = True
        elif opt['optimizer'] == 'qhadam':
            # set nus for qhadam
            kwargs['nus'] = opt.get('nus', (0.7, 1.0))
        if opt['optimizer'] in ['adam', 'sparseadam', 'adamax', 'qhadam']:
            # set betas for optims that use it
            kwargs['betas'] = opt.get('betas', (0.9, 0.999))

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)
        if self.fp16:
            self.optimizer = fp16_optimizer_wrapper(self.optimizer)

        # TODO: we might want to hard reset optimizers here in the
        # case of fine tuning. Some rudimentary experiments seemed to
        # indicate that keeping adam weights around was desirable, so this
        # will remain the behavior for the time being.
        if optim_states and saved_optim_type != opt['optimizer']:
            # we changed from adam to adamax, or sgd to adam, or similar
            print('WARNING: not loading optim state since optim class changed.')
        elif optim_states:
            # check for any fp16/fp32 conversions we need to do
            optimstate_fp16 = ('loss_scaler' in optim_states)
            if self.fp16 and optimstate_fp16:
                # previously trained in fp16, now we're training in fp16.
                # ideally no action needed, but APEX broke backwards
                # compatibility and this is the hack around it.
                optim_states['loss_scaler'] = (
                    self.optimizer.state_dict()['loss_scaler']
                )
            elif optimstate_fp16 and not self.fp16:
                # old optimizer was fp16 but now we're doing fp32,
                # drop the fp16 wrapper from the state_dict and just load
                # the fp16 weights into the fp32 tensors
                optim_states = optim_states['optimizer_state_dict']
            elif not optimstate_fp16 and self.fp16:
                # old optimizer was fp32, but now we're doing fp16.
                # this is a bit clunky, but alternatives are worse
                self.optimizer.optimizer.load_state_dict(optim_states)
                return
            else:
                # previously trained in fp32, loading in fp32.
                # no special treatment needed.
                pass

            # finally, try to actually load the optimizer state
            try:
                self.optimizer.load_state_dict(optim_states)
            except ValueError:
                print('WARNING: not loading optim state since model params changed.')

    def build_lr_scheduler(self, states=None, hard_reset=False):
        """
        Create the learning rate scheduler, and assign it to self.scheduler.

        This scheduler will be updated upon a call to receive_metrics.
        May also create self.warmup_scheduler, if appropriate.

        :param state_dict states: Possible state_dict provided by model
            checkpoint, for restoring LR state

        :param bool hard_reset: If true, the LR scheduler should ignore the
            state dictionary.
        """
        optimizer = self.optimizer
        if self.fp16:
            # lr schedulers don't work with apex, they expect the "real" optimizer
            optimizer = optimizer.optimizer

        if self.opt.get('warmup_updates', -1) > 0:

            def _warmup_lr(step):
                start = self.opt['warmup_rate']
                end = 1.0
                progress = min(1.0, step / self.opt['warmup_updates'])
                lr_mult = start + (end - start) * progress
                return lr_mult

            self.warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, _warmup_lr)
        else:
            self.warmup_scheduler = None

        patience = self.opt.get('lr_scheduler_patience', 3)
        decay = self.opt.get('lr_scheduler_decay', 0.5)

        if self.opt.get('lr_scheduler') == 'none':
            self.scheduler = None
        elif decay == 1.0:
            warn_once(
                "Your LR decay is set to 1.0. Assuming you meant you wanted "
                "to disable learning rate scheduling. Adjust --lr-scheduler-decay "
                "if this is not correct."
            )
            self.scheduler = None
        elif self.opt.get('lr_scheduler') == 'reduceonplateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', factor=decay, patience=patience, verbose=True
            )
        elif self.opt.get('lr_scheduler') == 'fixed':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, patience, gamma=decay)
        elif self.opt.get('lr_scheduler') == 'invsqrt':
            if self.opt.get('warmup_updates', -1) <= 0:
                raise ValueError(
                    '--lr-scheduler invsqrt requires setting --warmup-updates'
                )
            warmup_updates = self.opt['warmup_updates']
            decay_factor = np.sqrt(max(1, warmup_updates))

            def _invsqrt_lr(step):
                return decay_factor / np.sqrt(max(1, step))

            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, _invsqrt_lr)
        else:
            raise ValueError(
                "Don't know what to do with lr_scheduler '{}'".format(
                    self.opt.get('lr_scheduler')
                )
            )

        # time to load LR state from the checkpoint, if possible.
        if states is None:
            # first make sure there are no null pointers
            states = {}

        if (
            # there is already an old LR scheduler saved on disk
            states and
            # and the old LR scheduler is different
            states.get('lr_scheduler_type') != self.opt['lr_scheduler'] and
            # and we're not already using a fresh scheduler
            not hard_reset
        ):
            # the LR scheduler changed, start things fresh
            warn_once("LR scheduler is different from saved. Starting fresh!")
            hard_reset = True

        if hard_reset:
            # We're not going to use the LR schedule, let's just exit
            return

        # do the actual loading (if possible)
        if 'number_training_updates' in states:
            self._number_training_updates = states['number_training_updates']
        if self.scheduler and 'lr_scheduler' in states:
            self.scheduler.load_state_dict(states['lr_scheduler'])
        if states.get('warmup_scheduler') and getattr(self, 'warmup_scheduler', None):
            self.warmup_scheduler.load_state_dict(states['warmup_scheduler'])

    def report(self):
        """
        Report metrics.

        Report includes learning rate and number of training updates.
        """
        metrics = {}
        # only report LR if we have a scheduler
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            current_lr = round_sigfigs(self.optimizer.param_groups[0]['lr'], 4)
            metrics['lr'] = round_sigfigs(current_lr, 4)
        metrics['num_updates'] = self._number_training_updates
        return metrics

    def _is_lr_warming_up(self):
        """Check if we're warming up the learning rate."""
        return (
            self.warmup_scheduler is not None
            and self._number_training_updates <= self.opt['warmup_updates']
        )

    def receive_metrics(self, metrics_dict):
        """
        Use the metrics to decide when to adjust LR schedule.

        This uses the loss as the validation metric if present, if not this
        function does nothing. Note that the model must be reporting loss for
        this to work.

        Override this to override the behavior.
        """
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            return

        if self._is_lr_warming_up():
            # we're not done warming up, so don't start using validation
            # metrics to adjust schedule
            return

        if self.opt['lr_scheduler'] == 'none':
            # no scheduler, nothing to adjust here
            pass
        elif self.opt['lr_scheduler'] == 'reduceonplateau':
            if 'loss' not in metrics_dict:
                # nothing to step on, just skip
                warn_once("LR scheduler expected to see loss metric, but didn't.")
                return
            self.scheduler.step(metrics_dict['loss'])
        elif self.opt['lr_scheduler'] == 'fixed':
            self.scheduler.step()
        elif self.opt['lr_scheduler'] == 'invsqrt':
            # this is a training step lr scheduler, nothing to adjust in validation
            pass
        else:
            raise ValueError(
                "Don't know how to work with lr scheduler '{}'".format(
                    self.opt['lr_scheduler']
                )
            )

    def _get_embtype(self, emb_type):
        # set up preinitialized embeddings
        try:
            import torchtext.vocab as vocab
        except ImportError as ex:
            print('Please install torch text with `pip install torchtext`')
            raise ex
        pretrained_dim = 300
        if emb_type.startswith('glove'):
            if 'twitter' in emb_type:
                init = 'glove-twitter'
                name = 'twitter.27B'
                pretrained_dim = 200
            else:
                init = 'glove'
                name = '840B'
            embs = vocab.GloVe(
                name=name,
                dim=pretrained_dim,
                cache=modelzoo_path(self.opt.get('datapath'), 'zoo:glove_vectors'),
            )
        elif emb_type.startswith('fasttext_cc'):
            init = 'fasttext_cc'
            from parlai.zoo.fasttext_cc_vectors.build import download

            embs = download(self.opt.get('datapath'))
        elif emb_type.startswith('fasttext'):
            init = 'fasttext'
            from parlai.zoo.fasttext_vectors.build import download

            embs = download(self.opt.get('datapath'))
        else:
            raise RuntimeError(
                'embedding type {} not implemented. check arg, '
                'submit PR to this function, or override it.'
                ''.format(emb_type)
            )
        return embs, init

    def _project_vec(self, vec, target_dim, method='random'):
        """
        If needed, project vector to target dimensionality.

        Projection methods implemented are the following:

        random - random gaussian matrix multiplication of input vector

        :param vec:
            one-dimensional vector

        :param target_dim:
            dimension of returned vector

        :param method:
            projection method. will be used even if the dim is not changing if
            method ends in "-force".
        """
        pre_dim = vec.size(0)
        if pre_dim != target_dim or method.endswith('force'):
            if method.startswith('random'):
                # random projection
                if not hasattr(self, 'proj_rp'):
                    self.proj_rp = torch.Tensor(pre_dim, target_dim).normal_()
                    # rescale so we're not destroying norms too much
                    # http://scikit-learn.org/stable/modules/random_projection.html#gaussian-random-projection
                    self.proj_rp /= target_dim
                return torch.mm(vec.unsqueeze(0), self.proj_rp)
            else:
                # TODO: PCA
                # TODO: PCA + RP
                # TODO: copy
                raise RuntimeError(
                    'Projection method not implemented: {}' ''.format(method)
                )
        else:
            return vec

    def _copy_embeddings(self, weight, emb_type, log=True):
        """
        Copy embeddings from the pretrained embeddings to the lookuptable.

        :param weight:
            weights of lookup table (nn.Embedding/nn.EmbeddingBag)

        :param emb_type:
            pretrained embedding type
        """
        if not is_primary_worker():
            # we're in distributed mode, copying embeddings in the workers
            # slows things down considerably
            return
        embs, name = self._get_embtype(emb_type)
        cnt = 0
        for w, i in self.dict.tok2ind.items():
            if w in embs.stoi:
                vec = self._project_vec(embs.vectors[embs.stoi[w]], weight.size(1))
                weight.data[i] = vec
                cnt += 1

        if log:
            print(
                'Initialized embeddings for {} tokens ({}%) from {}.'
                ''.format(cnt, round(cnt * 100 / len(self.dict), 1), name)
            )

    def share(self):
        """
        Share fields from parent as well as useful objects in this class.

        Subclasses will likely want to share their model as well.
        """
        shared = super().share()
        shared['opt'] = self.opt
        shared['dict'] = self.dict
        shared['replies'] = self.replies
        return shared

    def _add_start_end_tokens(self, vec, add_start=False, add_end=False):
        if add_start:
            vec.insert(0, self.START_IDX)
        if add_end:
            vec.append(self.END_IDX)
        return vec

    def _v2t(self, vec):
        """Convert token indices to string of tokens."""
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.END_IDX:
                break
            new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def _vectorize_text(
        self, text, add_start=False, add_end=False, truncate=None, truncate_left=True
    ):
        """
        Return vector from text.

        :param text:
            String to vectorize.

        :param add_start:
            Add the start token to the front of the tensor.

        :param add_end:
            Add the end token to the end of the tensor.

        :param truncate:
            Truncate to this many tokens >= 0, or None.

        :param truncate_left:
            Truncate from the left side (keep the rightmost tokens). You
            probably want this True for inputs, False for targets.
        """
        vec = self.dict.txt2vec(text)
        vec = self._add_start_end_tokens(vec, add_start, add_end)
        vec = self._check_truncate(vec, truncate, truncate_left)
        tensor = torch.LongTensor(vec)
        return tensor

    def _check_truncate(self, vec, truncate, truncate_left=False):
        """Check that vector is truncated correctly."""
        if truncate is None:
            return vec
        if len(vec) <= truncate:
            return vec
        if truncate_left:
            return vec[-truncate:]
        else:
            return vec[:truncate]

    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            obs['text'] = history.get_history_str()
            if obs['text'] is not None:
                obs['text_vec'] = history.get_history_vec()

        # check truncation
        if 'text_vec' in obs:
            obs['text_vec'] = torch.LongTensor(
                self._check_truncate(obs['text_vec'], truncate, True)
            )

        return obs

    def _set_label_vec(self, obs, add_start, add_end, truncate):
        """
        Set the 'labels_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        # convert 'labels' or 'eval_labels' into vectors
        if 'labels' in obs:
            label_type = 'labels'
        elif 'eval_labels' in obs:
            label_type = 'eval_labels'
        else:
            label_type = None

        if label_type is None:
            return

        elif label_type + '_vec' in obs:
            # check truncation of pre-computed vector
            obs[label_type + '_vec'] = self._check_truncate(
                obs[label_type + '_vec'], truncate
            )
        else:
            # pick one label if there are multiple
            lbls = obs[label_type]
            label = lbls[0] if len(lbls) == 1 else self.random.choice(lbls)
            vec_label = self._vectorize_text(label, add_start, add_end, truncate, False)
            obs[label_type + '_vec'] = vec_label
            obs[label_type + '_choice'] = label

        return obs

    def _set_label_cands_vec(self, obs, add_start, add_end, truncate):
        """
        Set the 'label_candidates_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        if 'label_candidates_vecs' in obs:
            if truncate is not None:
                # check truncation of pre-computed vectors
                vecs = obs['label_candidates_vecs']
                for i, c in enumerate(vecs):
                    vecs[i] = self._check_truncate(c, truncate)
        elif self.rank_candidates and obs.get('label_candidates'):
            obs['label_candidates'] = list(obs['label_candidates'])
            obs['label_candidates_vecs'] = [
                self._vectorize_text(c, add_start, add_end, truncate, False)
                for c in obs['label_candidates']
            ]
        return obs

    def vectorize(
        self,
        obs,
        history,
        add_start=True,
        add_end=True,
        text_truncate=None,
        label_truncate=None,
    ):
        """
        Make vectors out of observation fields and store in the observation.

        In particular, the 'text' and 'labels'/'eval_labels' fields are
        processed and a new field is added to the observation with the suffix
        '_vec'.

        If you want to use additional fields on your subclass, you can override
        this function, call super().vectorize(...) to process the text and
        labels, and then process the other fields in your subclass.

        Additionally, if you want to override some of these default parameters,
        then we recommend using a pattern like:

        .. code-block:: python

          def vectorize(self, *args, **kwargs):
              kwargs['add_start'] = False
              return super().vectorize(*args, **kwargs)


        :param obs:
            Single observation from observe function.

        :param add_start:
            default True, adds the start token to each label.

        :param add_end:
            default True, adds the end token to each label.

        :param text_truncate:
            default None, if set truncates text vectors to the specified
            length.

        :param label_truncate:
            default None, if set truncates label vectors to the specified
            length.

        :return:
            the input observation, with 'text_vec', 'label_vec', and
            'cands_vec' fields added.
        """
        self._set_text_vec(obs, history, text_truncate)
        self._set_label_vec(obs, add_start, add_end, label_truncate)
        self._set_label_cands_vec(obs, add_start, add_end, label_truncate)
        return obs

    def is_valid(self, obs):
        """Determine if an observation is valid or not."""
        return 'text_vec' in obs or 'image' in obs

    def batchify(self, obs_batch, sort=False):
        """
        Create a batch of valid observations from an unchecked batch.

        A valid observation is one that passes the lambda provided to the
        function, which defaults to checking if the preprocessed 'text_vec'
        field is present which would have been set by this agent's 'vectorize'
        function.

        Returns a namedtuple Batch. See original definition above for in-depth
        explanation of each field.

        If you want to include additonal fields in the batch, you can subclass
        this function and return your own "Batch" namedtuple: copy the Batch
        namedtuple at the top of this class, and then add whatever additional
        fields that you want to be able to access. You can then call
        super().batchify(...) to set up the original fields and then set up the
        additional fields in your subclass and return that batch instead.

        :param obs_batch:
            List of vectorized observations

        :param sort:
            Default False, orders the observations by length of vectors. Set to
            true when using torch.nn.utils.rnn.pack_padded_sequence.  Uses the text
            vectors if available, otherwise uses the label vectors if available.
        """
        if len(obs_batch) == 0:
            return Batch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs, x_lens = None, None
        if any('text_vec' in ex for ex in exs):
            _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
            xs, x_lens = padded_tensor(
                _xs, self.NULL_IDX, self.use_cuda, fp16friendly=self.opt.get('fp16')
            )
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs)
        some_labels_avail = labels_avail or any('eval_labels_vec' in ex for ex in exs)

        ys, y_lens, labels = None, None, None
        if some_labels_avail:
            field = 'labels' if labels_avail else 'eval_labels'

            label_vecs = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels = [ex.get(field + '_choice') for ex in exs]
            y_lens = [y.shape[0] for y in label_vecs]

            ys, y_lens = padded_tensor(
                label_vecs,
                self.NULL_IDX,
                self.use_cuda,
                fp16friendly=self.opt.get('fp16'),
            )
            if sort and xs is None:
                ys, valid_inds, label_vecs, labels, y_lens = argsort(
                    y_lens, ys, valid_inds, label_vecs, labels, y_lens, descending=True
                )

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        return Batch(
            text_vec=xs,
            text_lengths=x_lens,
            label_vec=ys,
            label_lengths=y_lens,
            labels=labels,
            valid_indices=valid_inds,
            candidates=cands,
            candidate_vecs=cand_vecs,
            image=imgs,
            observations=exs,
        )

    def match_batch(self, batch_reply, valid_inds, output=None):
        """
        Match sub-batch of predictions to the original batch indices.

        Batches may be only partially filled (i.e when completing the remainder
        at the end of the validation or test set), or we may want to sort by
        e.g the length of the input sequences if using pack_padded_sequence.

        This matches rows back with their original row in the batch for
        calculating metrics like accuracy.

        If output is None (model choosing not to provide any predictions), we
        will just return the batch of replies.

        Otherwise, output should be a parlai.core.torch_agent.Output object.
        This is a namedtuple, which can provide text predictions and/or
        text_candidates predictions. If you would like to map additional
        fields into the batch_reply, you can override this method as well as
        providing your own namedtuple with additional fields.

        :param batch_reply:
            Full-batchsize list of message dictionaries to put responses into.

        :param valid_inds:
            Original indices of the predictions.

        :param output:
            Output namedtuple which contains sub-batchsize list of text outputs
            from model. May be None (default) if model chooses not to answer.
            This method will check for ``text`` and ``text_candidates`` fields.
        """
        if output is None:
            return batch_reply
        if output.text is not None:
            for i, response in zip(valid_inds, output.text):
                batch_reply[i]['text'] = response
        if output.text_candidates is not None:
            for i, cands in zip(valid_inds, output.text_candidates):
                batch_reply[i]['text_candidates'] = cands
        return batch_reply

    def last_reply(self, use_reply='label'):
        """
        Retrieve the last reply from the model.

        If available, we use the true label instead of the model's prediction.

        By default, batch_act stores the batch of replies and this method
        will extract the reply of the current instance from the batch.

        :param use_label:
            default true, use the label when available instead of the model's
            generated response.
        """
        # if the last observation was the end of an episode,
        # then we shouldn't use it as history
        if (use_reply == 'none' or not self.observation or
                self.observation['episode_done']):
            return None

        if use_reply == 'label':
            # first look for the true label, if we aren't on a new episode
            label_key = (
                'labels'
                if 'labels' in self.observation
                else 'eval_labels'
                if 'eval_labels' in self.observation
                else None
            )
            if label_key is not None:
                lbls = self.observation[label_key]
                last_reply = lbls[0] if len(lbls) == 1 else self.random.choice(lbls)
                return last_reply

        # otherwise, we use the last reply the model generated
        batch_reply = self.replies.get('batch_reply')
        if batch_reply is not None:
            return batch_reply[self.batch_idx].get('text')
        return None

    def observe(self, observation):
        """
        Process incoming message in preparation for producing a response.

        This includes remembering the past history of the conversation.
        """
        reply = self.last_reply(use_reply=self.opt.get('use_reply', 'label'))
        # update the history using the observation
        self.history.update_history(observation, add_next=reply)
        self.observation = observation
        return self.vectorize(
            self.observation,
            self.history,
            text_truncate=self.text_truncate,
            label_truncate=self.label_truncate,
        )

    def state_dict(self):
        """
        Get the state dict for saving.

        Override this method for more specific saving.
        """
        states = {}
        if hasattr(self, 'model'):  # save model params
            if hasattr(self.model, 'module'):
                # did we wrap in a DistributedDataParallel
                states['model'] = self.model.module.state_dict()
            else:
                states['model'] = self.model.state_dict()

        if hasattr(self, 'optimizer'):
            # save optimizer params
            states['optimizer'] = self.optimizer.state_dict()
            states['optimizer_type'] = self.opt['optimizer']

        # lr scheduler
        if torch.__version__.startswith('0.'):
            warn_once(
                "Must upgrade to Pytorch 1.0 to save the state of your " "LR scheduler."
            )
        else:
            states['number_training_updates'] = self._number_training_updates
            if getattr(self, 'scheduler', None):
                states['lr_scheduler'] = self.scheduler.state_dict()
                states['lr_scheduler_type'] = self.opt['lr_scheduler']
            if getattr(self, 'warmup_scheduler', None):
                states['warmup_scheduler'] = self.warmup_scheduler.state_dict()

        return states

    def save(self, path=None):
        """
        Save model parameters to path (or default to model_file arg).

        Please try to refrain from overriding this function, and instead
        override `state_dict(self)` for more specific saving.
        """
        path = self.opt.get('model_file', None) if path is None else path

        if path:
            if hasattr(self, 'dict'):  # force save dictionary
                # TODO: Look into possibly overriding opt('dict_file') with new path
                self.dict.save(path + '.dict', sort=False)
            states = self.state_dict()
            if states:  # anything found to save?
                with open(path, 'wb') as write:
                    torch.save(states, write)

                # save opt file
                with open(path + '.opt', 'w', encoding='utf-8') as handle:
                    if hasattr(self, 'model_version'):
                        self.opt['model_version'] = self.model_version()
                    json.dump(self.opt, handle)
                    # for convenience of working with jq, make sure there's a newline
                    handle.write('\n')

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        self.model.load_state_dict(state_dict)

    def load(self, path):
        """
        Return opt and model states.

        Override this method for more specific loading.
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        if 'model' in states:
            self.load_state_dict(states['model'])
        if 'optimizer' in states and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(states['optimizer'])
        return states

    def reset(self):
        """Clear internal states."""
        self.observation = {}
        self.history.reset()
        self.replies.clear()
        self.reset_metrics()

    def act(self):
        """Call batch_act with the singleton batch."""
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        """
        Process a batch of observations (batchsize list of message dicts).

        These observations have been preprocessed by the observe method.

        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        batch_size = len(observations)
        # initialize a list of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]

        # check if there are any labels available, if so we will train on them
        self.is_training = any('labels' in obs for obs in observations)

        # create a batch from the vectors
        batch = self.batchify(observations)

        if self.is_training:
            output = self.train_step(batch)
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back graidients.
                output = self.eval_step(batch)

        if output is None:
            self.replies['batch_reply'] = None
            return batch_reply

        self.match_batch(batch_reply, batch.valid_indices, output)
        self.replies['batch_reply'] = batch_reply

        return batch_reply

    @abstractmethod
    def train_step(self, batch):
        """[Abstract] Process one batch with training labels."""
        pass

    @abstractmethod
    def eval_step(self, batch):
        """[Abstract] Process one batch but do not train on it."""
        pass

    def backward(self, loss):
        """
        Perform a backward pass.

        It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        if self.fp16:
            self.optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    def update_params(self):
        """
        Perform step of optimization.

        Handles clipping gradients and adjusting LR schedule if needed.
        Gradient accumulation is also performed if agent is called with
        --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = self.opt.get('update_freq', 1)
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        # keep track up number of steps, compute warmup factor
        self._number_training_updates += 1

        # compute warmup adjustment if needed
        if self.opt.get('warmup_updates', -1) > 0:
            if not hasattr(self, 'warmup_scheduler'):
                raise RuntimeError('Looks like you forgot to call build_lr_scheduler')
            if self._is_lr_warming_up():
                self.warmup_scheduler.step(epoch=self._number_training_updates)

        if self.opt.get('lr_scheduler') == 'invsqrt' and not self._is_lr_warming_up():
            # training step scheduler
            self.scheduler.step(self._number_training_updates)

        if self.fp16:
            # we've been accumulating grads in fp16 and delaying the fp32 copy update.
            # finally time to perform the update.
            self.optimizer.update_master_grads()

        if self.opt.get('gradient_clip', -1) > 0:
            if self.fp16:
                self.optimizer.clip_master_grads(self.opt['gradient_clip'])
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.opt['gradient_clip']
                )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        if self._number_grad_accum != 0:
            # if we're accumulating gradients, don't actually zero things out yet.
            return

        self.optimizer.zero_grad()

    def _total_parameters(self):
        """
        Compute the total number of parameters in the model.

        :return:
            total number of parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters())

    def _trainable_parameters(self):
        """
        Compute the total number of trainable parameters in the model.

        :return:
            total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
