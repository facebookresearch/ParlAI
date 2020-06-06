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
* History class which handles tracking the dialogue state over the course of an episode.


See below for documentation on each specific tool.
"""

from typing import Dict, Any, Union, List, Tuple, Optional
from abc import ABC, abstractmethod
from collections import deque
import random
import os
import torch
import parlai.utils.logging as logging
from torch import optim

from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.utils.thread import SharedTable
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message
from parlai.utils.distributed import is_distributed
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.fp16 import (
    fp16_apex_available,
    fp16_optimizer_wrapper,
    MemoryEfficientFP16Optimizer,
    MemoryEfficientFP16Adam,
    Adafactor,
)
from parlai.core.metrics import (
    Metrics,
    Metric,
    GlobalAverageMetric,
    GlobalSumMetric,
    GlobalFixedMetric,
)
from parlai.utils.distributed import is_primary_worker
from parlai.utils.torch import argsort, compute_grad_norm, padded_tensor


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
        **kwargs,
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
            **kwargs,
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
        self._global_end_token = opt.get('history_add_global_end_token', None)
        if self._global_end_token is not None:
            self._global_end_token = self.dict[self.dict.end_token]

        # set up history objects
        if vec_type != 'deque' and vec_type != 'list':
            raise RuntimeError('Type {} is not supported for history'.format(vec_type))
        self.vec_type = vec_type
        self.max_len = maxlen

        self.history_strings = []
        self.history_raw_strings = []
        self.history_vecs = []
        self.temp_history = None

        # person token args
        self.add_person_tokens = opt.get('person_tokens', False)
        self.add_p1_after_newln = opt.get('add_p1_after_newln', False)
        self.p1_token = p1_token
        self.p2_token = p2_token

    def parse(self, text):
        """
        Tokenize text with the given dictionary.
        """
        return self.dict.txt2vec(text)

    def reset(self):
        """
        Clear the history.
        """
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

    def add_reply(self, text):
        """
        Add your own response to the history.
        """
        self._update_raw_strings(text)
        if self.add_person_tokens:
            text = self._add_person_tokens(text, self.p2_token)
        # update history string
        self._update_strings(text)
        # update history vecs
        self._update_vecs(text)

    def update_history(self, obs, temp_history=None):
        """
        Update the history with the given observation.

        param obs:     Observation used to update the history. param temp_history:
        Optional temporary string. If it is not None,     this string will be appended
        to the end of the     history. It will not be in the history on the     next
        dialogue turn. Set to None to stop adding     to the history.
        """
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

        self.temp_history = temp_history

    def get_history_str(self):
        """
        Return the string version of the history.
        """
        if len(self.history_strings) > 0:
            history = self.delimiter.join(self.history_strings)
            if self.temp_history is not None:
                history += self.temp_history
            return history

        return None

    def get_history_vec(self):
        """
        Return a vectorized version of the history.
        """
        if len(self.history_vecs) == 0:
            return None

        if self.vec_type == 'deque':
            history = deque(maxlen=self.max_len)
            for vec in self.history_vecs[:-1]:
                history.extend(vec)
                history.extend(self.delimiter_tok)
            history.extend(self.history_vecs[-1])
            if self.temp_history is not None:
                history.extend(self.parse(self.temp_history))
            if self._global_end_token is not None:
                history.extend([self._global_end_token])
        else:
            # vec type is a list
            history = []
            for vec in self.history_vecs[:-1]:
                history += vec
                history += self.delimiter_tok
            history += self.history_vecs[-1]
            if self.temp_history is not None:
                history.extend(self.parse(self.temp_history))
            if self._global_end_token is not None:
                history += [self._global_end_token]

        return history

    def get_history_vec_list(self):
        """
        Return a list of history vecs.
        """
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
            import apex.optimizers.fused_lamb as fused_lamb

            optims['fused_adam'] = fused_adam.FusedAdam
            optims['fused_lamb'] = fused_lamb.FusedLAMB
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

        # now pull in memory efficient implementations
        optims['mem_eff_adam'] = MemoryEfficientFP16Adam
        optims['adafactor'] = Adafactor

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
        """
        Add the default commandline args we expect most agents to want.
        """
        agent = argparser.add_argument_group('TorchAgent Arguments')
        agent.add_argument(
            '-i',
            '--interactive-mode',
            type='bool',
            default=False,
            help='Whether in full interactive mode or not,  which means generating text or'
            ' retrieving from a full set of candidates, which is necessary to actually'
            ' do full dialogue. However, during training or quick validation (e.g. PPL for'
            ' generation or ranking a few candidates for ranking models) you might want these'
            ' set to off.'
            ' Typically, scripts can set their preferred default behavior at the start,'
            ' e.g. eval scripts.',
        )
        # pretrained embedding arguments
        agent.add_argument(
            '-emb',
            '--embedding-type',
            default='random',
            choices=[
                'random',
                'glove',
                'glove-fixed',
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
        agent.add_argument(
            '--fp16', type='bool', default=False, help='Use fp16 computations.'
        )
        agent.add_argument(
            '--fp16-impl',
            type=str,
            default='apex',
            choices=['apex', 'mem_efficient'],
            help='Implementation of FP16 to use',
        )
        agent.add_argument(
            '--force-fp16-tokens',
            type='bool',
            default=False,
            hidden=True,
            help='Add the special fp16 tokens even if not using fp16.',
        )
        # optimizer arguments
        optim_group = agent.add_argument_group('Optimizer Arguments')
        optim_group.add_argument(
            '-opt',
            '--optimizer',
            default='sgd',
            choices=cls.optim_opts(),
            help='Choose between pytorch optimizers. Any member of torch.optim'
            ' should be valid.',
        )
        optim_group.add_argument(
            '-lr', '--learningrate', type=float, default=1, help='Learning rate'
        )
        optim_group.add_argument(
            '-clip',
            '--gradient-clip',
            type=float,
            default=0.1,
            help='gradient clipping using l2 norm',
        )
        optim_group.add_argument(
            '--adam-eps',
            type=float,
            default=1e-8,
            hidden=True,
            help='Epsilon value for Adam optimizers. Set to 1e-6 if your '
            'large model has stability issues, but prefer the default.',
        )
        optim_group.add_argument(
            '--adafactor-eps',
            default='1e-30,1e-3',
            type='floats',
            help='Epsilon values for adafactor optimizer: regularization '
            'constants for square gradient and parameter scale respectively',
            recommended='1e-30,1e-3',
        )
        optim_group.add_argument(
            '-mom',
            '--momentum',
            default=0,
            type=float,
            help='if applicable, momentum value for optimizer.',
        )
        optim_group.add_argument(
            '--nesterov',
            default=True,
            type='bool',
            help='if applicable, whether to use nesterov momentum.',
        )
        optim_group.add_argument(
            '-nu',
            '--nus',
            default='0.7',
            type='floats',
            help='if applicable, nu value(s) for optimizer. can use a single '
            'value like 0.7 or a comma-separated tuple like 0.7,1.0',
        )
        optim_group.add_argument(
            '-beta',
            '--betas',
            default='0.9,0.999',
            type='floats',
            help='if applicable, beta value(s) for optimizer. can use a single '
            'value like 0.9 or a comma-separated tuple like 0.9,0.999',
        )
        optim_group.add_argument(
            '-wdecay',
            '--weight-decay',
            type=float,
            default=None,
            help='Weight decay on the weights.',
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
        agent.add_argument(
            '--history-add-global-end-token',
            type='nonestr',
            default=None,
            hidden=True,
            choices=[None, 'end'],
            help='Add special token to the end of history encoding.',
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
        ParlAILRScheduler.add_cmdline_args(argparser)

    def __init__(self, opt: Opt, shared=None):
        """
        Initialize agent.
        """
        super().__init__(opt, shared)
        opt = self.opt

        # Safety checkers to ensure TorchAgent assumptions aren't being violated.
        self.__expecting_clear_history = False
        self.__expecting_to_reply = False

        # used for sharing metrics back to the teacher
        self._local_metrics: Dict[str, List[Metric]] = {}
        # we may want to temporarily disable local metrics, roughly similar to
        # `with torch.no_grad`. See TorchGeneratorAgent._init_cuda_buffer for
        # example
        self.__local_metrics_enabled = True

        # check for cuda
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        if self.use_cuda:
            if not shared:
                logging.info('Using CUDA')
            if not shared and opt['gpu'] != -1:
                torch.cuda.set_device(opt['gpu'])

        # whether we're using multi-gpu, a few different ways. these are not
        # supported by all models, but we can still keep track of the options
        self.model_parallel = opt.get('model_parallel', False) and self.use_cuda
        self.data_parallel = opt.get('data_parallel', False) and self.use_cuda
        if self.data_parallel and is_distributed():
            raise RuntimeError('Cannot combine --data-parallel and distributed mode.')
        if self.model_parallel and self.data_parallel:
            raise RuntimeError('Cannot combine --data-parallel and --model-parallel.')

        # indicate whether using fp16
        self.fp16 = self.use_cuda and self.opt.get('fp16', False)
        if self.fp16:
            # check that the implementation requested is available
            self.fp16_impl = self.opt.get('fp16_impl', 'apex')
            if self.fp16_impl == 'apex' and not fp16_apex_available():
                self.fp16 = False

        if shared is None:
            # intitialize any important structures from scratch
            self.dict = self.build_dictionary()

            if opt.get('fp16') or opt.get('force_fp16_tokens'):
                # Volta cores revert to FP32 hardware if tensors are not multiples
                # of 8 in all dimensions. This INCLUDES the embeddings layer! As
                # such, we need some extra magic to ensure the dictionary is padded
                # with extra tokens to make it a multiple of 8.
                from parlai.utils.torch import FP16_PAD_SIZE

                if len(self.dict) % FP16_PAD_SIZE != 0:
                    for i in range(FP16_PAD_SIZE - len(self.dict) % FP16_PAD_SIZE):
                        self.dict['__FP16_PAD_{}__'.format(i)] = 1

            # global_metrics keeps track of batch-level or global-level metrics
            self.global_metrics = Metrics(opt.get('numthreads', 1) > 1, shared=None)
            # self.metrics is there for legacy reasons
            self.metrics: Dict[str, Any] = {}
        else:
            # copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']
            self.model = shared['model']
            self.criterion = shared['criterion']
            self.metrics = shared['metrics']
            self.global_metrics = Metrics(
                opt.get('numthreads', 1) > 1, shared=shared['global_metrics']
            )

        if opt.get('numthreads', 1) > 1:
            torch.set_num_threads(1)

        # Default to the class name, sans "Agent". child can override
        self.id = type(self).__name__.replace("Agent", "")

        # now set up any fields that all instances may need
        self.EMPTY = torch.zeros(0, dtype=torch.long)
        self.NULL_IDX = self.dict[self.dict.null_token]
        self.START_IDX = self.dict[self.dict.start_token]
        self.END_IDX = self.dict[self.dict.end_token]

        # for gradient acumulation
        self._number_grad_accum = 0
        # for the LR scheduler
        self._number_training_updates = 0
        # fixed random seed
        self.random = random.Random(42)
        # can remember as few as zero utterances if desired
        self.histsz = opt['history_size']
        # truncate == 0 might give funny behavior
        self.truncate = opt['truncate'] if opt['truncate'] >= 0 else None
        text_truncate = opt.get('text_truncate') or opt['truncate']
        self.text_truncate = text_truncate if text_truncate >= 0 else None
        label_truncate = opt.get('label_truncate') or opt['truncate']
        self.label_truncate = label_truncate if label_truncate >= 0 else None
        # stores up to hist_utt past observations within current dialog
        self.history = self.build_history()

        self.is_training = False  # track whether model is training
        self.rank_candidates = opt['rank_candidates']
        self.add_person_tokens = opt.get('person_tokens', False)
        # set interactive mode or not according to options.
        self.set_interactive_mode(opt.get('interactive_mode', False), shared)

    def build_history(self):
        """
        Return the constructed history object.
        """
        return self.history_class()(
            self.opt,
            maxlen=self.text_truncate,
            size=self.histsz,
            p1_token=self.P1_TOKEN,
            p2_token=self.P2_TOKEN,
            dict_agent=self.dict,
        )

    def build_dictionary(self):
        """
        Return the constructed dictionary, which will be set to self.dict.

        If you need to add additional tokens to the dictionary, this is likely the right
        place to do it.
        """
        d = self.dictionary_class()(self.opt)
        if self.opt.get('person_tokens'):
            d[self.P1_TOKEN] = 999_999_999
            d[self.P2_TOKEN] = 999_999_998
        return d

    def _get_init_model(self, opt: Opt, shared):
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
            if (
                opt.get('load_from_checkpoint')
                and opt.get('init_model')
                and opt['init_model'].endswith('.checkpoint')
            ):
                # but if we're loading from a checkpoint, we should explicitly load
                # from that point
                init_model = opt['init_model']
                is_finetune = False

            if init_model is not None:
                # if we are loading a model, should load its dict too
                if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                    opt['dict_file'] = init_model + '.dict'

        return init_model, is_finetune

    @abstractmethod
    def build_model(self):
        """
        Construct the model and return it.
        """
        raise NotImplementedError('not implemented for this class')

    def _should_initialize_optimizer(self) -> bool:
        """
        Used to indicate whether we should initialize an optimizer.

        When this is off, we can save memory and use larger batches.
        """
        if self.opt.get('interactive_mode'):
            return False
        datatype = self.opt.get('datatype', '')
        is_train = 'train' in datatype and 'evalmode' not in datatype
        return is_train or self.opt.get('numthreads', 1) > 1

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
        if opt.get('weight_decay'):
            kwargs['weight_decay'] = opt['weight_decay']
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
            if self.fp16 and self.fp16_impl == 'mem_efficient':
                # grab this implementation instead
                opt['optimizer'] = 'mem_eff_adam'
        elif opt['optimizer'] == 'qhadam':
            # set nus for qhadam
            kwargs['nus'] = opt.get('nus', (0.7, 1.0))
        elif opt['optimizer'] == 'adafactor':
            # adafactor params
            kwargs['beta1'] = opt.get('betas', (0.9, 0.999))[0]
            kwargs['eps'] = opt['adafactor_eps']
            kwargs['warmup_init'] = opt.get('warmup_updates', -1) > 0

        if opt['optimizer'] in [
            'adam',
            'sparseadam',
            'fused_adam',
            'adamax',
            'qhadam',
            'fused_lamb',
        ]:
            # set betas for optims that use it
            kwargs['betas'] = opt.get('betas', (0.9, 0.999))
            # set adam optimizer, but only if user specified it
            if opt.get('adam_eps'):
                kwargs['eps'] = opt['adam_eps']

        # handle fused_adam where the user doesn't have apex installed
        if saved_optim_type == 'fused_adam' and 'fused_adam' not in self.optim_opts():
            # we trained with apex, but the user doesn't have apex installed.
            saved_optim_type = 'adam'

        if (
            self.opt['optimizer'] == 'fused_adam'
            and 'fused_adam' not in self.optim_opts()
        ):
            raise ImportError(
                'You are using --optimizer fused_adam, but you do not have APEX '
                'installed. Please install APEX (https://github.com/NVIDIA/apex) or '
                'switch to --optimizer adam.'
            )

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)
        if self.fp16:
            if self.fp16_impl == 'apex':
                self.optimizer = fp16_optimizer_wrapper(self.optimizer)
            else:
                # Using memory efficient optimizer
                opt_name = opt['optimizer']
                compatible_list = MemoryEfficientFP16Optimizer.compatible_optimizers()
                is_compat = opt_name in compatible_list
                if not is_compat:
                    raise RuntimeError(
                        f'The optimizer you selected {opt_name} is not compatible '
                        'with Memory Efficient FP16. Please select from among this '
                        f'list:\n{compatible_list}'
                    )
                self.optimizer = MemoryEfficientFP16Optimizer(self.optimizer)
        # TODO: we might want to hard reset optimizers here in the
        # case of fine tuning. Some rudimentary experiments seemed to
        # indicate that keeping adam weights around was desirable, so this
        # will remain the behavior for the time being.
        if optim_states and saved_optim_type != opt['optimizer']:
            # we changed from adam to adamax, or sgd to adam, or similar
            logging.warn('Not loading optim state since optim class changed.')
        elif optim_states:
            # check for any fp16/fp32 conversions we need to do
            optimstate_fp16 = 'loss_scaler' in optim_states
            if self.fp16 and optimstate_fp16:
                # previously trained in fp16, now we're training in fp16.
                # ideally no action needed, but APEX broke backwards
                # compatibility and this is the hack around it.
                optim_states['loss_scaler'] = self.optimizer.state_dict()['loss_scaler']
            elif optimstate_fp16 and not self.fp16:
                # old optimizer was fp16 but now we're doing fp32,
                # if apex, drop the fp16 wrapper from the state_dict and just load
                # the fp16 weights into the fp32 tensors
                if 'optimizer_state_dict' in optim_states:
                    # trained with apex
                    optim_states = optim_states['optimizer_state_dict']
            elif not optimstate_fp16 and self.fp16:
                # old optimizer was fp32, but now we're doing fp16.
                # this is a bit clunky, but alternatives are worse
                try:
                    self.optimizer.optimizer.load_state_dict(optim_states)
                except ValueError:
                    warn_once(
                        'WARNING: not loading optim state since model params changed.'
                    )
                return
            else:
                # previously trained in fp32, loading in fp32.
                # no special treatment needed.
                pass

            # finally, try to actually load the optimizer state
            try:
                self.optimizer.load_state_dict(optim_states)
            except (ValueError, KeyError):
                warn_once(
                    'WARNING: not loading optim state since model params changed.'
                )

    def build_lr_scheduler(self, states=None, hard_reset=False):
        """
        Create the learning rate scheduler, and assign it to self.scheduler. This
        scheduler will be updated upon a call to receive_metrics. May also create
        self.warmup_scheduler, if appropriate.

        :param state_dict states: Possible state_dict provided by model
            checkpoint, for restoring LR state
        :param bool hard_reset: If true, the LR scheduler should ignore the
            state dictionary.
        """
        if states is None:
            states = {}
        optimizer = self.optimizer
        if self.fp16:
            # lr schedulers don't work with apex, they expect the "real" optimizer
            optimizer = optimizer.optimizer
        self.scheduler = ParlAILRScheduler.lr_scheduler_factory(
            self.opt, optimizer, states, hard_reset
        )
        if self.scheduler:
            self._number_training_updates = (
                self.scheduler.get_initial_number_training_updates()
            )

    def _control_local_metrics(self, enabled: bool = False, disabled: bool = False):
        """
        Used to temporarily disable local metrics.

        This is useful for things like when you need to call super(), but
        prevent the parent from recording some metric. For example, if you're
        forwarding a dummy batch or calling super() but still want to modify
        the output.

        You can compare this to torch.no_grad in its goal.
        """
        if not (enabled ^ disabled):
            raise ValueError(
                'You must provide exactly one of enabled or disabled to '
                '_control_local_metrics.'
            )
        self.__local_metrics_enabled = enabled

    def record_local_metric(self, keyname: str, values: List[Metric]):
        """
        Record an example-level metric for all items in the batch.

        Local metrics are maybe recorded anywhere within batch act. They will
        automatically be collated and returned at the end of batch_act. The
        beginning of batch_act resets these, so you may not use them during
        observe.

        Example local metrics include ppl, token_acc, any other agent-specific
        metrics.
        """
        if not self.__local_metrics_enabled:
            return
        if keyname in self._local_metrics:
            # we could relax this already
            raise KeyError(f"Already recorded metrics for {keyname}")
        self._local_metrics[keyname] = values

    def report(self):
        """
        Report metrics.

        Report includes learning rate and number of training updates.
        """
        report = self.global_metrics.report()

        # only report LR if we have a scheduler
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            report['lr'] = GlobalAverageMetric(self.optimizer.param_groups[0]['lr'])

        if self.use_cuda:
            report['gpu_mem'] = GlobalAverageMetric(self._gpu_usage())

        if is_primary_worker() and self._number_training_updates:
            # number train updates doesn't work in hogwild sadly, and should only
            # be done on the primary worker
            report['total_train_updates'] = GlobalFixedMetric(
                self._number_training_updates
            )

        return report

    def _gpu_usage(self):
        """
        Compute GPU memory usage.

        Includes both allocated and cached memory; this should be close to the
        output of nvidia-smi, but not reflect of how much is currently demanded
        by the program. It may be viewed as a rough approximation of
        worst-case-until-now.

        :return: Percent of allocated GPU memory as a fraction of available.
        """
        if not self.use_cuda:
            return None
        if self.opt['gpu'] == -1:
            # use all gpus available locally
            devices = range(torch.cuda.device_count())
        else:
            devices = [self.opt['gpu']]
        memory_avail = 0
        memory_used = 0
        for dev in devices:
            props = torch.cuda.get_device_properties(dev)
            memory_avail += props.total_memory
            memory_used += torch.cuda.memory_allocated(dev) + torch.cuda.memory_cached(
                dev
            )
        return memory_used / memory_avail

    def receive_metrics(self, metrics_dict):
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            return
        self.scheduler.valid_step(metrics_dict)

    def _get_embtype(self, emb_type):
        # set up preinitialized embeddings
        if emb_type.startswith('glove'):
            init = 'glove'
            from parlai.zoo.glove_vectors.build import download

            embs = download(self.opt.get('datapath'))
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
        if (
            self.opt['embedding_type'] == 'random'
            or not self._should_initialize_optimizer()
        ):
            # Random embedding means no copying of pretrained embeddings
            return

        embs, name = self._get_embtype(emb_type)
        cnt = 0
        for w, i in self.dict.tok2ind.items():
            if w in embs.stoi:
                vec = self._project_vec(embs.vectors[embs.stoi[w]], weight.size(1))
                weight.data[i] = vec
                cnt += 1

        if log:
            logging.info(
                f'Initialized embeddings for {cnt} tokens '
                f'({cnt / len(self.dict):.1%}) from {name}.'
            )

    def share(self):
        """
        Share fields from parent as well as useful objects in this class.

        Subclasses will likely want to share their model as well.
        """
        shared = super().share()

        if self.opt.get('numthreads', 1) > 1 and isinstance(self.metrics, dict):
            # move metrics and model to shared memory
            self.metrics = SharedTable(self.metrics)
            self.model.share_memory()
        shared['metrics'] = self.metrics
        shared['global_metrics'] = self.global_metrics.share()

        shared['dict'] = self.dict
        shared['model'] = self.model
        shared['criterion'] = self.criterion
        shared['opt'] = self.opt
        return shared

    def _add_start_end_tokens(self, vec, add_start=False, add_end=False):
        """
        Add start and end tokens to a list or tensor.
        """
        if isinstance(vec, torch.Tensor):
            if len(vec.shape) != 1:
                raise Exception('_add_start_end_tokens expects a 1D tensor')
            tensors = [vec]
            if add_start:
                tensors.insert(0, vec.new_tensor([self.START_IDX]))
            if add_end:
                tensors.append(vec.new_tensor([self.END_IDX]))
            return torch.cat(tensors, 0)
        if add_start:
            vec.insert(0, self.START_IDX)
        if add_end:
            vec.append(self.END_IDX)
        return vec

    def _v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
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
        """
        Check that vector is truncated correctly.
        """
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
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs
            obs['full_text'] = history_string
            if history_string:
                obs['text_vec'] = history.get_history_vec()

        # check truncation
        if obs.get('text_vec') is not None:
            truncated_vec = self._check_truncate(obs['text_vec'], truncate, True)
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))
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
            truncated_vec = self._check_truncate(obs[label_type + '_vec'], truncate)
            obs.force_set(label_type + '_vec', torch.LongTensor(truncated_vec))
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
            obs.force_set('label_candidates', list(obs['label_candidates']))
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

    def _pad_tensor(
        self, items: List[Union[List[int], torch.LongTensor]]
    ) -> Tuple[torch.LongTensor, List[int]]:
        """
        Create a right padded matrix from an uneven list of lists.

        Returns (padded, lengths), where padded is the padded matrix, and lengths
        is a list containing the lengths of each row.

        :param list[iter[int]] items: List of items
        :returns: (padded, lengths) tuple
        :rtype: (Tensor[int64], list[int])

        This is intentionally overridable so that models can control how
        to pad their input.
        """
        return padded_tensor(
            items,
            pad_idx=self.NULL_IDX,
            use_cuda=self.use_cuda,
            fp16friendly=self.fp16,
            device=self.opt['gpu'],
        )

    def is_valid(self, obs):
        """
        Determine if an observation is valid or not.
        """
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
        if any(ex.get('text_vec') is not None for ex in exs):
            _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
            xs, x_lens = self._pad_tensor(_xs)
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

            ys, y_lens = self._pad_tensor(label_vecs)

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
        for k, v in output.items():
            if v is None:
                continue
            for i, sub_val in zip(valid_inds, v):
                batch_reply[i][k] = sub_val

        return batch_reply

    def get_temp_history(self, observation) -> Optional[str]:
        """
        Return a string to temporarily insert into history.

        Intentionally overrideable so more complex models can insert temporary history
        strings, i.e. strings that are removed from the history after a single turn.
        """
        return None

    def observe(self, observation):
        """
        Process incoming message in preparation for producing a response.

        This includes remembering the past history of the conversation.
        """
        # TODO: Migration plan: TorchAgent currently supports being passed
        # observations as vanilla dicts for legacy interop; eventually we
        # want to remove this behavior and demand that teachers return Messages
        observation = Message(observation)

        # Sanity check everything is in order
        self._validate_observe_invariants()

        if observation.get('episode_done'):
            self.__expecting_clear_history = True
        elif 'labels' in observation or 'eval_labels' in observation:
            # make sure we note that we're expecting a reply in the future
            self.__expecting_to_reply = True

        self.observation = observation
        # Update the history using the observation.
        # We may also consider adding a temporary string to the history
        # using the `get_temp_history()` function: this string will
        # persist until it is updated.
        self.history.update_history(
            observation, temp_history=self.get_temp_history(observation)
        )
        return self.vectorize(
            observation,
            self.history,
            text_truncate=self.text_truncate,
            label_truncate=self.label_truncate,
        )

    def self_observe(self, self_message: Message) -> None:
        """
        Observe one's own utterance.

        This is used so that the agent can incorporate its own response into
        the dialogue history after a batch_act. Failure to implement this will
        result in an agent that cannot hear itself speak.

        :param self_message:
            The message corresponding to the output from batch_act.
        """
        use_reply = self.opt.get('use_reply', 'label')

        # quick check everything is in order
        self._validate_self_observe_invariants()

        assert self.observation is not None
        if self.observation['episode_done']:
            # oh this was the last example in the episode. reset the history
            self.history.reset()
            # additionally mark the last observation as invalid
            self.observation = None
            # and clear the safety check
            self.__expecting_clear_history = False
            return

        # We did reply! Safety check is good next round.
        self.__expecting_to_reply = False

        # actually ingest the label
        if use_reply == 'none':
            # we're not including our own responses anyway.
            return
        elif use_reply == 'label':
            # first look for the true label
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
                self.history.add_reply(last_reply)
                return
            # you might expect a hard failure here, but in interactive mode we'll
            # never get a label

        # otherwise, we use the last output the model generated
        if self_message is not None:
            last_reply = self_message['text']
            self.history.add_reply(last_reply)
            return

        raise RuntimeError("Unexpected case in self_observe.")

    def _validate_observe_invariants(self):
        """
        Check that we properly called self_observe after the last batch_act.
        """
        if self.__expecting_to_reply:
            raise RuntimeError(
                "Last observe() had a label, but no call to self_observe ever "
                "happened. You are likely making multiple observe() calls without "
                "a corresponding act(). This was changed in #2043. File a GitHub "
                "issue if you require assistance."
            )

        if self.__expecting_clear_history:
            raise RuntimeError(
                "Last observe() was episode_done, but we never saw a corresponding "
                "self_observe to clear the history, probably because you missed an "
                "act(). This was changed in #2043. File a GitHub issue if you require "
                "assistance."
            )

    def _validate_self_observe_invariants(self):
        """
        Check some invariant conditions for self_observe.

        Goal is to catch potential places where we forget to call self_observe.
        """
        if self.observation is None:
            raise RuntimeError(
                "You're self_observing without having observed something. Check if "
                "you're missing a step in your observe/act/self_observe loop."
            )

        if self.observation['episode_done']:
            if not self.__expecting_clear_history:
                raise RuntimeError(
                    "You probably overrode observe() without implementing calling "
                    "super().observe(). This is unexpected. *If you must* avoid the "
                    "super call, then you should file a GitHub issue referencing "
                    "#2043."
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
                states['lr_scheduler'] = self.scheduler.get_state_dict()
                states['lr_scheduler_type'] = self.opt['lr_scheduler']
                states['warmup_scheduler'] = self.scheduler.get_warmup_state_dict()

        return states

    def save(self, path=None):
        """
        Save model parameters to path (or default to model_file arg).

        Please try to refrain from overriding this function, and instead override
        `state_dict(self)` for more specific saving.
        """
        path = self.opt.get('model_file', None) if path is None else path

        if path:
            model_dict_path = path + '.dict'
            if hasattr(self, 'dict') and not os.path.exists(
                model_dict_path
            ):  # force save dictionary
                # TODO: Look into possibly overriding opt('dict_file') with new path
                logging.debug(f'Saving dictionary to {model_dict_path}')
                self.dict.save(model_dict_path, sort=False)
            states = self.state_dict()
            if states:  # anything found to save?
                with open(path, 'wb') as write:
                    torch.save(states, write)
                # save opt file
                if hasattr(self, 'model_version'):
                    self.opt['model_version'] = self.model_version()
                self.opt.save(path + '.opt')

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as msg:
            msg_ = str(msg)
            if 'size mismatch' in msg_ and 'embedding' in msg_:
                raise RuntimeError(
                    f'{msg_}\n'
                    '-----------------\n'
                    'Could not load the model due to a size mismatch in the '
                    'embeddings. A common reason for this is trying to load '
                    'a model trained with fp16 but loaded without fp16. Try '
                    'adding --fp16 true or --force-fp16-tokens true.'
                )
            else:
                raise

    def load(self, path: str) -> Dict[str, Any]:
        """
        Return opt and model states.

        Override this method for more specific loading.
        """
        import parlai.utils.pickle

        states = torch.load(
            path, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
        )
        if 'model' in states:
            self.load_state_dict(states['model'])
        if 'optimizer' in states and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(states['optimizer'])
        return states

    @classmethod
    def upgrade_opt(cls, opt_from_disk: Opt):
        # call the parent upgrades
        opt_from_disk = super(TorchAgent, cls).upgrade_opt(opt_from_disk)

        if opt_from_disk.get('fp16'):
            # 2020-01-28 If the model was trained with fp16, we might not have saved
            # the dict with the special fp16 tokens (https://git.io/Jvm7N), IF the
            # dict was built the same time as the model. We set this to tell the
            # model it MUST add the fp16 tokens, even if it's not fp16 mode now.
            opt_from_disk['force_fp16_tokens'] = True

        return opt_from_disk

    def reset(self):
        """
        Clear internal states.
        """
        # assumption violation trackers
        self.__expecting_clear_history = False
        self.__expecting_to_reply = False

        self.observation = None
        self.history.reset()
        self.reset_metrics()

    def reset_metrics(self):
        """
        Reset all TorchAgentMetrics.
        """
        super().reset_metrics()
        self.global_metrics.clear()

    def act(self):
        """
        Call batch_act with the singleton batch.
        """
        # BatchWorld handles calling self_observe, but we're in a Hogwild or Interactive
        # world, so we need to handle this ourselves.
        response = self.batch_act([self.observation])[0]
        self.self_observe(response)
        return response

    def batch_act(self, observations):
        """
        Process a batch of observations (batchsize list of message dicts).

        These observations have been preprocessed by the observe method.

        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        # clear local metrics before anything else
        self._local_metrics.clear()

        # initialize a list of replies with this agent's id
        batch_reply = [
            Message({'id': self.getID(), 'episode_done': False}) for _ in observations
        ]

        # check if there are any labels available, if so we will train on them
        self.is_training = any('labels' in obs for obs in observations)

        # create a batch from the vectors
        batch = self.batchify(observations)

        if (
            'label_vec' in batch
            and 'text_vec' in batch
            and batch.label_vec is not None
            and batch.text_vec is not None
        ):
            # tokens per batch
            # we divide by the binary is_primary_worker() so that the numerator is
            # num_tokens in all workers, and the denominator is 1.
            tpb = GlobalAverageMetric(
                (batch.label_vec != self.NULL_IDX).sum().item(),
                float(is_primary_worker()),
            )
            self.global_metrics.add('tpb', tpb)

        if self.is_training:
            output = self.train_step(batch)
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back gradients.
                output = self.eval_step(batch)

        if output is not None:
            # local metrics are automatically matched up
            self.match_batch(batch_reply, batch.valid_indices, output)

        # broadcast the metrics back
        for k, values in self._local_metrics.items():
            if len(values) != len(batch.valid_indices):
                raise IndexError(
                    f"Batchsize mismatch on metric {k} (got {len(values)}, "
                    f"expected {len(batch.valid_indices)}"
                )
            for i, value in zip(batch.valid_indices, values):
                if 'metrics' not in batch_reply[i]:
                    batch_reply[i]['metrics'] = {}
                batch_reply[i]['metrics'][k] = value

        # Make sure we push all the metrics to main thread in hogwild/workers
        self.global_metrics.flush()

        return batch_reply

    @abstractmethod
    def train_step(self, batch):
        """
        [Abstract] Process one batch with training labels.
        """
        pass

    @abstractmethod
    def eval_step(self, batch):
        """
        [Abstract] Process one batch but do not train on it.
        """
        pass

    def set_interactive_mode(self, mode, shared):
        """
        Set interactive mode on or off.
        """
        if shared is None and mode:
            # Only print in the non-shared version.
            logging.info(f'{self.id}: full interactive mode on.')

    def backward(self, loss):
        """
        Perform a backward pass.

        It is recommended you use this instead of loss.backward(), for integration with
        distributed training and FP16 training.
        """
        update_freq = self.opt.get('update_freq', 1)

        if update_freq > 1:
            # gradient accumulation, but still need to average across the minibatches
            loss = loss / update_freq
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq

            # we're doing gradient accumulation, so we don't need to sync gradients
            # amoung GPUs
            if self._number_grad_accum != 0 and is_distributed():
                # accumulate without syncing
                with self.model.no_sync():
                    if self.fp16:
                        self.optimizer.backward(loss, update_master_grads=False)
                    else:
                        loss.backward()
                return

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
            # self._number_grad_accum is updated in backward function
            if self._number_grad_accum != 0:
                return

        if self.fp16:
            # we've been accumulating grads in fp16 and delaying the fp32 copy update.
            # finally time to perform the update.
            self.optimizer.update_master_grads()

        if self.opt.get('gradient_clip', -1) > 0:
            if self.fp16:
                grad_norm = self.optimizer.clip_master_grads(self.opt['gradient_clip'])
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.opt['gradient_clip']
                )
            self.global_metrics.add('gnorm', GlobalAverageMetric(grad_norm))
            self.global_metrics.add(
                'clip',
                GlobalAverageMetric(float(grad_norm > self.opt['gradient_clip'])),
            )
        else:
            parameters = self.model.parameters()
            grad_norm = compute_grad_norm(parameters)
            self.global_metrics.add('gnorm', GlobalAverageMetric(grad_norm))

        if self.fp16:
            self.global_metrics.add(
                'fp16_loss_scalar', GlobalAverageMetric(self.optimizer.loss_scale)
            )

        self.optimizer.step()

        # keep track up number of steps, compute warmup factor
        self._number_training_updates += 1

        # in distributed mode, all workers step together, but we need to only
        # count it once. Only the primary worker gets to make the count
        if is_primary_worker():
            self.global_metrics.add('updates', GlobalSumMetric(1))

        if getattr(self, 'scheduler', None):
            self.scheduler.step(self._number_training_updates)

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles gradient
        accumulation if agent is called with --update-freq.
        """
        if self._number_grad_accum != 0:
            # if we're accumulating gradients, don't actually zero things out yet.
            return

        self.optimizer.zero_grad()
