# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""This module provides a teacher that utilizes a pytorch `DataLoader` for
    data loading. The class assumes training will be performed on streaming
    data (i.e. it will not be loaded into memory).
    It contains the following classes:

    ``StreamDataset`` - a pytorch dataset that provides streaming iteration
    through data. Requires that the dataset be built in the appropriate format
    (as observation dicts serialized in JSON format). If not built yet, the
    dataset builds the data and loads it.

    ``PytorchDataTeacher`` - a teacher that utilizes a pytorch DataLoader
    for quick batch loading.
        - In order to use the PytorchDataTeacher, the data must be built
          using build_data from examples/build_pytorch_data. This process
          happens automatically, and requires one of the following:
            - `--datafile` set to the either the built .pytorch data file
                or the data file used to build the pytorch data file
            - `--pytorch-buildteacher` set to the task teacher that will be/was used
                to build the pytorch data (by passing observations to the agent)
        - If building the dictionary for the first time, please specify
          the `--pytorch-buildteacher` so that the dictionary can be built appropriately

    Briefly, to use the PytorchDataTeacher, specify `-t pytorch_teacher`
    when training.

    The following is a more in-depth explanation for PytorchDataTeacher usage;
    i.e., to use the PytorchDataTeacher, one must do the following:

    1. Ensure that an appropriate teacher exists that can read the data
       currently saved and produce an action dict for an agent (this will be
       the `pytorch-buildteacher`)
    2. Build the data so that it can be used by the PytorchDataTeacher
        - This can be accomplished in 2 ways:
            1. Specify a `pytorch-buildteacher`, `datafile` (where the data currently
               is, and what will be used to build the data), and `datatype`
               (e.g. train, valid, etc) when calling either `build_pytorch_data`
               or calling `train_model.py`. If one is training a model, the data
                will be built automatically in `train_model.py`.
            2. Implement the `pytorch-buildteacher` such that it saves the appropriate
               datafile in its `datafile` attribute (i.e. `self.datafile`) given
               the datatype, and then specify the `pytorch-buildteacher` when calling
               either `build_pytorch_data.py` or `train_model.py`

    Additionally, if `pytorch-preprocess` is set to `True`, then the model specified
    in the command line params will have its `observe` function called on the
    `pytorch-buildteacher`'s action, and the data will be saved for that model
    specifically.

    Here's an example of what would need to be done for `bAbI` 10k task 1,
    with preprocessed data from the `seq2seq` model.

    1. Implement a normal `bAbI` teacher that can read the data in its current
       format and create an action dict (this currently exists as the
       `Task1kTeacher`)
    2. If `Task1kTeacher` saves the datafile in it's attributes, use one of the
       following 2 commands:
       - `python examples/train_model.py -t pytorch_teacher --pytorch-buildteacher \
         babi:task10k:1 -m seq2seq -mf /tmp/pytorch_data_build --pytorch-preprocess 1`
            - if one would like to train the model after building the data
       - `python examples/build_pytorch_data.py -m seq2seq \
         --pytorch-buildteacher babi:task10k:1 --pytorch-preprocess 1`
    3. If you would like to specify a specific datafile to build, e.g. a
       validation file, you could do either of the following:
       - `python examples/train_model.py -t pytorch_teacher --pytorch-buildteacher \
         babi:task10k:1 --datafile data/bAbI/tasks_1-20_v1-2/en-valid-10k-nosf/qa1_valid.txt \
         -dt valid -m seq2seq -mf /tmp/pytorch_data_build --pytorch-preprocess 1`
       - `python examples/build_pytorch_data.py -m seq2seq \
         --pytorch-buildteacher babi:task10k:1 --pytorch-preprocess 1 \
         --datafile data/bAbI/tasks_1-20_v1-2/en-valid-10k-nosf/qa1_valid.txt`

"""
from .teachers import FixedDialogTeacher
from examples.build_pytorch_data import build_data
from .agents import get_agent_module
import json
import math
import random
from functools import wraps
import importlib
import copy
try:
    import torch
except Exception as e:
    raise ModuleNotFoundError('Need to install Pytorch: go to pytorch.org')
from torch.utils.data import Dataset, DataLoader, sampler
from torch.multiprocessing import Lock, Value
import ctypes
from threading import Thread, Condition, RLock


'''
    Maps episode length to dictionary with following keys:
        current_idx: which episode in the list are we at (if simply indexing
            into list)
        ep_list: list of episodes of the length of the key
        bucket_complete: if there are no more episodes left to consider in
            the bucket
'''
length_to_eps = {}                                # Maps episode length to list
                                                  # of episodes
batches = []                                      # List of batches if popping
                                                  # batches
load_complete = Value(ctypes.c_bool, False)       # If all episodes have been
                                                  # loaded into memory
batches_lock = Lock()                             # Lock to access batches
cache_lock = Lock()                               # Lock to access length_to_eps
fill_cache_lock = RLock()                         # Lock for condition variables
add_to_cache_cv = Condition(lock=fill_cache_lock) # Condition notifying Loader
                                                  # to add to cache
cache_filled_cv = Condition(lock=fill_cache_lock) # Condition notifying teacher
                                                  # that cache has episodes


def batch_cache(function):
    max_cache_size = 10000                   # Max unseen eps
    min_cache_size = 1000                    # Min unseen eps

    def get_cache_size():
        '''Returns number of available episodes '''
        return sum(len(v['ep_list']) - v['current_idx']for k, v in length_to_eps.items())

    def get_available_buckets(bsz):
        '''Returns buckets where there are enough episodes for a batch'''
        if load_complete.value:
            return {k: v for k, v in length_to_eps.items() if not v['bucket_complete'] or len(v['ep_list']) - v['current_idx'] > 0}
        else:
            return {k: v for k, v in length_to_eps.items() if len(v['ep_list']) - v['current_idx'] >= bsz}

    def reset():
        '''Resets the indices into the buckets'''
        with cache_lock:
            for idx in length_to_eps:
                length_to_eps[idx]['current_idx'] = 0
                length_to_eps[idx]['bucket_complete'] = False

    def consolidate(caller):
        '''Consolidate remaining episodes into batches'''
        load_complete.value = True
        bsz = caller.bsz
        batch = []
        sorted_lengths = sorted(length_to_eps.keys())
        with cache_lock:
            if caller.batch_cache_type == 'index':
                for length in sorted_lengths:
                    current_idx = length_to_eps[length]['current_idx']
                    ep_list = length_to_eps[length]['ep_list']
                    unseen_eps = ep_list[current_idx:]
                    length_to_eps[length]['ep_list'] = ep_list[:current_idx]
                    batch = unseen_eps + batch
                    while len(batch) >= bsz:
                        length_to_eps[length]['ep_list'] += batch[:bsz]
                        batch = batch[bsz:]
                if len(batch) > 0:
                    length_to_eps[-1] = {
                        'current_idx': 0,
                        'ep_list': batch,
                        'bucket_complete': False
                    }
            elif caller.batch_cache_type == 'pop':
                for length in sorted_lengths:
                    batch += length_to_eps[length]['ep_list']
                with batches_lock:
                    while len(batch) >= bsz:
                        batches.append(batch[:bsz])
                        batch = batch[bsz:]
                if len(batch) > 0:
                    with batches_lock:
                        batches.append(batch)

    def flatten(l):
        '''Helper function for flattening a list'''
        return [item for sublist in l for item in sublist]

    def put_in_cache(ep_idx, episode, caller):
        '''Put episode `ep_idx` into cache'''
        length = episode['text'].count(' ')
        lengths = [length] + flatten([[length + i, length + (i * -1)] for i in range(1, caller.batch_length_range)])
        lengths = [max(i, 1) for i in lengths]
        in_cache = False
        for l in lengths:
            if l in length_to_eps:
                with cache_lock:
                    length_to_eps[l]['ep_list'] += [(ep_idx, episode)]
                in_cache = True
                break
        if not in_cache:
            with cache_lock:
                length_to_eps[length] = {
                    'current_idx': 0,
                    'ep_list': [(ep_idx, episode)],
                    'bucket_complete': False
                }
        if ep_idx == caller.dataset.num_episodes() - 1:
            consolidate(caller)
            with add_to_cache_cv:
                cache_filled_cv.notify_all()

    @wraps(function)
    def wrapper(*args):
        caller = args[0]
        batch_cache_type = caller.batch_cache_type
        bsz = caller.bsz
        if batch_cache_type == 'none' or not caller.datatype.startswith('train'):
            return function(*args)
        # If Loader, put episodes in cache
        if isinstance(caller, LoaderProcess):
            with add_to_cache_cv:
                while get_cache_size() >= max_cache_size and len(get_available_buckets(bsz)) > 0:
                    cache_filled_cv.notify_all()
                    add_to_cache_cv.wait()
            idx_and_batch = function(*args)
            if idx_and_batch is None:
                return None
            for ep_index, ep in idx_and_batch[1]:
                put_in_cache(ep_index, ep, caller)
            return idx_and_batch
        # If teacher, return batch of episodes
        else:
            teacher = caller
            num_batches = teacher.num_batches
            while True:
                with cache_filled_cv:
                    while (not load_complete.value and
                        (get_cache_size() <= min_cache_size or len(get_available_buckets(bsz)) == 0)):
                        add_to_cache_cv.notify()
                        cache_filled_cv.wait()
                        available_buckets = get_available_buckets(bsz)
                if load_complete.value and batch_cache_type == 'pop':
                    return teacher.batch_idx + 1, random.choice(batches)
                batch = None
                available_buckets = get_available_buckets(bsz)
                if len(available_buckets) != 0:
                    # Pick length index at random
                    length = random.choice(list(available_buckets.keys()))
                    with cache_lock:
                        current_idx = length_to_eps[length]['current_idx']
                        ep_list = length_to_eps[length]['ep_list']
                        num_eps = len(ep_list)
                        if num_eps - current_idx >= bsz:
                            if batch_cache_type == 'pop':
                                batch = ep_list[:bsz]
                                length_to_eps[length]['ep_list'] = ep_list[bsz:]
                            else:
                                batch = ep_list[current_idx: current_idx + bsz]
                                length_to_eps[length]['current_idx'] = (current_idx + bsz)
                        elif load_complete.value and num_eps > 0:
                            if batch_cache_type == 'pop':
                                batch = ep_list
                            elif num_eps - current_idx > 0:
                                batch = ep_list[current_idx:]
                                length_to_eps[length]['current_idx'] = num_eps - 1
                            length_to_eps[length]['bucket_complete'] = True

                if batch is not None:
                    if batch_cache_type == 'pop':
                        with batches_lock:
                            batches.append(batch)
                    elif teacher.batch_idx + 1 >= num_batches:
                        reset()
                    return teacher.batch_idx + 1, batch

    return wrapper


class LoaderProcess(Thread):
    """A background process that submits jobs to the DataLoader
       to load examples into cache
    """
    def __init__(self, opt):
        super().__init__(daemon=True)
        self.dataset = opt['dataset_class'](opt)
        self.bsz = opt.get('batchsize', 1)
        self.num_workers = opt.get('num_workers', 4)
        collate_fn = opt.get('collate_fn', default_collate)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.bsz,
            shuffle=False,
            sampler=sampler.SequentialSampler(self.dataset),
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            )
        self.datatype = opt.get('datatype')
        self.data = enumerate(self.dataloader)
        self.batch_cache_type = opt.get('batch_sort_cache')
        self.batch_length_range = opt.get('batch_length_range')

    def run(self):
        while True:
            idx_and_batch = self.load_next()
            if idx_and_batch is None:
                return

    @batch_cache
    def load_next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

# Default collate function (for how to prepare a batch)
def default_collate(batch):
    return [(b[0], b[1][0]) for b in batch]


class StreamDataset(Dataset):
    """A Pytorch Dataset utilizing streaming"""
    def __init__(self, opt):
        self.opt = opt
        self.datatype = opt.get('datatype')
        self.datafile = build_data(self.opt)
        self.data_gen = self._data_generator(self.datafile)
        self.length_datafile = self.datafile + ".length"
        self.num_epochs = self.opt.get('num_epochs', 0)
        self.training = self.datatype.startswith('train')
        self._load_lens()

    def __getitem__(self, index):
        while True:
            index %= self.num_episodes()
            idx, ep = next(self.data_gen)
            if idx == index:
                return (index, ep)

    def __len__(self):
        num_epochs = self.num_epochs if self.num_epochs > 0 else 1000
        num_iters = num_epochs if self.training else 1
        return int(num_iters * self.num_episodes())

    def _load_lens(self):
        with open(self.length_datafile) as length:
            lengths = json.load(length)
            self.num_eps = lengths['num_eps']
            self.num_exs = lengths['num_exs']

    def _data_generator(self, datafile):
        while True:
            for idx, episode in self._read_episode(self.datafile):
                yield idx, episode

    def _read_episode(self, datafile):
        read = open(datafile)
        episode = []
        for idx, line in enumerate(read):
            example = json.loads(line)
            episode.append(example)
            if example['episode_done']:
                yield idx, episode
                episode = []
        read.close()

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs


class PytorchDataTeacher(FixedDialogTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        arg_group = argparser.add_argument_group('PytorchData Arguments')
        arg_group.add_argument('--datafile', type=str, default='',
            help='datafile for pytorch data loader')
        arg_group.add_argument('-nw', '--numworkers', type=int, default=4,
            help='how many workers the Pytorch dataloader should use')
        arg_group.add_argument('--pytorch-buildteacher', type=str, default='',
            help='Which teacher to use when building the pytorch data')
        arg_group.add_argument('--pytorch-preprocess', type='bool', default=False,
            help='Whether the agent should preprocess the data while building'
                 'the pytorch data')
        arg_group.add_argument('--batch-sort-cache', type=str,
            choices=['pop', 'index', 'none'], default='none',
            help='Whether to have batches of similarly sized episodes, and how'
            'to build up the cache')
        arg_group.add_argument('--batch-length-range', type=int, default=5,
            help='degree of variation of size allowed in batch')
        arg_group.add_argument('--dataset', type=str, default='StreamDataset',
            help='which dataset to use in dataloader')

    def __init__(self, opt, shared=None):
        opt['batch_sort'] = False
        super().__init__(opt, shared)
        self.use_batch_act = self.bsz > 1
        self.num_workers = opt['numworkers']
        self.batch_cache_type = opt.get('batch_sort_cache')
        # One can specify a collate function to use for preparing a batch
        self.opt = copy.deepcopy(opt)
        dataset_class, self.collate_fn = self.get_dataset_class(opt)
        opt['dataset_class'] = dataset_class
        opt['collate_fn'] = self.collate_fn

        if not shared:
            self.dataset = dataset_class(opt)
            if self.datatype == 'train' and not isinstance(self.dataset, StreamDataset):
                data_sampler = sampler.RandomSampler(self.dataset)
            else:
                data_sampler = sampler.SequentialSampler(self.dataset)
            pin_memory = not isinstance(self.dataset, StreamDataset)
            self.pytorch_dataloader = DataLoader(
                self.dataset,
                batch_size=self.bsz,
                shuffle=False,
                sampler=data_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=pin_memory,
                drop_last=False,
                )
            self.lastYs = [None] * self.bsz
            if self.batch_cache_type != 'none':
                self.loader_process = LoaderProcess(opt)
                self.loader_process.start()
            self.data = enumerate(self.pytorch_dataloader)
        else:
            self.dataset = shared['dataset']
            self.pytorch_dataloader = shared['pytorch_dataloader']
            self.lastYs = shared['lastYs']
            self.data = shared['data']

        self.num_batches = math.ceil(self.dataset.num_episodes()/self.bsz)
        self.reset()

    def get_dataset_class(self, opt):
        """ To use a custom dataset (as opposed to the StreamDataset above),
            you can subclass the pytorch Dataset class and specify its
            location on the command line.

            For example, the VQA v1 task provides a custom dataset, which can
            be specified on the command line as follows:
            ``--dataset parlai.tasks.vqa_v1.agents:VQADataset``

            Note that if the dataset is named ``DefaultDataset``, then you do
            not need to specify its name following the colon; e.g., it
            would just be:
            ``--dataset parlai.tasks.vqa_v1.agents``
        """
        dataset_name = opt.get('dataset')
        sp = dataset_name.strip().split(':')
        agent_class = get_agent_module(opt.get('model'))
        if hasattr(agent_class, 'collate'):
            collate = agent_class.collate
        else:
            collate = default_collate
        if sp[0] == 'StreamDataset':
            return StreamDataset, collate
        module_name = sp[0]
        if len(sp) > 1:
            dataset = sp[1]
        else:
            dataset = 'DefaultDataset'
        my_module = importlib.import_module(module_name)
        return getattr(my_module, dataset), collate

    def reset(self):
        """Reset the dialog so that it is at the start of the epoch,
        and all metrics are reset.
        """
        super().reset()
        self.reset_data()

    def reset_data(self):
        self.lastY = None
        self.epochDone = False
        self.episode = None
        self.episode_done = True
        self.episode_idx = 0
        self.batch_idx = 0

    def share(self):
        shared = super().share()
        shared['pytorch_dataloader'] = self.pytorch_dataloader
        shared['dataset'] = self.dataset
        shared['data'] = self.data
        return shared

    def next_example(self):
        if self.epochDone:
            if not self.training:
                return {'episode_done': True, 'id': self.getID()}, True
            else:
                # Reset the data because it is streaming data
                self.reset_data()
        if self.episode_done:
            try:
                self.episode_idx, self.episode = next(self.data)
                self.entry_idx = 0
                epoch_done = False
            except StopIteration:
                ex = {'episode_done': True, 'id': self.getID()}
                epoch_done = True
        else:
            self.entry_idx += 1

        if not epoch_done:
            if self.collate_fn == default_collate:
                self.episode[self.entry_idx] = self.episode[self.entry_idx][1]
            ex = self.episode[self.entry_idx]
            self.episode_done = ex['episode_done']
            if (self.episode_done
                    and self.episode_idx + self.bsz >= self.num_episodes()):
                epoch_done = True
        return ex, epoch_done

    @batch_cache
    def get_next_batch(self):
        # employs a cache to see if there is a batch of equal size ready
        return next(self.data)

    def next_batch(self):
        if self.epochDone:
            if not self.training:
                return [{'episode_done': True, 'id': self.getID()}] * self.bsz
            else:
                # Reset the data because it is streaming data
                self.reset_data()
        try:
            self.batch_idx, batch = self.get_next_batch()
            if self.collate_fn == default_collate:
                batch = [b[1] for b in batch]
            epoch_done = False
        except StopIteration:
            batch = [{'episode_done': True, 'id': self.getID()}] * self.bsz
            epoch_done = True
        if not epoch_done and self.batch_idx == self.num_batches:
            epoch_done = True
        self.epochDone = epoch_done
        return batch

    def num_episodes(self):
        """Get the number of episodes in this dataset."""
        return self.dataset.num_episodes()

    def num_examples(self):
        """Get the total number of examples in this dataset."""
        return self.dataset.num_examples()

    def act(self):
        """Send new dialog message."""
        action = super().act()
        self.lastY = action.get('labels', action.get('eval_labels', None))
        return action

class DefaultTeacher(PytorchDataTeacher):
    pass
