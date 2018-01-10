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
            - `--pytorch_buildteacher` set to the task teacher that will be/was used
                to build the pytorch data (by passing observations to the agent)
        - If building the dictionary for the first time, please specify
          the `--pytorch_buildteacher` so that the dictionary can be built appropriately

    Briefly, to use the PytorchDataTeacher, specify `-t pytorch_teacher`
    when training.

    The following is a more in-depth explanation for PytorchDataTeacher usage;
    i.e., to use the PytorchDataTeacher, one must do the following:

    1. Ensure that an appropriate teacher exists that can read the data
       currently saved and produce an action dict for an agent (this will be
       the `pytorch_buildteacher`)
    2. Build the data so that it can be used by the PytorchDataTeacher
        - This can be accomplished in 2 ways:
            1. Specify a `pytorch_buildteacher`, `datafile` (where the data currently
               is, and what will be used to build the data), and `datatype`
               (e.g. train, valid, etc) when calling either `build_pytorch_data`
               or calling `train_model.py`. If one is training a model, the data
                will be built automatically in `train_model.py`.
            2. Implement the `pytorch_buildteacher` such that it saves the appropriate
               datafile in its `datafile` attribute (i.e. `self.datafile`) given
               the datatype, and then specify the `pytorch_buildteacher` when calling
               either `build_pytorch_data.py` or `train_model.py`

    Additionally, if `pytorch_preprocess` is set to `True`, then the model specified
    in the command line params will have its `observe` function called on the
    `pytorch_buildteacher`'s action, and the data will be saved for that model
    specifically.

    Here's an example of what would need to be done for `bAbI` 10k task 1,
    with preprocessed data from the `seq2seq` model.

    1. Implement a normal `bAbI` teacher that can read the data in its current
       format and create an action dict (this currently exists as the
       `Task1kTeacher`)
    2. If `Task1kTeacher` saves the datafile in it's attributes, use one of the
       following 2 commands:
       - `python examples/train_model.py -t pytorch_teacher --pytorch_buildteacher \
         babi:task10k:1 -m seq2seq -mf /tmp/pytorch_data_build --pytorch_preprocess 1`
            - if one would like to train the model after building the data
       - `python examples/build_pytorch_data.py -m seq2seq \
         --pytorch_buildteacher babi:task10k:1 --pytorch_preprocess 1`
    3. If you would like to specify a specific datafile to build, e.g. a
       validation file, you could do either of the following:
       - `python examples/train_model.py -t pytorch_teacher --pytorch_buildteacher \
         babi:task10k:1 --datafile data/bAbI/tasks_1-20_v1-2/en-valid-10k-nosf/qa1_valid.txt \
         -dt valid -m seq2seq -mf /tmp/pytorch_data_build --pytorch_preprocess 1`
       - `python examples/build_pytorch_data.py -m seq2seq \
         --pytorch_buildteacher babi:task10k:1 --pytorch_preprocess 1 \
         --datafile data/bAbI/tasks_1-20_v1-2/en-valid-10k-nosf/qa1_valid.txt`

"""
from .teachers import FixedDialogTeacher
from examples.build_pytorch_data import build_data

import json
import math
import random
from functools import wraps
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
            return {k: v for k, v in length_to_eps.items() if not v['bucket_complete']}
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
                    ep_list = length_to_eps[length]['ep_list']
                    if len(ep_list) % bsz != 0:
                        extra_ep_index = int(len(ep_list)/bsz) * bsz
                        batch += ep_list[extra_ep_index:]
                        length_to_eps[length]['ep_list'] = ep_list[:extra_ep_index]
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
                    while len(batch) >= bsz:
                        with batches_lock:
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

    @wraps(function)
    def wrapper(*args):
        caller = args[0]
        batch_cache_type = caller.batch_cache_type
        bsz = caller.bsz
        if batch_cache_type == 'none':
            return function(*args)
        # If Loader, put episodes in cache
        if isinstance(caller, LoaderProcess):
            with add_to_cache_cv:
                counter = 0
                while get_cache_size() >= max_cache_size and len(get_available_buckets(bsz)) >= bsz:
                    cache_filled_cv.notify_all()
                    counter += 1
                    if counter < 64:
                        print("notified teachers")

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
                if batch_cache_type == 'pop' and len(batches) == num_batches:
                    batch = random.choice(batches)
                    return teacher.batch_idx + 1, batch
                available_buckets = get_available_buckets(bsz)
                with cache_filled_cv:
                    counter = 0
                    while get_cache_size() <= min_cache_size or len(available_buckets) == 0:
                        add_to_cache_cv.notify()
                        counter += 1
                        if counter < 64 == 0:
                            print("notified loader")
                        cache_filled_cv.wait()
                        available_buckets = get_available_buckets(bsz)

                batch = None
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
                            elif num_eps - current_idx >= 0:
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
        self.dataset = StreamDataset(opt)
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
        self.data = enumerate(self.dataloader)
        self.batch_cache_type = opt.get('batch_cache')
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
        self.datafile = build_data(self.opt)
        self.data_gen = self._data_generator(self.datafile)
        self.length_datafile = self.datafile + ".length"
        self._load_lens()

    def __getitem__(self, index):
        while True:
            idx, ep = next(self.data_gen)
            if idx == index:
                return (index, ep)

    def __len__(self):
        return self.num_eps

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
        arg_group.add_argument('--pytorch_buildteacher', type=str, default='',
            help='Which teacher to use when building the pytorch data')
        arg_group.add_argument('--pytorch_preprocess', type='bool', default=True,
            help='Whether the agent should preprocess the data while building'
                 'the pytorch data')
        arg_group.add_argument('--batch_cache', type=str,
            choices=['pop', 'index', 'none'], default='none',
            help='Whether to build up cache of batches of similar size')
        arg_group.add_argument('--batch_length_range', type=int, default=5,
            help='degree of variation of size allowed in batch')

    def __init__(self, opt, shared=None):
        opt['batch_sort'] = False
        super().__init__(opt, shared)
        self.use_batch_act = self.bsz > 1
        self.num_workers = opt['numworkers']
        self.batch_cache_type = opt.get('batch_cache')
        # One can specify a collate function to use for preparing a batch
        collate_fn = opt.get('collate_fn', default_collate)
        if not shared:
            self.dataset = StreamDataset(opt)
            self.pytorch_dataloader = DataLoader(
                self.dataset,
                batch_size=self.bsz,
                shuffle=False,
                sampler=sampler.SequentialSampler(self.dataset),
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=False,
                )
            self.lastYs = [None] * self.bsz
            if self.batch_cache_type != 'none':
                self.loader_process = LoaderProcess(opt)
                self.loader_process.start()
        else:
            self.dataset = shared['dataset']
            self.pytorch_dataloader = shared['pytorch_dataloader']
            self.lastYs = shared['lastYs']

        self.num_batches = math.ceil(self.dataset.num_episodes()/self.bsz)
        self.reset()

    def reset(self):
        """Reset the dialog so that it is at the start of the epoch,
        and all metrics are reset.
        """
        super().reset()
        self.reset_data()

    def reset_data(self):
        self.data = enumerate(self.pytorch_dataloader)
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
            ex = self.episode[self.entry_idx][1]
            self.episode_done = ex['episode_done']
            if (self.episode_done
                    and self.episode_idx + self.bsz >= self.num_episodes()):
                epoch_done = True

        return ex, epoch_done

    @batch_cache
    def get_next_batch(self):
        #employs a cache to see if there is a batch of equal size ready
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
