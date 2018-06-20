# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""
    (NOTE: To use this class, please follow the tutorial here:
    http://parl.ai/static/docs/tutorial_worlds.html#multiprocessed-pytorch-dataloader)

"""
from .teachers import FixedDialogTeacher
from parlai.scripts.build_pytorch_data import build_data
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
    raise ImportError('Need to install Pytorch: go to pytorch.org')
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
    new_batch = []
    for b in batch:
        idx = b[0]
        if type(b[1]) is list:
            ep = b[1][0]
        else:
            ep = b[1]
        new_batch.append((idx, ep))
    return new_batch


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

    def __init__(self, opt, shared=None):
        opt['batch_sort'] = False
        super().__init__(opt, shared)
        self.use_batch_act = self.bsz > 1
        self.num_workers = opt['numworkers']
        self.batch_cache_type = opt.get('batch_sort_cache')
        # One can specify a collate function to use for preparing a batch
        self.opt = copy.deepcopy(opt)
        self.is_shared = shared is not None
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
            ``-pytd vqa_v1:VQADataset``

            Note that if the dataset is named ``DefaultDataset``, then you do
            not need to specify its name following the colon; e.g., it
            would just be:
            ``-pytd vqa_v1``
        """
        dataset_name = opt.get('pytorch_teacher_dataset')
        if not dataset_name:
            return StreamDataset, default_collate
        sp = dataset_name.strip()
        repo = 'parlai'
        if sp.startswith('internal:'):
            # To switch to local repo, useful for non-public projects
            # (make a directory called 'parlai_internal' with your private agents)
            repo = 'parlai_internal'
            sp = sp[9:]
        sp = sp.split(':')
        if '.' in sp[0]:
            module_name = sp[0]
        else:
            dataset = sp[0].lower()
            module_name = '{}.tasks.{}.agents'.format(repo, dataset)
        if len(sp) > 1:
            sp[1] = sp[1][0].upper() + sp[1][1:]
            dataset = sp[1]
            if '.' not in sp[0] and 'Dataset' not in dataset:
                # Reformat from underscore to CamelCase and append "Dataset" to
                # class name by default if a complete path is not given.
                words = dataset.split('_')
                teacher_name = ''
                for w in words:
                    teacher_name += (w[0].upper() + w[1:])
                dataset = teacher_name + 'Dataset'
        else:
            dataset = 'DefaultDataset'
        my_module = importlib.import_module(module_name)
        dataset_class = getattr(my_module, dataset)

        collate = default_collate
        if hasattr(dataset_class, 'collate'):
            collate = dataset_class.collate
        elif opt.get('model', False):
            agent_class = get_agent_module(opt.get('model'))
            if hasattr(agent_class, 'collate'):
                collate = agent_class.collate
        return dataset_class, collate

    def reset(self):
        """Reset the dialog so that it is at the start of the epoch,
        and all metrics are reset.
        """
        super().reset()
        self.reset_data()

    def reset_data(self):
        if not self.training and not self.is_shared:
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
