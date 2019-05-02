#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    (NOTE: To use this class, please follow the tutorial here:
    http://parl.ai/static/docs/tutorial_worlds.html#multiprocessed-pytorch-dataloader)

"""
from .teachers import FixedDialogTeacher
from parlai.core.utils import warn_once
from parlai.scripts.build_pytorch_data import build_data
from .agents import get_agent_module
import json
import math
import collections
import random
import os
from functools import wraps
import importlib
from functools import lru_cache
try:
    import torch  # noqa: F401
except ImportError:
    raise ImportError('Need to install Pytorch: go to pytorch.org')
from torch.utils.data import ConcatDataset, Dataset, DataLoader, sampler
from torch.multiprocessing import Lock, Value
import ctypes
from threading import Thread, Condition, RLock


if torch.version.__version__.startswith('0.'):
    raise ImportError(
        "Please upgrade to PyTorch >=1.0; "
        "visit https://pytorch.org for instructions."
    )


class BatchSortCache(object):
    """
        Object that encapsulates the functionality of the batch sort cache.

        Maps episode length to dictionary with following keys:
            current_idx: which episode in the list are we at (if simply indexing
                into list)
            ep_list: list of episodes of the length of the key
            bucket_complete: if there are no more episodes left to consider in
                the bucket
    """
    @classmethod
    def create(cls):
        if not hasattr(cls, 'length_to_eps'):
            # Maps episode length to list of episodes
            cls.length_to_eps = {}
            # Set of episode indices already in the cache
            cls.ep_indices = set()
            # List of batches if popping batches
            cls.batches = []
            # If all episodes have been loaded into memory
            cls.load_complete = Value(ctypes.c_bool, False)
            # Lock to access batches
            cls.batches_lock = Lock()
            # Lock to access length_to_eps
            cls.cache_lock = Lock()
            # Lock for condition variables
            cls.fill_cache_lock = RLock()
            # Condition notifying Loader to add to cache
            cls.add_to_cache_cv = Condition(lock=cls.fill_cache_lock)
            # Condition notifying teacher that cache has episodes
            cls.cache_filled_cv = Condition(lock=cls.fill_cache_lock)

    @classmethod
    def destroy(cls):
        if hasattr(cls, 'length_to_eps'):
            del cls.length_to_eps
            del cls.ep_indices
            del cls.batches
            del cls.load_complete
            del cls.batches_lock
            del cls.cache_lock
            del cls.fill_cache_lock
            del cls.add_to_cache_cv
            del cls.cache_filled_cv

    @classmethod
    def batch_cache(cls, function):
        max_cache_size = 10000  # Max unseen eps
        min_cache_size = 1000  # Min unseen eps

        def get_cache_size():
            '''Returns number of available episodes '''
            return sum(
                len(v['ep_list']) - v['current_idx']
                for k, v in cls.length_to_eps.items()
            )

        def get_available_buckets(bsz):
            '''Returns buckets where there are enough episodes for a batch'''
            if cls.load_complete.value:
                return {
                    k: v
                    for k, v in cls.length_to_eps.items()
                    if not v['bucket_complete']
                    or len(v['ep_list']) - v['current_idx'] > 0
                }
            else:
                return {
                    k: v
                    for k, v in cls.length_to_eps.items()
                    if len(v['ep_list']) - v['current_idx'] >= bsz
                }

        def reset():
            '''Resets the indices into the buckets'''
            with cls.cache_lock:
                for idx in cls.length_to_eps:
                    cls.length_to_eps[idx]['current_idx'] = 0
                    cls.length_to_eps[idx]['bucket_complete'] = False

        def consolidate(caller):
            '''Consolidate remaining episodes into batches'''
            cls.load_complete.value = True
            bsz = caller.bsz
            batch = []
            sorted_lengths = sorted(cls.length_to_eps.keys())
            with cls.cache_lock:
                if caller.batch_cache_type == 'index':
                    for length in sorted_lengths:
                        current_idx = cls.length_to_eps[length]['current_idx']
                        ep_list = cls.length_to_eps[length]['ep_list']
                        unseen_eps = ep_list[current_idx:]
                        cls.length_to_eps[length]['ep_list'] = ep_list[:current_idx]
                        batch = unseen_eps + batch
                        while len(batch) >= bsz:
                            cls.length_to_eps[length]['ep_list'] += batch[:bsz]
                            batch = batch[bsz:]
                    if len(batch) > 0:
                        cls.length_to_eps[-1] = {
                            'current_idx': 0,
                            'ep_list': batch,
                            'bucket_complete': False
                        }
                elif caller.batch_cache_type == 'pop':
                    for length in sorted_lengths:
                        batch += cls.length_to_eps[length]['ep_list']
                    with cls.batches_lock:
                        while len(batch) >= bsz:
                            cls.batches.append(batch[:bsz])
                            batch = batch[bsz:]
                    if len(batch) > 0:
                        with cls.batches_lock:
                            cls.batches.append(batch)

        def flatten(l):
            '''Helper function for flattening a list'''
            return [item for sublist in l for item in sublist]

        def put_in_cache(ep_idx, episode, caller):
            '''Put episode `ep_idx` into cache'''
            length = ep_length(episode[caller.batch_sort_field])
            lengths = [length] + flatten([
                [length + i, length + (i * -1)]
                for i in range(1, caller.batch_length_range)
            ])
            lengths = [max(i, 1) for i in lengths]
            in_cache = ep_idx in cls.ep_indices
            # first check if episode can go in existing bucket
            if not in_cache:
                for l in lengths:
                    if l in cls.length_to_eps:
                        with cls.cache_lock:
                            cls.length_to_eps[l]['ep_list'] += [(ep_idx, episode)]
                            cls.ep_indices.add(ep_idx)
                        in_cache = True
                        break
            # otherwise, make a new bucket
            if not in_cache:
                with cls.cache_lock:
                    cls.length_to_eps[length] = {
                        'current_idx': 0,
                        'ep_list': [(ep_idx, episode)],
                        'bucket_complete': False
                    }
                    cls.ep_indices.add(ep_idx)
            if ep_idx == caller.dataset.num_episodes() - 1:
                consolidate(caller)
                with cls.add_to_cache_cv:
                    cls.cache_filled_cv.notify_all()

        @wraps(function)
        def wrapper(*args):
            caller = args[0]
            batch_sort = caller.batch_sort
            batch_cache_type = caller.batch_cache_type
            bsz = caller.bsz
            if not batch_sort or not caller.datatype.startswith('train'):
                return function(*args)
            # If Loader, put episodes in cache
            if isinstance(caller, LoaderProcess):
                with cls.add_to_cache_cv:
                    while (get_cache_size() >= max_cache_size and
                            len(get_available_buckets(bsz)) > 0):
                        cls.cache_filled_cv.notify_all()
                        cls.add_to_cache_cv.wait()
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
                    with cls.cache_filled_cv:
                        while (not cls.load_complete.value and
                                (get_cache_size() <= min_cache_size or
                                    len(get_available_buckets(bsz)) == 0)):
                            cls.add_to_cache_cv.notify()
                            cls.cache_filled_cv.wait()
                            available_buckets = get_available_buckets(bsz)
                    if cls.load_complete.value and batch_cache_type == 'pop':
                        return teacher.batch_idx + 1, random.choice(cls.batches)
                    batch = None
                    available_buckets = get_available_buckets(bsz)
                    if len(available_buckets) != 0:
                        # Pick length index at random
                        length = random.choice(list(available_buckets.keys()))
                        with cls.cache_lock:
                            current_idx = cls.length_to_eps[length]['current_idx']
                            ep_list = cls.length_to_eps[length]['ep_list']
                            num_eps = len(ep_list)
                            if num_eps - current_idx >= bsz:
                                if batch_cache_type == 'pop':
                                    batch = ep_list[:bsz]
                                    cls.length_to_eps[length]['ep_list'] = ep_list[bsz:]
                                else:
                                    batch = ep_list[current_idx: current_idx + bsz]
                                    cls.length_to_eps[length]['current_idx'] = (
                                        current_idx + bsz
                                    )
                            elif cls.load_complete.value and num_eps > 0:
                                if batch_cache_type == 'pop':
                                    batch = ep_list
                                elif num_eps - current_idx > 0:
                                    batch = ep_list[current_idx:]
                                    cls.length_to_eps[length]['current_idx'] = \
                                        num_eps - 1
                                cls.length_to_eps[length]['bucket_complete'] = True

                    if batch is not None:
                        if batch_cache_type == 'pop':
                            with cls.batches_lock:
                                cls.batches.append(batch)
                        elif teacher.batch_idx + 1 >= num_batches:
                            reset()
                        return teacher.batch_idx + 1, batch

        return wrapper


def ep_length(val):
    '''Determines the length of an episode, given the specified value'''
    if isinstance(val, (int, bytes, bool)):
        return 1
    if isinstance(val, str):
        return len(val.replace('\n', ' ').split(' '))
    if isinstance(val, (collections.Mapping,
                        collections.Sequence,
                        torch.Tensor)):
        if (isinstance(val, collections.Mapping) and
                val.get('deserialized_tensor', False)):
            return len(val['value'])
        return len(val)


# Get Datasets from the options
def get_dataset_classes(opt):
    """ To use a custom dataset (as opposed to the StreamDataset or ParlAIDataset),
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
    if 'stream' in opt.get('datatype'):
        default_dataset = StreamDataset
    else:
        default_dataset = ParlAIDataset
    dataset_name = opt.get('pytorch_teacher_dataset')
    task_name = opt.get('pytorch_teacher_task')
    datasets = []
    if task_name is not None:
        datasets += [
            (default_dataset, default_collate, task)
            for task in task_name.split(',')
        ]
    if not dataset_name:
        return datasets
    sps = [d.strip() for d in dataset_name.split(',')]
    for sp in sps:
        full_task_name = sp
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
        datasets.append((dataset_class, collate, full_task_name))
    return datasets


class LoaderProcess(Thread):
    """A background process that submits jobs to the DataLoader
       to load examples into cache
    """
    def __init__(self, opt):
        super().__init__(daemon=True)
        dataset_classes = get_dataset_classes(opt)
        if len(dataset_classes) > 1:
            datasets = []
            for class_name, collate_fn, task_name in dataset_classes:
                opt['pytorch_teacher_task'] = task_name
                opt['task'] = task_name
                datasets.append(class_name(opt))
                self.collate = collate_fn
            self.dataset = ParlAIConcatDataset(datasets)
        else:
            class_name, self.collate, task_name = dataset_classes[0]
            self.dataset = class_name(opt)
        self.bsz = opt.get('batchsize', 1)
        self.num_workers = opt.get('num_workers', 4)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.bsz,
            shuffle=False,
            sampler=sampler.SequentialSampler(self.dataset),
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=False,
            drop_last=False,
        )
        self.datatype = opt.get('datatype')
        self.data = enumerate(self.dataloader)
        self.batch_sort = opt.get('pytorch_teacher_batch_sort')
        self.batch_cache_type = opt.get('batch_sort_cache_type')
        self.batch_length_range = opt.get('batch_length_range')
        self.batch_sort_field = opt.get('batch_sort_field')

    def run(self):
        while True:
            idx_and_batch = self.load_next()
            if idx_and_batch is None:
                return

    @BatchSortCache.batch_cache
    def load_next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None


"""
    Collating, deserializing, processing batches
"""
TORCH_DTYPES = [torch.float32, torch.float64, torch.float16, torch.uint8,
                torch.int8, torch.int16, torch.int32, torch.int64]
STR_TO_TORCH_DTYPE = {str(d): d for d in TORCH_DTYPES}


def default_collate(batch):
    """
        Default collate function, used for ParlAIDataset and StreamDataset
    """
    new_batch = []
    for b in batch:
        idx = b[0]
        if type(b[1]) is list:
            ep = b[1][0]
        else:
            ep = b[1]
        new_batch.append((idx, ep))
    return new_batch


def deserialize(obj):
    """
        Deserializes lists into Tensors
    """
    keys = list(obj.keys())
    for key in keys:
        if type(obj[key]) is dict and obj[key].get('deserialized_tensor', False):
            dtype = STR_TO_TORCH_DTYPE[obj[key]['type']]
            val = obj[key]['value']
            del obj[key]
            obj[key] = torch.as_tensor(val, dtype=dtype)
    return obj


def process(ex_or_batch):
    """
        Process examples/batches, i.e. deserialize if necessary
    """
    if type(ex_or_batch) is list:
        if all([ep.get('preprocessed') for ep in ex_or_batch]):
            ex_or_batch = [deserialize(ep) for ep in ex_or_batch]
    else:
        if ex_or_batch.get('preprocessed'):
            ex_or_batch = deserialize(ex_or_batch)
    return ex_or_batch


"""
    ParlAI Implementations of Pytorch Datasets
"""


class StreamDataset(Dataset):
    """A Pytorch Dataset utilizing streaming"""
    def __init__(self, opt):
        self.opt = opt
        self.datatype = opt.get('datatype')
        self.datapath = build_data(self.opt)
        self.length_datafile = os.path.join(self.datapath, 'data_length')
        self.char_index_file = os.path.join(self.datapath, 'char_index')
        self.datafile = os.path.join(self.datapath, 'data')
        self.training = self.datatype.startswith('train')
        self.ordered = ('ordered' in self.datatype or
                        ('stream' in self.datatype and not opt.get('shuffle')))
        self._load_lens()

    def __getitem__(self, index):
        if self.ordered or not self.training:
            if not hasattr(self, 'data_gen'):
                self.data_gen = self._read_episode()
            while True:
                idx, ep = next(self.data_gen)
                if idx == index:
                    return (index, ep)
        else:
            episode = []
            episode_done = False
            with open(self.datafile) as f:
                ex_offset = self.char_index[index]
                f.seek(ex_offset)
                while not episode_done:
                    example = json.loads(f.readline())
                    episode.append(example)
                    episode_done = example['episode_done']
            return (index, episode)

    def __len__(self):
        return self.num_episodes()

    def _load_lens(self):
        with open(self.length_datafile) as length:
            lengths = json.load(length)
            self.num_eps = lengths['num_eps']
            self.num_exs = lengths['num_exs']
        with open(self.char_index_file) as char:
            self.char_index = json.load(char)

    def _data_generator(self):
        while True:
            for idx, episode in self._read_episode():
                yield idx, episode

    def _read_episode(self):
        read = open(self.datafile)
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


class ParlAIDataset(Dataset):
    """A Pytorch Dataset, for random sampling"""
    def __init__(self, opt):
        self.opt = opt
        self.datatype = opt.get('datatype')
        self.datapath = build_data(self.opt)
        self.length_datafile = os.path.join(self.datapath, 'data_length')
        self.datafile = os.path.join(self.datapath, 'data')
        self.training = self.datatype.startswith('train')
        self._load_lens()
        self._setup_data()

    def __getitem__(self, index):
        return index, self.data[index]

    def __len__(self):
        return self.num_episodes()

    def _load_lens(self):
        with open(self.length_datafile) as length:
            lengths = json.load(length)
            self.num_eps = lengths['num_eps']
            self.num_exs = lengths['num_exs']

    def _setup_data(self):
        self.data = []
        with open(self.datafile) as f:
            for line in f:
                self.data.append(json.loads(line))

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs


class ParlAIConcatDataset(ConcatDataset):
    """Override to set num_eps and num_exs"""

    @lru_cache(maxsize=1)
    def num_episodes(self):
        return sum(d.num_episodes() for d in self.datasets)

    @lru_cache(maxsize=1)
    def num_examples(self):
        return sum(d.num_examples() for d in self.datasets)


class PytorchDataTeacher(FixedDialogTeacher):
    """
        A teacher that loads data using Pytorch Datasets. For details on how
        to use, please follow the tutorial here:
        http://parl.ai/static/docs/tutorial_worlds.html#multiprocessed-pytorch-dataloader
    """
    def __init__(self, opt, shared=None):
        opt['batch_sort'] = False
        super().__init__(opt, shared)
        self.use_batch_act = self.bsz > 1
        self.num_workers = opt['numworkers']
        self.batch_sort = opt.get('pytorch_teacher_batch_sort') and \
            'train' in self.datatype
        self.batch_cache_type = opt.get('batch_sort_cache_type')
        self.batch_sort_field = opt.get('batch_sort_field')
        # One can specify a collate function to use for preparing a batch
        self.opt = opt.copy()
        self.is_shared = shared is not None
        dataset_classes = self.get_dataset_class(opt)
        self.ordered = ('ordered' in self.datatype or
                        ('stream' in self.datatype and not opt.get('shuffle')))
        if self.ordered:
            # force index for ordered, so that we see every example
            warn_once(
                '\nNote: You are using PytorchDataTeacher with ordered '
                'examples. Please specify `--shuffle` if you would like '
                'to have examples loaded in randomized order.\n'
            )
            self.batch_cache_type = 'index'

        if not shared:
            BatchSortCache.create()
            if len(dataset_classes) > 1:
                datasets = []
                for class_name, collate_fn, task_name in dataset_classes:
                    dataset_opt = opt.copy()
                    dataset_opt['pytorch_teacher_task'] = task_name
                    dataset_opt['task'] = task_name
                    datasets.append(class_name(dataset_opt))
                    self.collate_fn = collate_fn
                self.id = ','.join([d[2] for d in dataset_classes])
                self.dataset = ParlAIConcatDataset(datasets)
            else:
                class_name, self.collate_fn, task_name = dataset_classes[0]
                self.id = task_name
                self.dataset = class_name(opt)
            if self.ordered or not self.training:
                data_sampler = sampler.SequentialSampler(self.dataset)
            else:
                data_sampler = sampler.RandomSampler(self.dataset)

            self.pytorch_dataloader = DataLoader(
                self.dataset,
                batch_size=self.bsz,
                sampler=data_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=False,
                drop_last=False,
            )

            self.lastYs = [None] * self.bsz
            if self.batch_sort:
                self.loader_process = LoaderProcess(opt)
                self.loader_process.start()
            self.data = enumerate(self.pytorch_dataloader)
        else:
            self.dataset = shared['dataset']
            self.pytorch_dataloader = shared['pytorch_dataloader']
            self.lastYs = shared['lastYs']
            self.data = shared['data']
            self.id = shared['id']

        self.num_batches = math.ceil(self.dataset.num_episodes() / self.bsz)
        self.reset()

    def get_dataset_class(self, opt):
        return get_dataset_classes(opt)

    def reset(self):
        """Reset the dialog so that it is at the start of the epoch,
        and all metrics are reset.
        """
        super().reset()
        self.reset_data()

    def reset_data(self):
        if not self.is_shared:
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
        shared['id'] = self.id
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
                self.episode_idx, episode = next(self.data)
                if self.collate_fn == default_collate:
                    episode = [ex[1] for ex in episode]
                self.episode = process(episode)
                self.entry_idx = 0
                epoch_done = False
            except StopIteration:
                ex = {'episode_done': True, 'id': self.getID()}
                epoch_done = True
        else:
            self.entry_idx += 1

        if not epoch_done:
            ex = self.episode[self.entry_idx]
            self.episode_done = ex['episode_done']
            if (self.episode_done and
                    self.episode_idx + self.bsz >= self.num_episodes()):
                epoch_done = True
        return ex, epoch_done

    @BatchSortCache.batch_cache
    def get_next_batch(self):
        # employs a cache to see if there is a batch of equal size ready
        batch = next(self.data)
        return batch

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
            batch = process(batch)
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

    def shutdown(self):
        super().shutdown()
        BatchSortCache.destroy()


class DefaultTeacher(PytorchDataTeacher):
    pass
