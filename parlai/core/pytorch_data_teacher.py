# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""This module provides a teacher that utilizes a pytorch `DataLoader` for
    data loading. It contains the following classes:

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
            - `--buildteacher` set to the task teacher that will be/was used
                to build the pytorch data (by passing observations to the agent)
        - If building the dictionary for the first time, please specify
          the `--buildteacher` so that the dictionary can be built appropriately

    The following is a more in-depth explanation for PytorchDataTeacher usage;
    i.e., to use the PytorchDataTeacher, one must do the following:

    1. Ensure that an appropriate teacher exists that can read the data
       currently saved and produce an action dict for an agent (this will be
       the `buildteacher`)
    2. Subclass PytorchDataTeacher in a task agents file (one does not
       necessarily need to implement any of the methods in PytorchDataTeacher,
       it just should exist in the `agents.py` file).
    3. Build the data so that it can be used by the PytorchDataTeacher
        - This can be accomplished in 2 ways:
            1. Specify a `buildteacher`, `datafile` (where the data currently
               is, and what will be used to build the data), and `datatype`
               (e.g. train, valid, etc) when calling either `build_pytorch_data`
               or calling `train_model.py`. If one is training a model, the data
                will be built automatically in `train_model.py`.
            2. Implement the `buildteacher` such that it saves the appropriate
               datafile in its `datafile` attribute (i.e. `self.datafile`) given
               the datatype, and then specify the `buildteacher` when calling
               either `build_pytorch_data.py` or `train_model.py`

    Additionally, if `preprocessed` is set to `True`, then the model specified
    in the command line params will have its `observe` function called on the
    `buildteacher`'s action, and the data will be saved for that model
    specifically.

    Here's an example of what would need to be done for `bAbI` 10k task 1,
    with preprocessed data from the `seq2seq` model.

    1. Implement a normal `bAbI` teacher that can read the data in its current
       format and create an action dict (this currently exists as the
       `Task1kTeacher`
    2. Implement a `bAbI` teacher that subclasses PytorchDataTeacher, e.g.
       `Task1kPyTeacher`
    3. If `Task1kTeacher` saves the datafile in it's attributes, use one of the
       following 2 commands:
       - `python examples/train_model.py -t babi:task10kpy:1 --buildteacher
         babi:task10k:1 -m seq2seq -mf /tmp/pytorch_data_build --preprocessed 1`
            - if one would like to train the model after building the data
       - `python examples/build_pytorch_data.py -m seq2seq -t babi:task10kpy:1
         --buildteacher babi:task10k:1 --preprocessed 1`
    4. If you would like to specify a specific datafile to build, e.g. a
       validation file, you could do either of the following:
       - `python examples/train_model.py -t babi:task10kpy:1 --buildteacher
         babi:task10k:1 --datafile data/bAbI/tasks_1-20_v1-2/en-valid-10k-nosf/qa1_valid.txt
         -dt valid -m seq2seq -mf /tmp/pytorch_data_build --preprocessed 1`
       - `python examples/build_pytorch_data.py -m seq2seq -t babi:task10kpy:1
         --buildteacher babi:task10k:1 --preprocessed 1
         --datafile data/bAbI/tasks_1-20_v1-2/en-valid-10k-nosf/qa1_valid.txt`

"""
from .agents import Teacher
from examples.build_pytorch_data import build_data

import json
import math
import copy
try:
    import torch
except Exception as e:
    raise ModuleNotFoundError('Need to install Pytorch: go to pytorch.org')
from torch.utils.data import Dataset, DataLoader, sampler


# Default collate function (for how to prepare a batch)
def default_collate(batch):
    return [b[0] for b in batch]


class StreamDataset(Dataset):
    """A Pytorch Dataset utilizing streaming"""
    def __init__(self, opt):
        self.opt = opt
        self.datafile = build_data(self.opt)
        self.data_gen = self._data_generator(self.datafile)
        self.length_datafile = self.datafile + ".length"
        self._load_lens()

    def __getitem__(self, index):
        # (ignore index because it is streaming data)
        return next(self.data_gen)

    def __len__(self):
        return self.num_eps

    def _load_lens(self):
        with open(self.length_datafile) as length:
            lengths = json.load(length)
            self.num_eps = lengths['num_eps']
            self.num_exs = lengths['num_exs']

    def _data_generator(self, datafile):
        while True:
            for episode in self._read_episode(self.datafile):
                yield episode

    def _read_episode(self, datafile):
        read = open(datafile)
        episode = []
        for line in read:
            example = json.loads(line)
            episode.append(example)
            if example['episode_done']:
                yield episode
                episode = []
        read.close()

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs


class PytorchDataTeacher(Teacher):

    @staticmethod
    def add_cmdline_args(argparser):
        arg_group = argparser.add_argument_group('PytorchData Arguments')
        arg_group.add_argument('--datafile', type=str, default='',
            help='datafile for pytorch data loader')
        arg_group.add_argument('-nw', '--numworkers', type=int, default=4,
            help='how many workers the Pytorch dataloader should use')
        arg_group.add_argument('--buildteacher', type=str, default='',
            help='Which teacher to use when building the pytorch data')
        arg_group.add_argument('--preprocess', type=bool, default=True,
            help='Whether the agent should preprocess the data while building'
                 'the pytorch data')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not hasattr(self, 'datatype'):
            self.datatype = opt['datatype']
        if not hasattr(self, 'training'):
            self.training = self.datatype.startswith('train')

        self.bsz = opt['batchsize']
        self.step_size = self.bsz
        self.batchindex = opt.get('batchindex', 0)
        self.num_workers = opt['numworkers']
        # One can specify a collate function to use for preparing a batch
        collate_fn = opt.get('collate_fn', default_collate)
        if not shared:
            self.dataset = StreamDataset(opt)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.bsz,
                shuffle=False,
                sampler=sampler.SequentialSampler(self.dataset),
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=False,
                timeout=0)
            self.lastYs = [None] * self.bsz
        else:
            self.dataset = shared['dataset']
            self.dataloader = shared['dataloader']
            self.lastYs = shared['lastYs']

        self.num_batches = math.ceil(self.dataset.num_examples()/self.bsz)
        self.reset()

    def reset(self):
        """Reset the dialog so that it is at the start of the epoch,
        and all metrics are reset.
        """
        super().reset()
        self.metrics.clear()
        self.reset_data()

    def reset_data(self):
        self.data = enumerate(self.dataloader)
        self.lastY = None
        self.epochDone = False
        self.episode = None
        self.episode_done = True
        self.episode_idx = 0
        self.data = enumerate(self.dataloader)

    def share(self):
        shared = super().share()
        shared['dataloader'] = self.dataloader
        shared['dataset'] = self.dataset
        shared['lastYs'] = self.lastYs
        return shared

    def next_example(self):
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
            ex = self.episode[self.entry_idx]
            self.episode_done = ex['episode_done']
            if (self.episode_done
                    and self.episode_idx + self.step_size >= self.num_episodes()):
                epoch_done = True

        return ex, epoch_done

    def next_batch(self):
        try:
            batch_idx, batch = next(self.data)
            epoch_done = False
        except StopIteration:
            batch = [{'episode_done': True, 'id': self.getID()}] * self.bsz
            epoch_done = True
        if not epoch_done and batch_idx == self.num_batches:
            epoch_done = True

        return batch, epoch_done

    def num_episodes(self):
        """Get the number of episodes in this dataset."""
        return self.dataset.num_episodes()

    def num_examples(self):
        """Get the total number of examples in this dataset."""
        return self.dataset.num_examples()

    def observe(self, observation):
        """Process observation for metrics."""
        if self.bsz > 1:
            self.lastY = self.lastYs[self.batchindex]
        if self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    def batch_act(self, observations):
        # we ignore observations
        if not hasattr(self, 'epochDone'):
            # reset if haven't yet
            self.reset()

        if self.epochDone:
            if not self.training:
                return [{'episode_done': True, 'id': self.getID()}] * self.bsz
            else:
                self.reset_data()

        # get next batch
        batch, self.epochDone = self.next_batch()

        # pad batch
        if len(batch) < self.bsz:
            batch += [{'episode_done': True, 'id': self.getID()}] * (self.bsz - len(batch))

        for i, ex in enumerate(batch):
            self.lastYs[i] = ex.get('labels', ex.get('eval_labels'))

        return batch

    def act(self):
        """Send new dialog message."""
        if self.epochDone:
            if not self.training:
                return {'episode_done': True, 'id': self.getID()}
            else:
                self.reset_data()

        # get next example
        action, self.epochDone = self.next_example()
        action['id'] = self.getID()

        # remember correct answer if available
        self.lastY = action.get('labels', action.get('eval_labels', None))
        if not self.datatype.startswith('train') and 'labels' in action:
            # move labels to eval field so not used for training
            # but this way the model can use the labels for perplexity or loss
            action['eval_labels'] = action.pop('labels')
        return action
