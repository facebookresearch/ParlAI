# Copyright 2004-present Facebook. All Rights Reserved.

from multiprocessing import RawArray
import ctypes
import random
import sys


class TextData(object):
    """Provides a data structure for accessing text data.
    This can be used whenever the dialog data is a fixed log of chats
    (i.e not a simulator setting). The logs can include dialog text and possibly
    supervised labels, candidate labels and rewards.

    All these are stored in this internal data format which is used by the
    DialogTeacher class in dialog.py.

    data_loader is an iterable, with each call returning:

    (x, ...), new_episode?

    Where...
    x is a query and possibly context
    ... can contain additional fields, specifically
        y is an iterable of label(s) for that query
        r is the str reward for getting that query correct
        c is an iterable of label candidates that the student can choose from
    new_episode? is a boolean value specifying whether that example is the start
    of a new episode. If you don't use episodes set this to True every time.

    cands can be set to provide a list of candidate labels for every example
        in this dataset, which the agent can choose from (the correct answer
        should be in this set).

    random tells the data class whether or not to visit episodes sequentially
    or randomly when returning examples to the caller.
    """
    # data_loader should be a generator returning (Entry, new) tuples
    def __init__(self, data_loader, cands=None, random=False):
        self.data = []
        self._load(data_loader)
        self.cands = set(sys.intern(c) for c in cands) if (
            cands is not None) else None
        self.addedCands = []
        self.random = random
        self.entry_generator = self._gen_entries()

    # returns number of entries
    def __len__(self):
        length = 0
        for l in self.data:
            length += len(l)
        return length

    def __iter__(self):
        return self

    def __next__(self):
        return self._get_observation()

    # data is an iterator over ((x,y,...), new_episode) of the data
    def _load(self, data_loader):
        episode = []
        last_cands = None
        for entry, new in data_loader:
            if new:
                if len(episode) > 0:
                    self.data.append(tuple(episode))
                    episode = []
                    last_cands = None

            # intern all strings so we don't store them more than once
            new_entry = []
            if len(entry) > 0:
                # process text
                if entry[0] is not None:
                    new_entry.append(sys.intern(entry[0]))
                else:
                    new_entry.append(None)
                if len(entry) > 1:
                    # process labels
                    if entry[1] is not None:
                        new_entry.append(tuple(sys.intern(e) for e in entry[1]))
                    else:
                        new_entry.append(None)
                    if len(entry) > 2:
                        # process reward
                        if entry[2] is not None:
                            new_entry.append(sys.intern(entry[2]))
                        else:
                            new_entry.append(None)
                        if len(entry) > 3 and entry[3] is not None:
                            # process label candidates
                            if last_cands and entry[3] is last_cands:
                                new_entry.append(
                                    sys.intern('same as last time'))
                            else:
                                last_cands = entry[3]
                                new_entry.append(tuple(
                                    sys.intern(e) for e in entry[3]))
            episode.append(tuple(new_entry))

        if len(episode) > 0:
            self.data.append(tuple(episode))

    def _get_observation(self):
        entry, done, end_of_data = next(self.entry_generator)

        table = {}
        table['text'] = entry[0]
        if len(entry) > 1:
            table['labels'] = entry[1]
            if len(entry) > 2:
                table['reward'] = entry[2]
                if len(entry) > 3:
                    table['label_candidates'] = entry[3]

        if (table.get('labels', None) is not None
            and self.cands is not None):
            if self.addedCands:
                # remove elements in addedCands
                self.cands.difference_update(self.addedCands)
                self.addedCands.clear()
            for label in table['labels']:
                if label not in self.cands:
                    # add labels, queue them for removal next time
                    self.cands.add(label)
                    self.addedCands.append(label)
            table['label_candidates'] = self.cands

        if 'labels' in table and 'label_candidates' in table:
            if table['labels'][0] not in table['label_candidates']:
                raise RuntimeError('true label missing from candidate labels')

        # last entry in this episode
        table['done'] = done
        return table, end_of_data

    # returns entries, doing episodes in order
    # randomly switches between episodes if random flag set
    def _gen_entries(self):
        NUM_EPISODES = len(self.data)
        episode_idx = -1
        while True:
            if self.random:
                # select random episode
                episode_idx = random.randrange(NUM_EPISODES)
            else:
                # select next episode
                episode_idx = (episode_idx + 1) % NUM_EPISODES
            episode = self.data[episode_idx]
            for i in range(len(episode)):
                curr_entry = episode[i]
                end_of_ep = i == len(episode) - 1
                end_of_data = (end_of_ep and not self.random and
                               episode_idx == NUM_EPISODES - 1)
                yield episode[i], end_of_ep, end_of_data


class HogwildTextData(TextData):

    def __init__(self, data_loader, cands=None, random=False):
        super().__init__(data_loader, random=False)
        self.arr, self.ep_idxs = self._data2array(self.data)
        del self.data
        self.entry_generator = self._gen_entries()

    def __len__(self):
        return self.len

    def _get_observation(self):
        return next(self.entry_generator)

    def _gen_entries(self):
        table = {}
        episode_idx = -1
        while True:
            # iterate randomly if training, ordered in valid/test
            if self.random:
                episode_idx = random.randrange(len(self.ep_idxs) - 1)
            else:
                episode_idx = (episode_idx + 1) % (len(self.ep_idxs) - 1)
            # loop over each example in this episode, building that ex
            for idx in range(self.ep_idxs[episode_idx],
                             self.ep_idxs[episode_idx + 1],
                             4):
                table.clear()
                table['text'] = self.arr[idx]
                if self.arr[idx + 1]:
                    table['labels'] = self.arr[idx + 1].split('|')
                if self.arr[idx + 2]:
                    table['reward'] = self.arr[idx + 2]
                if self.arr[idx + 3]:
                    table['label_candidates'] = self.arr[idx + 3].split('|')
                table['done'] = idx == self.ep_idxs[episode_idx + 1] - 4
                yield table

    # returns data in array form, and indices to the start of each episode
    def _data2array(self, data):
        num_eps = len(data)
        num_exs = super().__len__()
        self.len = num_exs
        ep_idxs = RawArray('i', num_eps + 1)
        arr = RawArray(ctypes.c_wchar_p, num_exs * 4)

        arr_idx = 0
        for i, episode in enumerate(data):
            ep_idxs[i] = arr_idx
            for entry in episode:
                text = entry[0]
                # default to blank (need something in array), replace if avail.
                labels, reward, label_candidates = [sys.intern('')] * 3
                if len(entry) > 1:
                    labels = sys.intern('|'.join(entry[1]))
                    if len(entry) > 2:
                        reward = entry[2]
                        if len(entry) > 3:
                            label_candidates = sys.intern('|'.join(entry[3]))

                arr[arr_idx] = text
                arr[arr_idx + 1] = labels
                arr[arr_idx + 2] = reward
                arr[arr_idx + 3] = label_candidates
                arr_idx += 4
        ep_idxs[-1] = arr_idx
        return arr, ep_idxs
