#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
File for Message object and associated functions.

The Message object's key function is to prevent users from editing fields in an action
or observation dict unintentionally.
"""

from __future__ import annotations
from typing import Optional


class Message(dict):
    """
    Class for observations and actions in ParlAI.

    Functions like a dict, but triggers a RuntimeError when calling __setitem__ for a
    key that already exists in the dict.
    """

    def __setitem__(self, key, val):
        if key in self:
            raise RuntimeError(
                'Message already contains key `{}`. If this was intentional, '
                'please use the function `force_set(key, value)`.'.format(key)
            )
        super().__setitem__(key, val)

    def force_set(self, key, val):
        super().__setitem__(key, val)

    def copy(self):
        return type(self)(self)

    @classmethod
    def padding_example(cls) -> Message:
        """
        Create a Message for batch padding.
        """
        return cls({'batch_padding': True, 'episode_done': True})

    def is_padding(self) -> bool:
        """
        Determine if a message is a padding example or not.
        """
        return bool(self.get('batch_padding'))


class History(object):
    """
    History handles tracking the dialogue state over the course of an episode.

    History may also be used to track the history of any field.

    :param field:
        field in the observation to track over the course of the episode
        (defaults to 'text')

    :param maxlen:
        sets the maximum number of tunrs

    :param p1_token:
        token indicating 'person 1'; opt must have 'person_tokens' set to True
        for this to be added

    :param p1_token:
        token indicating 'person 2'; opt must have 'person_tokens' set to True
        for this to be added
    """

    def __init__(
        self,
        opt,
        field='text',
        maxlen=None,
        size=-1,
        p1_token='__p1__',
        p2_token='__p2__',
    ):
        self.field = field
        self.delimiter = opt.get('delimiter', '\n')
        self.size = size
        self.split_on_newln = opt.get('split_lines', False)
        self.reversed = opt.get('history_reversed', False)

        # set up history objects
        self.max_len = maxlen

        self.history_strings = []
        self.history_raw_strings = []
        self.temp_history = None

        # person token args
        self.add_person_tokens = opt.get('person_tokens', False)
        self.add_p1_after_newln = opt.get('add_p1_after_newln', False)
        self.p1_token = p1_token
        self.p2_token = p2_token

    def reset(self):
        """
        Clear the history.
        """
        self.history_raw_strings = []
        self.history_strings = []

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

    def add_reply(self, text):
        """
        Add your own response to the history.
        """
        self._update_raw_strings(text)
        if self.add_person_tokens:
            text = self._add_person_tokens(text, self.p2_token)
        # update history string
        self._update_strings(text)

    def update_history(self, obs: Message, temp_history: Optional[str] = None):
        """
        Update the history with the given observation.

        :param obs:
            Observation used to update the history.
        :param temp_history:
            Optional temporary string. If it is not None, this string will be
            appended to the end of the history. It will not be in the history
            on the next dialogue turn. Set to None to stop adding to the
            history.
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

        self.temp_history = temp_history

    def get_history_str(self) -> Optional[str]:
        """
        Return the string version of the history.
        """
        if len(self.history_strings) > 0:
            history = self.history_strings[:]
            history = self.delimiter.join(history)
            if self.temp_history is not None:
                history += self.temp_history
            return history

        return None

    def _add_person_tokens(self, text, token, add_after_newln=False):
        if add_after_newln:
            split = text.split('\n')
            split[-1] = token + ' ' + split[-1]
            return '\n'.join(split)
        else:
            return token + ' ' + text

    def __str__(self) -> str:
        return self.get_history_str() or ''

    def __len__(self) -> int:
        return len(self.history_strings)
