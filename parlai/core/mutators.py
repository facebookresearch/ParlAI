#!/usr/bin/env python3


# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import importlib
import pkgutil
from typing import Optional, Iterable, Tuple, List, Iterator, Callable, Type

import parlai.mutators
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.message import Message

MUTATOR_REGISTRY: dict[str, Type] = {}


def setup_mutator_registry():
    """
    Loads the mutators so that @register_mutator is hit for all.
    """
    global MUTATOR_REGISTRY
    if hasattr(setup_mutator_registry, 'loaded'):
        return
    for module in pkgutil.iter_modules(parlai.mutators.__path__, 'parlai.mutators.'):
        importlib.import_module(module.name)
    setup_mutator_registry.loaded = True
    return MUTATOR_REGISTRY


def register_mutator(name: str) -> Callable[[Type], Type]:
    """
    Register a mutator.
    """

    def _inner(cls_: Type) -> Type:
        global MUTATOR_REGISTRY
        MUTATOR_REGISTRY[name] = cls_
        return cls_

    return _inner


class Mutator(abc.ABC):
    """
    Base class for mutators.

    Users are not advised to use this class.
    """

    @classmethod
    def load_mutator_types(cls, mutator_names: Optional[str]) -> List[Type]:
        """
        Map mutator names to actual classes via the registry.

        :param mutator_names:
            A list of one or more mutators separated by '+'. E.g.
            'flatten+word_shuffle'.
        :returns: a list of mutators
        """

        global MUTATOR_REGISTRY
        setup_mutator_registry()
        if not mutator_names:
            return []
        assert isinstance(mutator_names, str)
        names = mutator_names.replace('+', ',').split(',')
        mutators = [MUTATOR_REGISTRY[name] for name in names]
        return mutators

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        pass

    def __init__(self, opt):
        self.opt = opt

    def _pop_episode_done(self, message: Message) -> Tuple[Message, bool]:
        try:
            episode_done = message.pop('episode_done')
        except KeyError:
            episode_done = False
        return message, episode_done

    def _turn_to_messagenew_pair(
        self, messages: Iterable[Message]
    ) -> Iterator[Tuple[Message, bool]]:
        next_is_new_episode = True
        for message in messages:
            message, episode_done = self._pop_episode_done(message)
            yield message, next_is_new_episode
            next_is_new_episode = episode_done

    @abc.abstractmethod
    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        pass


class ExampleMutator(Mutator):
    """
    Example-level mutators.

    Example level mutators have a function applied per-example. They are ideal
    for transformations of data which don't create any new conversations or
    turns, but only apply simple text-transformations.

    Examples include:

    * Shuffling words in context
    * Adding a special token based on a non-text field
    * Replacing words with synonyms or other simple augmentations
    """

    @abc.abstractmethod
    def example_mutation(self, example: Message) -> Message:
        """
        Abstract example mutation.

        The main method to implement when implementing an ExampleMutator.

        :param example:
            An individual message you should mutate.
        :returns:
            The mutated message.
        """
        pass

    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        """
        Apply the mutator to a series of messages.

        Not meant to be called directly by a user.
        """
        for message in messages:
            if message.is_padding():
                yield message
                continue
            message, episode_done = self._pop_episode_done(message)
            message = self.example_mutation(message)
            if 'episode_done' in message:
                raise ValueError('Example Mutators should not modify episode_done.')
            message['episode_done'] = episode_done
            yield message


class EpisodeMutator(Mutator):
    """
    Episode-level mutators.
    """

    @abc.abstractmethod
    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        """
        Abstract epsiode mutation.

        The main method to implement when implementing an EpisodeMutator.

        The "episode_done" field will be automatically stripped before providing
        as input, and automatically added back to the finalized episode.

        :param messages:
            All the messages in one episode. You may manipulate any or all of
            them, or change the ordering entirely.
        :returns:
            The new, mutated episode.
        """
        pass

    def _postprocess_episode(self, unmutated_episode: List[Message]) -> List[Message]:
        if unmutated_episode and unmutated_episode[0].is_padding():
            for message in unmutated_episode:
                yield message
            return
        # make a list in case the user actually returned a generator
        mutated_episode = list(self.episode_mutation(unmutated_episode))
        if not mutated_episode:
            raise ValueError('Episode mutation returned an empty episode.')
        # set episode_done = False for everything except final
        for i, m in enumerate(mutated_episode):
            if 'episode_done' in m:
                raise ValueError('Episode mutators should not set episode_done.')
            m['episode_done'] = i == len(mutated_episode) - 1
            yield m

    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        """
        Apply the mutator to a series of messages.

        Not meant to be called directly by a user.
        """
        messagenew_pairs = self._turn_to_messagenew_pair(messages)
        episode: List[Message] = []
        for message, new_episode in messagenew_pairs:
            if new_episode and episode:
                yield from self._postprocess_episode(episode)
                episode = []
            episode.append(message)
        if episode:
            yield from self._postprocess_episode(episode)


class ManyEpisodeMutator(Mutator):
    """
    Episode mutator than can map one episode to zero or more.
    """

    @abc.abstractmethod
    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        """
        Abstract many-episode mutation.

        The main method to implement when creation a ManyEpisodeMutator.
        You should map this episode to one-or-more episodes.

        If you wish to create multiple episodes, you need to output
        one-sublist-per-new-episode. As with EpisodeMutator, "episode_done"
        will be automatically stripped and re-inserted for you.

        :param episode:
            A single episode (provided list of Messages).
        :returns:
            A list of list of messages. Each sub-list will be turned into a new
            episode.
        """
        pass

    def _postprocess_episode(self, unmutated_episode):
        # make a list in case the user actually returned a generator
        mutated_episodes = list(self.many_episode_mutation(unmutated_episode))
        for episode in mutated_episodes:
            episode = list(episode)
            for j, entry in enumerate(episode):
                if 'episode_done' in entry:
                    raise ValueError('Episode mutators should not set episode_done.')
                # set episode_done = False for everything except final
                entry['episode_done'] = j == len(episode) - 1
                yield entry

    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        """
        Apply the mutator to a series of messages.

        Not meant to be called directly by a user.
        """
        messagenew_pairs = self._turn_to_messagenew_pair(messages)
        episode: List[Message] = []
        for message, new_episode in messagenew_pairs:
            if message.is_padding():
                yield message
                continue
            if new_episode and episode:
                yield from self._postprocess_episode(episode)
                episode = []
            episode.append(message)
        if episode:
            yield from self._postprocess_episode(episode)
