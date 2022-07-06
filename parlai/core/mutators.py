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
    try:
        import parlai_fb.mutators

        for module in pkgutil.iter_modules(
            parlai_fb.mutators.__path__, 'parlai_fb.mutators.'
        ):
            importlib.import_module(module.name)
    except ImportError:
        pass
    try:
        import parlai_internal.mutators

        for module in pkgutil.iter_modules(
            parlai_internal.mutators.__path__, 'parlai_internal.mutators.'
        ):
            importlib.import_module(module.name)
    except ImportError:
        pass
    setup_mutator_registry.loaded = True
    return MUTATOR_REGISTRY


def register_mutator(name: str) -> Callable[[Type], Type]:
    """
    Register a mutator.
    """

    def _inner(cls_: Type) -> Type:
        global MUTATOR_REGISTRY
        if name in MUTATOR_REGISTRY and cls_ is not MUTATOR_REGISTRY[name]:
            raise NameError(
                "Mutators must be uniquely named, but detected two mutators with "
                f"the name '{name}'."
            )
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

    def _group_into_episodes(
        self, message_stream: Iterable[Message]
    ) -> Iterator[List[Message]]:
        """
        Apply fn to grouped episodes, yielding back the results of the application.
        """
        episode: List[Message] = []
        for message in message_stream:
            if message.is_padding():
                assert not episode
                yield [message]
                continue
            message, episode_done = self._pop_episode_done(message)
            episode.append(message)
            if episode_done:
                yield episode
                episode = []
        if episode:
            yield episode

    def _add_episode_done(self, episode: List[Message]) -> List[Message]:
        for i, message in enumerate(episode):
            message['episode_done'] = i == len(episode) - 1
        return episode

    @abc.abstractmethod
    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        pass


class MessageMutator(Mutator):
    """
    Message-level mutators.

    Message-level mutators have a function applied per-utterance. They are ideal
    for transformations of data which don't create any new conversations or
    turns, but only apply simple text-transformations.

    Examples include:

    * Shuffling words in context
    * Adding a special token based on a non-text field
    * Replacing words with synonyms or other simple augmentations
    """

    @abc.abstractmethod
    def message_mutation(self, message: Message) -> Message:
        """
        Abstract message mutation.

        The main method to implement when implementing an MessageMutator.

        :param message:
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
            message = self.message_mutation(message)
            if 'episode_done' in message:
                raise ValueError('MessageMutators should not modify episode_done.')
            message['episode_done'] = episode_done
            yield message


class EpisodeMutator(Mutator):
    """
    Episode-level mutators.
    """

    @abc.abstractmethod
    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        """
        Abstract episode mutation.

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

    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        """
        Apply the mutator to a series of messages.

        Not meant to be called directly by a user.
        """
        for episode in self._group_into_episodes(messages):
            if episode and episode[0].is_padding():
                for message in episode:
                    yield message
            else:
                mutated_episode = self._add_episode_done(self.episode_mutation(episode))
                yield from mutated_episode


class ManyEpisodeMutator(Mutator):
    """
    Episode mutator than can map one episode to zero or more.
    """

    @abc.abstractmethod
    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        """
        Abstract many-episode mutation.

        The main method to implement when creation a ManyEpisodeMutator.
        You should map this episode to zero-or-more episodes.

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

    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        """
        Apply the mutator to a series of messages.

        Not meant to be called directly by a user.
        """

        for episode in self._group_into_episodes(messages):
            if episode and episode[0].is_padding():
                yield from episode
            else:
                mutated_episodes = self.many_episode_mutation(episode)
                for mutated_episode in mutated_episodes:
                    yield from self._add_episode_done(mutated_episode)
