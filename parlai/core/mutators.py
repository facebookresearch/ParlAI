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
    if hasattr(setup_mutator_registry, 'loaded'):
        return
    for module in pkgutil.iter_modules(parlai.mutators.__path__, 'parlai.mutators.'):
        importlib.import_module(module.name)
    setattr(setup_mutator_registry, 'loaded', True)


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

    def _turn_to_episode_done(
        self, message_new_pairs: Iterator[Tuple[Message, bool]]
    ) -> Iterator[Message]:
        iterable = iter(message_new_pairs)
        last_message: Message
        try:
            last_message, new_episode = next(iterable)
        except StopIteration:
            return
        for message, new_episode in iterable:
            if 'episode_done' in message:
                raise KeyError("Mutated messages should not have episode_done keys.")
            last_message['episode_done'] = new_episode
            yield last_message
            last_message = message
        last_message['episode_done'] = True
        yield last_message

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
        pass

    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        for message in messages:
            message = message.copy()
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
        pass

    def _postprocess_episode(self, unmutated_episode):
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
        messagenew_pairs = self._turn_to_messagenew_pair(messages)
        episode: List[Message] = []
        for message, new_episode in messagenew_pairs:
            if new_episode and episode:
                yield from self._postprocess_episode(episode)
                episode = []
            episode.append(message)
        if episode:
            yield from self._postprocess_episode(episode)
