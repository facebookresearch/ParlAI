#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Iterable, List

from parlai.utils.safety import OffensiveStringMatcher


class AcceptabilityChecker:
    def __init__(self):
        self.offensive_lang_detector = OffensiveStringMatcher()
        self.possible_violation_types = [
            'min_words',
            'penalize_greetings',
            'all_caps',
            'exact_match',
            'safety',
        ]

    def check_messages(
        self,
        messages: List[str],
        is_worker_0: bool,
        violation_types: Iterable[str] = (),
    ) -> str:
        """
        Returns a list of acceptability guidelines that the input messages violate.

        :param messages: List of all messages by one speaker
        :param is_worker_0: True if `messages` represent the messages from the first
            speaker in the conversation
        :param violation_types: Set of all violation types to check messages for. See
            `self.possible_violation_types` for a list of all possible violation types.
        :return: comma-separated list of all violations
        """

        if any(
            [
                violation_type not in self.possible_violation_types
                for violation_type in violation_types
            ]
        ):
            raise ValueError('One or more violation types are unrecognized!')

        if len(messages) == 0:
            # There may have been a disconnect, so in this case let's give them a pass
            return ''

        violations = []

        # Do messages have the minimum acceptable average number of words?
        if 'min_words' in violation_types:
            total_num_words = sum([len(message.split()) for message in messages])
            if total_num_words / len(messages) < 3:
                violations.append('under_min_length')

        # Does the first message start with a greeting, indicating that the Turker
        # hasn't understood the directions (i.e. continue along with the previous lines
        # of conversation)? Only penalize the worker who speaks first for this, because
        # the worker who speaks second should not be at fault if the first worker does
        # this and the second just follows along.
        if 'penalize_greetings' in violation_types:
            greetings = [
                'hi',
                'hii',
                'hiii',
                'hiiii',
                'hai',
                'hello',
                'hey',
                'howdy',
                'greetings',
            ]
            if is_worker_0 and (
                sum(
                    [
                        messages[0].split()[0].lower() == greeting
                        for greeting in greetings
                    ]
                )
                > 0
            ):
                violations.append('starts_with_greeting')

        # Does the Turker tend to speak in all caps?
        if 'all_caps' in violation_types:
            num_all_caps = sum([message == message.upper() for message in messages])
            if num_all_caps >= 2 or (num_all_caps == 1 and len(messages) == 1):
                violations.append('too_much_all_caps')

        # Are later messages an exact match of the first one?
        if 'exact_match' in violation_types:
            if len(messages) >= 2:
                c = messages[0]
                if exact_match(c, messages[1:]):
                    violations.append('exact_match')

        # Do the messages not pass the safety classifier?
        if 'safety' in violation_types:
            for idx, message in enumerate(messages):
                if self.offensive_lang_detector.contains_offensive_language(message):
                    violations.append(f'unsafe:{idx+1:d}')
                    # The messages are 1-indexed

        return ','.join(violations)


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(guess, answers):
    """
    Check if guess is a (normalized) exact match with any answer.
    """
    if guess is None or answers is None:
        return False
    guess = normalize_answer(guess)
    for a in answers:
        if guess == normalize_answer(a):
            return True
    return False
