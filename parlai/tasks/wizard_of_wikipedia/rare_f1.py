#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from typing import Iterable, Type

from parlai.core.metrics import F1Metric, normalize_answer
from parlai.core.opt import Opt
from parlai.core.teachers import (
    DialogData,
    ParlAIDialogTeacher,
    Teacher,
    StreamDialogData,
)
from parlai.utils.logging import logger


class RareF1Computer:
    """
    Helper class for computing F1, ignoring frequent words.

    The reference corpus is usually all message text from the training set.

    The API is very similar to other metrics, except this instance needs
    to be kept in memory to cache the reference distribution.
    """

    def __init__(self, corpus: str, top_p: float = 0.5):
        try:
            import nltk
        except ImportError:
            raise ImportError('Please install nltk (e.g. pip install nltk).')
        words = normalize_answer(corpus).split()
        self._freq_dist = nltk.FreqDist(words)
        self._cutoff_count = RareF1Computer._find_cutoff_count(self._freq_dist, top_p)

    @property
    def freq_dist(self):
        return self._freq_dist

    @staticmethod
    def _find_cutoff_count(freq_dist, top_p: float) -> int:
        """
        Finds the word occurance for which the cumulative occurances are `top_p` of the
        overall word count.
        """
        assert top_p < 1
        target = sum(freq_dist.values()) * top_p
        cumul = 0
        for _, v in freq_dist.most_common():
            cumul += v
            if cumul > target:
                return v
        raise RuntimeError(f"Invalid top {top_p*100}% of the corpus distribution")

    @staticmethod
    def _filter(freq_dist, cutoff: int, text: str) -> str:
        """
        For words that are found in the reference distribution, filters those with an
        occurrence count less than the cutoff.
        """
        words = normalize_answer(text).split()
        return " ".join([w for w in words if freq_dist.get(w, 0) < cutoff])

    def compute(self, guess: str, answers: Iterable[str]) -> F1Metric:
        if guess is None or answers is None:
            return F1Metric(0, 0)
        guess = RareF1Computer._filter(self._freq_dist, self._cutoff_count, guess)
        answers = [
            RareF1Computer._filter(self._freq_dist, self._cutoff_count, a)
            for a in answers
        ]
        if not any(len(a) for a in answers):
            # no rare words in labels, set denominator to zero
            return F1Metric(0, 0)
        return F1Metric.compute(guess, answers)

    @classmethod
    def from_reference_dialog_data(cls: Type, data: DialogData) -> RareF1Computer:
        all_text = ''
        if isinstance(data, StreamDialogData):
            logger.warning(
                "The stream will be consumed during processing. Don't "
                "use the StreamDialogData that is meant for your "
                "actual task to build RareF1Computer."
            )
            epoch_done = False
            while not epoch_done:
                action, epoch_done = data.get()
                all_text += action.get('text', '') + ' '
        else:
            for msg in data:
                all_text += msg.get('text', '') + ' '
        return cls(corpus=all_text)

    @classmethod
    def from_reference_parlai_format(cls: Type, file_path: str) -> RareF1Computer:
        teacher = ParlAIDialogTeacher(
            opt=Opt(
                task=cls.__name__,
                parlaidialogteacher_datafile=file_path,
                datatype='train:ordered',  # :ordered so we only do one epoch
            )
        )
        return cls.from_reference_teacher(teacher=teacher)

    @classmethod
    def from_reference_teacher(cls: Type, teacher: Teacher) -> RareF1Computer:
        all_text = ''
        for msg in teacher:
            if teacher.epoch_done():
                break
            all_text += msg.get('text', '') + ' '
        teacher.reset()  # reset in case teacher is reused
        return cls(corpus=all_text)
