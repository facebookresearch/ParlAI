#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities related to handling data.
"""
import random
from typing import List


class DatatypeHelper:
    """
    Helper class to determine properties from datatype strings.
    """

    @classmethod
    def fold(cls, datatype: str) -> str:
        """
        Extract the fold part of the datatype.

        :param datatype:
            parlai datatype

        :return: the fold

        >>> DatatypeHelper.fold("train:ordered")
        ... "train"
        """
        return datatype.split(':')[0]

    @classmethod
    def strip_stream(cls, datatype: str) -> str:
        """
        Remove :stream from the datatype.

        Used by ChunkTeacher where behavior does not change based on streaming.

        :param datatype:
            parlai datatype

        :return:
            a non-streaming version of the datatype.

        >>> DatatypeHelper.fold("train:stream")
        "train"
        >>> DatatypeHelper.fold("train")
        "train"
        """
        return datatype.replace(":stream", "")

    @classmethod
    def should_cycle(cls, datatype: str) -> bool:
        """
        Return whether we should cycle data based on the datatype.

        :param datatype:
            parlai datatype

        :return should_cycle:
            given datatype, return whether we should cycle
        """
        assert datatype is not None, 'datatype must not be none'
        return (
            'train' in datatype
            and 'evalmode' not in datatype
            and 'ordered' not in datatype
        )

    @classmethod
    def should_shuffle(cls, datatype: str) -> bool:
        """
        Return whether we should shuffle data based on the datatype.

        :param datatype:
            parlai datatype

        :return should_shuffle:
            given datatype, return whether we should shuffle
        """
        assert datatype is not None, 'datatype must not be none'
        return (
            'train' in datatype
            and 'evalmode' not in datatype
            and 'ordered' not in datatype
            and 'stream' not in datatype
        )

    @classmethod
    def is_training(cls, datatype: str) -> bool:
        """
        Return whether we should return eval_labels or labels.

        :param datatype:
            parlai datatype

        :return is_training:
            bool indicating whether should return eval_labels or labels
        """
        assert datatype is not None, 'datatype must not be none'
        return 'train' in datatype and 'evalmode' not in datatype

    @classmethod
    def is_streaming(cls, datatype: str) -> bool:
        """
        Return whether this is streaming.

        :param datatype:
            parlai datatype

        :returns:
            bool indicating whether we are streaming
        """
        return 'stream' in datatype

    @classmethod
    def split_domains_by_fold(
        cls,
        fold: str,
        domains: List[List],
        train_frac: float,
        valid_frac: float,
        test_frac: float,
        seed: int = 42,
    ):
        """
        Need to be careful about how we setup random to not leak examples between trains
        if we're in a scenario where a single dataset has different ways of mixing +
        matching subcomponents.
        """
        assert train_frac + valid_frac + test_frac == 1
        if "train" in fold:
            start = 0.0
            end = train_frac
        elif "valid" in fold:
            start = train_frac
            end = train_frac + valid_frac
        else:
            start = train_frac + valid_frac
            end = 1.0

        result = []
        for domain in domains:
            random.Random(seed).shuffle(domain)
            result.extend(domain[int(start * len(domain)) : int(end * len(domain))])
        random.Random(seed).shuffle(result)
        return result
