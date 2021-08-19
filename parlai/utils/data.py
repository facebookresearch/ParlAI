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
    def split_data_by_fold(
        cls,
        fold: str,
        data: List,
        train_frac: float,
        valid_frac: float,
        test_frac: float,
        seed: int = 42,
    ):
        """
        Splits a list of data into train/valid/test folds. The members of these folds
        are randomized (in a consistent manner) by a seed. This is a convenience
        function for datasets that do not have a canonical split.

        :param fold:
           parlai fold/datatype
        :param data:
            List of data examples to be split
        :param train_frac:
            Fraction of data to be used for the "train" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param valid_frac:
            Fraction of data to be used for the "valid" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param test_frac:
            Fraction of data to be used for the "test" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param seed:
            Seed for shuffling
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

        random.Random(seed).shuffle(data)
        return data[int(start * len(data)) : int(end * len(data))]

    @classmethod
    def split_subset_data_by_fold(
        cls,
        fold: str,
        subsets: List[List],
        train_frac: float,
        valid_frac: float,
        test_frac: float,
        seed: int = 42,
    ):
        """
        Splits a list of subsets of data, where we want equal samples from each subset,
        into train/valid/test folds, ensuring that samples from a given subset are not
        changed to another fold as more subsets are added.

        For example, say a dataset has domains A, B. Let's say we have an experiment where we train and validate a model on domain A, then on domains A + B. If we naively concatinate the subset of data from A + B together, randomize it, and split the result into train, valid, and test folds, there is no guarantee that valid or test examples from A-only will not end up into the train fold of the A + B split from this naive concatination process.

        The members of these folds are randomized (but in a fixed manner) by a seed.

        :param fold:
           parlai fold/datatype
        :param subsets:
            List of subsets of data examples to be split
        :param train_frac:
            Fraction of data to be used for the "train" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param valid_frac:
            Fraction of data to be used for the "valid" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param test_frac:
            Fraction of data to be used for the "test" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param seed:
            Seed for shuffling
        """
        result = []
        for subset in subsets:
            result.extend(
                cls.split_data_by_fold(
                    fold, subset, train_frac, valid_frac, test_frac, seed
                )
            )
        random.Random(seed).shuffle(result)
        return result
