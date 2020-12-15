#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities related to handling data.
"""


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
            given datatype, return whether we should return eval_labels or labels
        """
        assert datatype is not None, 'datatype must not be none'
        return 'train' in datatype and 'evalmode' not in datatype
