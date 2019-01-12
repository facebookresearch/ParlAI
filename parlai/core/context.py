#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# This file contains a number of basic context types that can be extended for
# more specific cases. Special context forms should provide proper
# serialize and deserialize methods. You can then put any helper functions
# in here that it would make sense for a model to have. You don't need to
# extend these or use any context objects at all if you feel a primitive
# correctly covers the job.


class DictContext(dict):
    def __init__(self, *args):
        """This function is used to directly create a dict context object"""
        dict.__init__(self, *args)

    def serialize(self):
        return dict(self)

    @staticmethod
    def deserialize(input_data):
        return DictContext(input_data)

    def to_context_string(self):
        """This function should return a string that can be prepended to an
        act's text in order to support models that don't use context objects
        """
        return "DictContext: {}\n".format(dict.__repr__(self))


class ArrayContext(list):
    def __init__(self, *args):
        """This function is used to directly create an array context object"""
        list.__init__(self, *args)

    def serialize(self):
        return list(self)

    @staticmethod
    def deserialize(input_data):
        return ArrayContext(input_data)

    def to_context_string(self):
        """This function should return a string that can be prepended to an
        act's text in order to support models that don't use context objects
        """
        return "ArrayContext: {}\n".format(list.__repr__(self))


class TaskData(dict):
    def __init__(self, *args):
        """Task data is created as what should be an immutable dict."""
        dict.__init__(self, *args)
        self.locked = True

    def serialize(self):
        """Makes a best effort to serialize the data here so that it can be
        sent over wire if necessary
        """
        return {(k, v.serialize() if hasattr(v, 'serialize') else v)
                for (k, v) in self.items()}

    def __setitem__(self, key, val):
        if self.locked:
            raise Exception(
                'Cannot update the value of an immutable TaskData object')
        else:
            dict.__setitem__(self, key, val)

    def copy(self):
        """Makes a best effort to copy this TaskData object as something that
        can be altered"""
        return {(k, v.copy() if hasattr(v, 'copy') else v)
                for (k, v) in self.items()}

    def __repr__(self):
        return '{{<TaskData>: {}}}'.format(dict.__repr__(self))

    def __str__(self):
        context_strings = [v.to_context_string() for v in self.values()
                           if hasattr(v, 'to_context_string')]
        return ''.join(context_strings)
