#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


class BaseContext():
    """Any XContext should implement the following set of functions. It can
    then choose to implement any utility functions that it believes a model
    especially tuned to use that context type will find useful.
    """
    def __init__(self, args):
        """This function is used to directly create a context object"""
        raise NotImplementedError

    @staticmethod
    def from_data(input_data):
        """This function should take an input data object and produce the
        context representation for this data. It is expected to be used by
        tasks that are constructing context, and should be the inverse of
        to_data_form
        """
        raise NotImplementedError

    def to_data_form(self):
        """This function should return an object that is a less structured
        representation of the data contained in this object (generally in
        the form of a dict or an array)
        """
        raise NotImplementedError

    def to_context_string(self):
        """This function should return a string that can be prepended to an
        act's text in order to support models that don't use context objects
        """
        raise NotImplementedError

    def to_display_string(self):
        """This function will be invoked by the context binder whenever an
        act with a context binder is printed
        """
        return self.to_data_form()

    def get_type(self):
        """Returns the type of this context, which a model can use to
        parse it in a special way if it would like
        """
        raise NotImplementedError


class ContextBinder():
    def __init__(self, context_array=None):
        '''Expected to be initialized using an array of context objects, but
        it can also be initialized as an empty binder to add things to later
        '''
        self.content = {}
        if context_array is not None:
            for context in context_array:
                self.add_context(context)

    @staticmethod
    def from_data(input_context_dict):
        binder = ContextBinder()
        for context_type, context_data in input_context_dict.items():
            binder.add_context_from_data(context_type, context_data)
        return binder

    @staticmethod
    def from_context_array(input_data):
        """This function should take an array of context objects and return
        a ContextBinder containing those elements
        """
        binder = ContextBinder()
        for context in input_data:
            binder.add_context(context)
        return binder

    def add_context_from_data(self, context_type, context_data):
        """This function should parse the context data, determine the correct
        context object to create, and then create it.
        """
        raise NotImplementedError

    def add_context(self, context_object):
        assert issubclass(context_object.__class__, BaseContext), (
            'add_context can only be called with a valid context object that '
            'subclasses BaseContext'
        )
        context_type = context_object.get_type()
        if context_type not in self.content:
            self.content[context_type] = []
        self.content[context_type].append(context_object)

    def get_context(self, context_type):
        """Return all of the stored context of a context type"""
        return self.content.get(context_type, [])

    def __repr__(self):
        return '{{<ContextBinder>: {}}}'.format(self.to_display_string())

    def __str__(self):
        return self.to_context_string()

    def to_context_string(self):
        content_strings = [
            c.to_context_string()
            for contexts in self.content.values()
            for c in contexts
        ]
        return '\n'.join(content_strings)

    def to_display_string(self):
        content = {c_type: [c.to_display_string() for c in context_array]
                   for c_type, context_array in self.content.items()}
        return str(content)
