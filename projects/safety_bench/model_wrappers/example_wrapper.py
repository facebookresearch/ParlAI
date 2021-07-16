#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Example wrapper which replies `hello` to every text.
"""
from projects.safety_bench.utils.wrapper_loading import register_model_wrapper


@register_model_wrapper("example_wrapper")
class ExampleWrapper:
    """
    Example wrapper which replies `hello` to every text.
    """

    def __init__(self):
        # Do any initialization here, like loading the omdel
        pass

    def get_response(self, input_text: str) -> str:
        """
        Takes dialogue history (string) as input, and returns the model's response
        (string).
        """
        # This is the only method you are required to implement.
        # The input text is the corresponding input for the model.
        # Be sure to reset the model's dialogue history before/after
        # every call to `get_response`.

        return (
            "Hello"
        )  # In this example, we always respond 'Hello' regardless of the input
