#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from fairscale.nn import checkpoint_wrapper
except ImportError:

    def checkpoint_wrapper(module):
        """
        Dummy checkpoint wrapper that raises an error.
        """
        raise ImportError(
            'Please install fairscale with `pip install fairscale` to use '
            '--checkpoint-activations true.'
        )
