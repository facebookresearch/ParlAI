#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .bart import BartAgent
from parlai.zoo.bart.build import BASE_ARGS


class BaseAgent(BartAgent):
    @classmethod
    def _get_cmdline_defaults(cls):
        return BASE_ARGS

    def _get_download_info(self):
        return 'bart.base'
