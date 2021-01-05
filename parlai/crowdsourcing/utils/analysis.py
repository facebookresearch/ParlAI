#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class AbstractResultsCompiler:
    """
    Abstract class for compiling results of crowdsourcing runs.

    Currently only provides utility attributes/methods for analyzing turn annotations.
    """

    PROBLEM_BUCKETS = [
        'bucket_0',
        'bucket_1',
        'bucket_2',
        'bucket_3',
        'bucket_4',
        'none_all_good',
    ]

    def __init__(self, opt: Dict[str, Any]):
        if 'results_folders' in opt:
            self.results_folders = opt['results_folders'].split(',')
        else:
            self.results_folders = None
