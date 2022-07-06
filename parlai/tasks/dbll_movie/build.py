#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
import parlai.tasks.dbll_babi.build as dbll_babi_build
import parlai.tasks.wikimovies.build as wikimovies_build

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/dbll/dbll.tgz',
        'dbll.tgz',
        'd8c727dac498b652c7f5de6f72155dce711ff46c88401a303399d3fad4db1e68',
    )
]


def build(opt):
    # Depends upon another dataset, wikimovies, build that first.
    wikimovies_build.build(opt)
    dbll_babi_build.build(opt)
