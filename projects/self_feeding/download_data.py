#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.params as params
from parlai.tasks.self_feeding.build import build


if __name__ == '__main__':
    opt = params.ParlaiParser().parse_args()
    build(opt)
