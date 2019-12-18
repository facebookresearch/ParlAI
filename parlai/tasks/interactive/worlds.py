#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.worlds import DialogPartnerWorld


class InteractiveWorld(DialogPartnerWorld):
    """
    Simple interactive world involving just two agents talking.

    In more sophisticated worlds the environment could supply information, e.g. in
    tasks/convai2 both agents are given personas, so a world class should be written
    especially for those cases for given tasks.
    """

    pass
