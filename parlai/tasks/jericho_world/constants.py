#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class GraphMutations(Enum):
    NO_MUT = 0
    DEL = 1
    ADD = 2


LOCATION_NAME = 'loc-name'
LOCATION_DESCRIPTION = 'loc-desc'
SURROUNDING_OBJECTS = 'surr_obj'

# The delimiter characetrs between members of a set (eg objects, graph edges, etc.)
SET_MEMBERS_DELIM = ' ; '

EMPTY_GRAPH_TOKEN = '__empty-graph__'
