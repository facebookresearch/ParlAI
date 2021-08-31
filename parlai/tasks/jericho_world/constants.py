#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
import nltk
from nltk.corpus import stopwords


class GraphMutations(Enum):
    NO_MUTATION = 0
    DEL = 1
    ADD = 2


try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


# The set of words in the description of graph that we skip while checking
# for overlap between knowledge graph vertices and the description.
# We remove the word "you" from this set becuase otherwise the edges that have you (player) vertices,
# which are quite important, will be reduced to empty vertices, and thus dropped.
GRAPH_VERT_SKIP_TOKEN = set(stopwords.words('english')) - {'you'}

LOCATION_NAME = 'loc-name'
LOCATION_DESCRIPTION = 'loc-desc'
SURROUNDING_OBJECTS = 'surr_obj'
KNOWLEDGE_GRAPH = 'KG'
ACTION = 'action'
OBSERVATION = 'obs'

# The delimiter characetrs between members of a set (eg objects, graph edges, etc.)
SET_MEMBERS_DELIM = ' ; '
GRAPH_DELIM = ','

EMPTY_GRAPH_TOKEN = '__empty-graph__'
