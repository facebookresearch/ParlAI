#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

ANLI = 'ANLI'
ANLI_PREFIX = 'anli_'
ANLI_PREMISE_PREFIX = 'Premise: '
ANLI_HYPO_PREFIX = 'Hypothesis: '
ANLI_LABEL_DICT = {'e': 'entailment', 'c': 'contradiction', 'n': 'neutral'}
ANLI_LABELS = list(ANLI_LABEL_DICT.values())
ANLI_PREMISE_KEY = 'context'
ANLI_HYPO_KEY = 'hypothesis'
ANLI_ANSWER_KEY = 'label'
ANLI_ROUNDS = ['R1', 'R2', 'R3']
