#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re

import torch


# Failure modes
ISAID = 1
NOTSENSE = 2
UM = 3
YOUWHAT = 4
WHATYOU = 5
WHATDO = 6


class FeedbackClassifierRegex(object):
    def __init__(self):
        self.failure_regexes = {
            ISAID: r"i .*(?:said|asked|told).*",
            NOTSENSE: r"((not|nt|n't).*mak.*sense)|(mak.*no .*sense)",
            UM: r"u(m|h)+\W",
            YOUWHAT: r"you.*what\?",
            WHATYOU: r"what.*you (?:mean|refer|talk).*\?",
            WHATDO: r"what.*to do with.*\?",
        }

    def predict_proba(self, contexts):
        # Do naive for loop for now
        probs = []
        for context in contexts:
            start = context.rindex('__p1__')
            try:
                end = context.index('__null__')
            except ValueError:
                end = len(context)
            last_response = context[start:end]  # includes padding
            failure_mode = self.identify_failure_mode(last_response)
            probs.append(failure_mode is None)
        return torch.FloatTensor(probs)

    def identify_failure_mode(self, text):
        if re.search(self.failure_regexes[ISAID], text, flags=re.I):
            return ISAID
        elif re.search(self.failure_regexes[NOTSENSE], text, flags=re.I):
            return NOTSENSE
        elif re.search(self.failure_regexes[UM], text, flags=re.I):
            return UM
        elif re.search(self.failure_regexes[YOUWHAT], text, flags=re.I):
            return YOUWHAT
        elif re.search(self.failure_regexes[WHATYOU], text, flags=re.I):
            return WHATYOU
        elif re.search(self.failure_regexes[WHATDO], text, flags=re.I):
            return WHATDO
        else:
            return None
