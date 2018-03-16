# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.seq2seq.seq2seq import Seq2seqAgent


import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

from collections import deque

import os
import math

class NewPersonaSeq2seqAgent(Seq2seqAgent):
    pass
