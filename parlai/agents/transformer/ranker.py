# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure calling '-m transformer/ranker' works.
from .transformer import TransformerRankerAgent as RankerAgent  # noqa: F401
