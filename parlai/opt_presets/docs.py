#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
List of all opt presets distributed with ParlAI.

This file is for automatically generating docs.
"""

# PRESET_DESCRIPTIONS is a dictionary mapping alias names to human descriptions
# for the sake of documentation
PRESET_DESCRIPTIONS = {
    "gen/meena": (
        "Inference parameters for the Sample & Rank procedure of Meena. "
        "See [Adiwardana et al. (2020)](https://arxiv.org/abs/2001.09977)."
    ),
    "arch/blenderbot_3B": (
        "Architecture parameters (number layers, etc) for BlenderBot 3B. See "
        "[Roller et al. (2020)](https://arxiv.org/abs/2004.13637)"
    ),
    "gen/blenderbot": (
        "Beam search parameters for BlenderBot. See"
        "[Roller et al. (2020)](https://arxiv.org/abs/2004.13637)"
    ),
    "arch/bart_large": (
        "Architecture parameters (number layers, etc.) for BART-Large. See "
        "[Lewis et. al. (2019](https://arxiv.org/abs/1910.13461)"
    ),
}
