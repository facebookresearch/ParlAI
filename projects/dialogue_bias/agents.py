#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, SumMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch
from parlai.utils import logging as logging
from parlai.utils.io import PathManager
from parlai.utils.misc import round_sigfigs
from parlai.utils.torch import NEAR_INF, NEAR_INF_FP16
from projects.dialogue_unlikelihood.agents import div
from projects.style_gen.modules import STYLE_SEP_TOKEN
from projects.style_gen.style_gen import StyleGenAgent


class NoBiasStyleGenAgent(StyleGenAgent):
    """
    Subclass of style gen agent that appends "no_bias" to every example's context.
    """

    def get_temp_history(self, observation: Message) -> Optional[str]:
        return STYLE_SEP_TOKEN + 'no_bias'


def report_bin_biases(
    partial_bin_percentages: List[float],
    frequencies_df: pd.DataFrame,
    dict_: Optional[DictionaryAgent] = None,
) -> Tuple[List[float], Dict[int, str], Dict[str, Dict[str, float]], List[str]]:
    """
    Return detailed information about the amount of bias per token bin.
    """

    log = []

    assert len(partial_bin_percentages) == 3 and sum(partial_bin_percentages) == 0.50
    bin_percentages = partial_bin_percentages + list(reversed(partial_bin_percentages))

    if dict_ is not None:
        truebins = dict()
    else:
        truebins = None
    current_bin_fraction = {'female': defaultdict(float), 'male': defaultdict(float)}
    bin_counts = defaultdict(int)
    female_male_freq_ratio = defaultdict(
        lambda: {'min': float('inf'), 'max': float('-inf')}
    )
    cdf_sorted_by_female_male_bias = frequencies_df.sort_values(
        'female_male_freq_ratio'
    )['avg_freq'].cumsum()
    for token, cdf in cdf_sorted_by_female_male_bias.items():
        if cdf < sum(bin_percentages[:1]):
            bin_name = 'hi_male'
        elif cdf < sum(bin_percentages[:2]):
            bin_name = 'med_male'
        elif cdf < sum(bin_percentages[:3]):
            bin_name = 'lo_male'
        elif cdf < sum(bin_percentages[:4]):
            bin_name = 'lo_female'
        elif cdf < sum(bin_percentages[:5]):
            bin_name = 'med_female'
        elif cdf < sum(bin_percentages[:6]):
            bin_name = 'hi_female'
        else:
            if cdf < 1 + 1e-10:
                # Rounding error, so just put it in the top bin
                bin_name = 'hi_female'
            else:
                raise ValueError('Bin percentage is invalid!')
        if dict_ is not None:
            truebins[dict_[token]] = bin_name
        bin_counts[bin_name] += 1
        for gender in ['female', 'male']:
            current_bin_fraction[gender][bin_name] += frequencies_df.loc[token, gender]
        female_male_freq_ratio[bin_name]['min'] = min(
            female_male_freq_ratio[bin_name]['min'],
            frequencies_df.loc[token, 'female_male_freq_ratio'],
        )
        female_male_freq_ratio[bin_name]['max'] = max(
            female_male_freq_ratio[bin_name]['max'],
            frequencies_df.loc[token, 'female_male_freq_ratio'],
        )
    for percentage, bin_name in zip(bin_percentages, bin_counts.keys()):
        female_frac = current_bin_fraction["female"][bin_name]
        male_frac = current_bin_fraction["male"][bin_name]
        bias = (female_frac - male_frac) / ((female_frac + male_frac) / 2)
        # This is the difference in the frequencies per gender as a fraction of
        # the frequency averaged between genders
        log.append(
            f'The "{bin_name}" bin has {bin_counts[bin_name]:d} unique tokens, '
            f'~{100 * percentage:0.2f}% of all generated tokens, '
            f'{100 * female_frac:0.2f}% of all generated female tokens, and '
            f'{100 * male_frac:0.2f}% of all generated male tokens, for a bias of '
            f'{100 * bias:0.2f}%.'
        )
    for bin_name, vals in female_male_freq_ratio.items():
        log.append(
            f'The "{bin_name}" bin has a minimum female/male frequency ratio of '
            f'{100 * vals["min"]:0.2f}% and a maximum ratio of '
            f'{100 * vals["max"]:0.2f}%.'
        )
    return bin_percentages, truebins, current_bin_fraction, log
