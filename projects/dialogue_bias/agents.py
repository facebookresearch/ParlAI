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


class TransformerGenderDebiasAgent(TransformerGeneratorAgent):
    """
    Agent for reducing gender bias via unlikelihood.

    During training, tokens will be generated until EOS, and tokens overindexed for the
    specified gender will be penalized in the loss term.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.penalize_generations = opt['penalize_generations']
        if shared is None:
            self._reset_running_histories()
            self._last_was_training = True
            frequencies_file = self.opt['frequencies_file']
            if frequencies_file is None:
                frequencies_file = os.path.join(
                    os.path.dirname(self.opt['model_file']), 'freq_by_name_list.csv'
                )
                if not PathManager.exists(frequencies_file):
                    raise RuntimeError(
                        'Please give a --frequencies-file to use the debiasing agent'
                    )

            partial_bin_percentages = [
                float(pct) / 100 for pct in self.opt['bin_percentages'].split(',')
            ]

            logging.info('Reading in tokens.')
            with PathManager.open(frequencies_file) as f:
                frequencies_df = pd.read_csv(f).set_index('token')
                # Each value represents where a token lies along the cumulative
                # distribution function of all tokens, sorted by female/male
                # frequency ratio. For example, a value of 0.8 means that, for 80% of
                # all generated tokens, the amount of overuse in conversations with a
                # female name will be less than that of this token.

            logging.info('Calculating relative entropies per token.')
            self.relative_entropies = {'female': {}, 'male': {}}
            for token, freq in frequencies_df.iterrows():
                if freq['female_male_freq_ratio'] > 1:
                    gender = 'female'
                    other_gender = 'male'
                else:
                    gender = 'male'
                    other_gender = 'female'
                frequency_ratio = min(freq[gender] / freq[other_gender], 5)
                # Let's assume that even the most gendered word is only likely to be
                # used 5 times as often for one gender as the other, to avoid exploding
                # gradient norms
                rel_entropy = freq[gender] * math.log(frequency_ratio)
                assert rel_entropy > 0
                self.relative_entropies[gender][self.dict[token]] = rel_entropy

            logging.info('Assigning tokens to bias bins.')
            (
                self.bin_percentages,
                self.truebins,
                current_bin_fraction,
                log,
            ) = report_bin_biases(
                partial_bin_percentages=partial_bin_percentages,
                frequencies_df=frequencies_df,
                dict_=self.dict,
            )
            for str_ in log:
                logging.info(str_)

            # Print the L2 distance between the original and ideal bin frequencies
            initial_l2dist = self._l2dist(current_bin_fraction)
            logging.info(f'Initial L2 distance: {initial_l2dist:.8f}')

    def reset(self):
        super().reset()
        self._reset_running_histories()

    def _reset_running_histories(self):
        self.generation_history = {'female': [], 'male': []}
        self.running_generation = {'female': Counter(), 'male': Counter()}

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        grp = super().add_cmdline_args(parser, partial_opt=partial_opt)
        grp.add_argument('--alpha', default=1.0, type=float)
        grp.add_argument('--frequencies-file', type=str, default=None)
        grp.add_argument(
            '--bin-percentages',
            type=str,
            default='2,8,40',
            help='The percentage of all generated tokens in each of the first 3 bins, comma-separated',
        )
        grp.add_argument(
            '--penalize-generations',
            type='bool',
            default=False,
            help='Generate until EOS for *every time step* and penalize based on that. Very slow but closer to real-world use!',
        )
        return parser

    def _init_cuda_buffer(self, *args, **kwargs):
        pass

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['num_penalize'] = 0
        self.metrics['steps'] = 0
        self.metrics['ul_weights'] = 0
        # better clean our statistics so not to leak test statistics into train

    def _kldiv(self, p_counter, q_counter) -> float:
        ptotal = sum(p_counter.values())
        qtotal = sum(q_counter.values())
        if ptotal == 0 or qtotal == 0:
            return np.nan
        else:
            kldiv = 0.0
            for word, _ in p_counter.items():
                prob_p = p_counter[word] / ptotal
                prob_q = q_counter[word] / qtotal
                kldiv += prob_p * math.log(1e-20 + (prob_q / prob_p))
            return -kldiv

    def _jsdiv(self, dist1: Counter, dist2: Counter) -> float:
        half = dist1 + dist2
        return 0.5 * self._kldiv(dist1, half) + 0.5 * self._kldiv(dist2, half)

    def _get_bins(self, counts: Counter):
        """
        Get the distribution of token frequency split by the 6 token bins.
        """
        c = Counter()
        for k, v in counts.items():
            if k == self.END_IDX:
                # This isn't a gender-able token
                continue
            c.update({self.truebins.get(k, 'never'): v})
        t = sum(c.values())
        return {k: round_sigfigs(v / t, 4) for k, v in c.items()}

    def _l2dist(self, dist: Dict[str, Dict[str, float]]):
        """
        Return the (squared) L2 distance between the desired + actual token frequencies.
        """
        l2dist = 0.0
        for gend in ['female', 'male']:
            other_gend = 'male' if gend == 'female' else 'female'
            bin_pcts = self.bin_percentages
            l2dist += (
                (dist[gend].get(f'hi_{other_gend}', 0) - bin_pcts[0]) ** 2
                + (dist[gend].get(f'med_{other_gend}', 0) - bin_pcts[1]) ** 2
                + (dist[gend].get(f'lo_{other_gend}', 0) - bin_pcts[2]) ** 2
                + (dist[gend].get(f'lo_{gend}', 0) - bin_pcts[3]) ** 2
                + (dist[gend].get(f'med_{gend}', 0) - bin_pcts[4]) ** 2
                + (dist[gend].get(f'hi_{gend}', 0) - bin_pcts[5]) ** 2
                + (dist[gend].get('never', 0) - 0.0) ** 2
            )
        return l2dist

    def report(self):
        r = super().report()
        r['kldiv_female_male'] = self._kldiv(
            self.running_generation['female'], self.running_generation['male']
        )
        r['kldiv_male_female'] = self._kldiv(
            self.running_generation['male'], self.running_generation['female']
        )
        r['jsdiv'] = self._jsdiv(
            self.running_generation['female'], self.running_generation['male']
        )
        if any(self.running_generation.values()):
            gender_dist = {}
            for gender in ['female', 'male']:
                gender_dist[gender] = self._get_bins(self.running_generation[gender])
                for k, v in gender_dist[gender].items():
                    r[f'{gender}_{k}'] = v

            bias_l2 = 0
            for bin_name in [
                'hi_male',
                'med_male',
                'lo_male',
                'lo_female',
                'med_female',
                'hi_female',
            ]:
                for bin_ in [f'female_{bin_name}', f'male_{bin_name}']:
                    if bin_ not in r:
                        r[bin_] = 0
                if r[f'female_{bin_name}'] + r[f'male_{bin_name}'] == 0:
                    # Can't calculate bias with unpopulated bins
                    bias_l2 = np.nan
                    break
                r[f'bias_{bin_name}'] = (
                    r[f'female_{bin_name}'] - r[f'male_{bin_name}']
                ) / ((r[f'female_{bin_name}'] + r[f'male_{bin_name}']) / 2)
                # This is the difference in the frequencies per gender as a fraction of
                # the frequency averaged between genders
                bias_l2 += r[f'bias_{bin_name}'] ** 2
            r['bias_l2'] = bias_l2

            r['dist_l2'] = self._l2dist(gender_dist)

        return r

    def batchify(self, obs_batch, sort=True):
        """
        Add for each observation which gender is associated with it.

        Allows the unlikelihood technique to know which tokens to penalize: for
        instance, for female observations, all overindexed female tokens will be
        penalized.
        """
        batch = super().batchify(obs_batch, sort=sort)
        batch['is_female'] = torch.BoolTensor(
            [
                obs_batch[i]['category'].split(':')[-1] == 'female'
                for i in batch.valid_indices
            ]
        ).unsqueeze(-1)
        # [bsz, 1] (for broadcasting across token dim)
        return batch

    def compute_loss(self, batch, return_output=False):
        """
        Compute the loss, including the unlikelihood penalty loss term.

        Generate from a randomly chosen time step until EOS, compute the unlikelihood
        penalty for all generated tokens, and add it to the NLL loss.
        """

        if self._last_was_training is not self.is_training:
            self._reset_running_histories()
            self._last_was_training = self.is_training

        nll_loss, model_output = super().compute_loss(batch, True)
        scores, preds, *_ = model_output  # scores is bsz x time x vocab

        if self.penalize_generations:

            label_vec = batch['label_vec']
            num_time_steps = (batch['label_vec'] != self.NULL_IDX).sum(
                dim=0
            ).nonzero().max().item() + 1
            # The last timestep that's not completely NULL_IDX, plus 1

            # Compute the loss per gender for one randomly chosen time step
            if self.is_training:
                step_idx = random.randrange(num_time_steps)
            else:
                # Just generate the *full* sequence to see how well the model does
                step_idx = 0

            with torch.no_grad():
                beam_pred_scores, _ = self._generate(
                    batch=batch,
                    beam_size=self.beam_size,
                    max_ts=self.opt['label_truncate'],
                    prefix_tokens=label_vec[:, :step_idx],
                )

                # forward pass to create graph for beam search case
                generations = [g for (g, s) in beam_pred_scores]
                gentoks = torch.nn.utils.rnn.pad_sequence(
                    generations, batch_first=True, padding_value=self.NULL_IDX
                )
                # strip the BOS tokens
                gentoks = gentoks[:, 1:]

            # Strip out the prefix tokens, which were not generated
            assert torch.equal(label_vec[:, :step_idx], gentoks[:, :step_idx])
            gentoks = gentoks[:, step_idx:]

            ul_losses = self._compute_loss_per_gender(batch=batch, gentoks=gentoks)
            loss = nll_loss + self.opt['alpha'] * (
                ul_losses['female'] + ul_losses['male']
            )

        else:
            ul_losses = self._compute_loss_per_gender(
                batch=batch, gentoks=preds, scores=scores
            )
            loss = nll_loss + self.opt['alpha'] * (
                ul_losses['female'] + ul_losses['male']
            )

        self.global_metrics.add('total_loss', AverageMetric(loss))
        # `loss` is already a metric, but `sweep-results` suppresses reporting of it.
        # Thus, adding a new loss metric here is a hack to get this metric to show up in
        # the `sweep-results` output

        if return_output:
            return loss, model_output
        else:
            return loss

    def _compute_loss_per_gender(
        self, batch: Batch, gentoks: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute unlikelihood penality losses per gender.

        For observations of each gender, figure out how to penalize all generated
        tokens, and return the mean penalty per token.
        """

        ul_losses = dict()
        gen_mask = gentoks != self.NULL_IDX
        gender_mask = {'female': batch.is_female, 'male': ~batch.is_female}
        for gender in ['female', 'male']:

            this_gender_gen_mask = gen_mask * gender_mask[gender]

            # Add all tokens to the counter
            self.generation_history[gender].append(
                Counter(gentoks[this_gender_gen_mask].view(-1).tolist())
            )
            self.running_generation[gender] += self.generation_history[gender][-1]

            # Loop over tokens and figure out how much to penalize each one by
            to_penalize = {
                token: self.relative_entropies[gender][token]
                for token in self.generation_history[gender][-1].keys()
                if token in self.relative_entropies[gender]
            }

            self.global_metrics.add('num_penalize', SumMetric(len(to_penalize)))
            ul_weights = torch.zeros(this_gender_gen_mask.shape)
            ul_mask = torch.zeros_like(this_gender_gen_mask)
            for wordid, weight in to_penalize.items():
                word_mask = (gentoks == wordid) * this_gender_gen_mask
                assert (
                    torch.count_nonzero(word_mask) > 0
                ), 'Mask should not have filtered out this token completely'
                ul_mask = ul_mask | word_mask
                ul_weights[word_mask] = weight
            ul_weights = ul_weights.to(this_gender_gen_mask.device)
            self.global_metrics.add(
                'ul_weights', AverageMetric(ul_weights[ul_mask].mean())
            )

            # and whack it
            if scores is None:
                model_output = self.model(*self._model_input(batch), ys=gentoks)
                scores, *_ = model_output
            downweight = gentoks[ul_mask]

            almost_scores = F.log_softmax(scores[ul_mask], dim=-1)
            ul_scores = almost_scores[torch.arange(len(downweight)), downweight]

            clamp_min = 1e-6 if self.opt['fp16'] else 1e-20
            clamp_max = NEAR_INF_FP16 if self.opt['fp16'] else NEAR_INF

            ul_loss = (
                -(torch.log(torch.clamp(1 - ul_scores.exp(), min=clamp_min)))
                * torch.clamp(ul_weights[ul_mask], max=clamp_max)
            ).sum()
            num_ul = ul_mask.sum()

            self.global_metrics.add('ul_loss', AverageMetric(ul_loss, num_ul))
            self.global_metrics.add('ul_num_tokens', SumMetric(num_ul))

            ul_losses[gender] = div(ul_loss, num_ul)

        return ul_losses


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
