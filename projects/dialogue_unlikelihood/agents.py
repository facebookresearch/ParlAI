#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from nltk import ngrams

from parlai.core.torch_generator_agent import PPLMetric
from parlai.agents.image_seq2seq.image_seq2seq import ImageSeq2seqAgent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.metrics import AverageMetric, SumMetric, GlobalAverageMetric
from parlai.utils.misc import round_sigfigs
from parlai.utils.io import PathManager


def div(x, y):
    if y == 0:
        return x
    else:
        return x / y


class NGramIterator:
    """
    N-Gram iterator for a list.
    """

    def __init__(self, lst, n):
        self.lst = lst
        self.n = n
        self.max = len(lst) - n

    def __iter__(self):
        self.counter = -1
        return self

    def __next__(self):
        self.counter += 1
        if self.counter > self.max:
            raise StopIteration
        return tuple(self.lst[self.counter : self.counter + self.n])


class RewardUnlikelihoodAgentTrait(object):
    """
    Abstract Trait.

    Applies unlikelihood loss based on the presence of negative rewards in the task.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        grp = super(RewardUnlikelihoodAgentTrait, cls).add_cmdline_args(argparser)
        grp.add_argument('--alpha', default=1.0, type=float)

    def batchify(self, obs_batch, **kwargs):
        batch = super().batchify(obs_batch, **kwargs)
        rewards = torch.FloatTensor(
            [float(o.get('reward', 0)) for o in batch.observations]
        ).to(batch.text_vec.device)
        batch['rewards'] = rewards
        return batch

    def _dummy_batch(self, batchsize, maxlen):
        batch = super()._dummy_batch(batchsize, maxlen)
        batch['rewards'] = torch.ones(batchsize, dtype=torch.long).cuda()
        return batch

    def compute_loss(self, batch, return_output=False):
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output  # scores is bsz x time x vocab

        scores = F.log_softmax(scores, dim=-1)
        scores_view = scores.view(-1, scores.size(-1))
        targets = batch.label_vec
        targets_view = targets.view(-1)

        notnull = targets.ne(self.NULL_IDX)
        if self.is_training:
            # note it's >= because convai2 and other teachers all provide a 0 reward
            mle_notnull = notnull & (batch.rewards >= 0).unsqueeze(1).expand_as(notnull)
        else:
            mle_notnull = notnull

        mle_loss = (
            F.nll_loss(
                scores_view, targets_view, ignore_index=self.NULL_IDX, reduction='none'
            ).view_as(mle_notnull)
            * mle_notnull.float()
        ).sum()

        # limit loss to only the positive rewards
        mle_target_tokens = mle_notnull.long().sum()
        correct = ((targets == preds) * mle_notnull).sum()
        self.global_metrics.add('token_acc', AverageMetric(correct, mle_target_tokens))
        self.global_metrics.add('nll_loss', AverageMetric(mle_loss, mle_target_tokens))
        self.global_metrics.add('ppl', PPLMetric(mle_loss, mle_target_tokens))
        if mle_target_tokens > 0:
            mle_loss /= mle_target_tokens  # average loss per token

        if not self.is_training:
            if return_output:
                return (mle_loss, model_output)
            else:
                return mle_loss

        # and now we want the unlikelihood loss on the negative examples
        ul_notnull = notnull & (batch.rewards < 0).unsqueeze(1).expand_as(notnull)
        ul_target_tokens = ul_notnull.long().sum()
        range_ = torch.arange(targets_view.size(0)).to(batch.label_vec.device)
        ul_scores = scores_view[range_, targets_view]
        clamp_min = 1e-6 if self.opt['fp16'] else 1e-20
        ul_loss = (
            -torch.log(torch.clamp(1.0 - ul_scores.exp(), min=clamp_min)).view_as(
                ul_notnull
            )
            * ul_notnull.float()
        ).sum()
        self.global_metrics.add('ul_loss', AverageMetric(ul_loss, ul_target_tokens))
        if ul_target_tokens > 0:
            ul_loss /= ul_target_tokens

        loss = mle_loss + self.opt['alpha'] * ul_loss

        if return_output:
            return (loss, model_output)
        else:
            return loss


class RepetitionUnlikelihoodAgentTrait(object):
    """
    Abstract Trait.

    Applies unliikelihood loss to repetition some proportion of train steps by
    generating, marking repeats, and calculating loss accordingly.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.pred_logsoftmax = torch.nn.LogSoftmax(dim=2)

    @classmethod
    def add_cmdline_args(cls, argparser):
        print(super())
        grp = super().add_cmdline_args(argparser)
        grp.add_argument('--seq-ul-ratio', default=0.5, type=float)
        grp.add_argument('--seq-ul-n', default=4, type=int)
        grp.add_argument('--mask-n', default=100, type=int)
        grp.add_argument('--ctxt-beta', default=0.5, type=float)
        grp.add_argument('--crep-pen', default='crep', type=str)

    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        pass

    def _count_n_grams(self, token_lst, n):
        n_grams = defaultdict(int)
        for n_gram in NGramIterator(token_lst, n):
            n_grams[n_gram] += 1
        return n_grams

    def compute_loss(self, batch, return_output=False):
        if self.is_training and (torch.rand(1).item() >= self.opt['seq_ul_ratio']):
            total_loss, model_output = super().compute_loss(batch, return_output=True)
            # No sequence level unlikelihood
            if return_output:
                return total_loss, model_output
            else:
                return total_loss

        # Generate
        clamp_min = 1e-6 if self.opt['fp16'] else 1e-20
        maxlen = self.label_truncate or 256
        with torch.no_grad():
            beam_pred_scores, _ = self._generate(batch, self.beam_size, maxlen)

        # forward pass to create graph for beam search case
        generations = [g[1:] for (g, s) in beam_pred_scores]
        pred_toks = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True)
        model_output = self.model(*self._model_input(batch), ys=pred_toks)
        logits, preds, _ = model_output

        # construct mask marking repeats
        n = self.opt['seq_ul_n']  # label n-grams
        crep_mask = torch.zeros_like(pred_toks).type_as(logits)
        lrep_mask = torch.zeros_like(pred_toks).type_as(logits)

        for i, gen in enumerate(generations):
            gen_i = gen.tolist()

            # Collect context ngrams
            context_i = batch.text_vec[i].tolist()
            context_n_grams = self._count_n_grams(context_i, n)

            seen_n_grams = defaultdict(int)

            # penalize if there is a context repeat
            for j, n_gram in enumerate(NGramIterator(gen_i, n)):
                if context_n_grams[n_gram] > 0:
                    crep_mask[i, j : j + n] = 1

            # penalize if there is a label repeat
            for j, n_gram in enumerate(NGramIterator(gen_i, n)):
                if seen_n_grams[n_gram] > 0:
                    lrep_mask[i, j : j + n] = 1
                seen_n_grams[n_gram] += 1

        # Compute unlikelihood loss
        lprobs = self.pred_logsoftmax(logits)
        pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
        one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=clamp_min).view(
            pred_toks.size(0), pred_toks.size(1)
        )

        mask = ((1 - self.opt['ctxt_beta']) * lrep_mask) + (
            self.opt['ctxt_beta'] * crep_mask
        )

        ul_loss = -(torch.log(one_minus_probs)) * mask
        total_loss = div(ul_loss.sum(), mask.sum())
        self.record_local_metric(
            'ul_loss', AverageMetric.many(ul_loss.sum(dim=-1), mask.sum(dim=-1))
        )

        if not self.is_training:
            # in eval mode, we want metrics (e.g. PPL) provided by tga's compute_loss
            _, _ = super().compute_loss(batch, return_output=True)

        if return_output:
            return total_loss, model_output
        return total_loss

    def _add_generation_metrics(self, batch, preds):
        self._ngram_metrics(batch, preds)

    def _ngram_metrics(self, batch, preds):
        text_vecs_cpu = batch.text_vec.cpu()
        lrep, crep = 0, 0
        total_pred_ngs = 0
        n = self.opt['seq_ul_n']
        for i, pred in enumerate(preds):
            pred_token_list = pred.tolist()
            if self.END_IDX in pred_token_list:
                pred_token_list = pred_token_list[
                    : pred_token_list.index(self.END_IDX)
                ]  # remove possible padding
            if self.START_IDX in pred_token_list:
                pred_token_list = pred_token_list[
                    pred_token_list.index(self.START_IDX) :
                ]
            pred_ngs = [ng for ng in ngrams(pred_token_list, n)]
            pred_counter = Counter(pred_ngs)
            total_pred_ngs += len(pred_ngs)
            lrep += len(pred_ngs) - len(pred_counter)

            text_token_list = text_vecs_cpu[i].tolist()
            if self.NULL_IDX in text_token_list:
                text_token_list = text_token_list[
                    : text_token_list.index(self.NULL_IDX)
                ]  # remove possible padding
            context_counter = Counter([ng for ng in ngrams(text_token_list, n)])

            for ng in pred_counter:
                if ng in context_counter:
                    crep += pred_counter[ng]

        self.global_metrics.add(
            'lrep_%dgrams' % n, GlobalAverageMetric(lrep, total_pred_ngs)
        )
        self.global_metrics.add(
            'crep_%dgrams' % n, GlobalAverageMetric(crep, total_pred_ngs)
        )


class _VocabUnlikelihoodTrait(object):
    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['high_freq'] = 0
        self.metrics['gold_high'] = 0

    def _kldiv(self, p_counter, q_counter) -> float:
        ptotal = sum(p_counter.values())
        qtotal = sum(q_counter.values())
        kldiv = 0.0
        for word, _ in p_counter.items():
            prob_p = p_counter[word] / ptotal
            prob_q = q_counter[word] / qtotal
            kldiv += prob_p * math.log(1e-20 + (prob_q / prob_p))
        return -kldiv

    def _jsdiv(self, dist1: Counter, dist2: Counter) -> float:
        half = dist1 + dist2
        return 0.5 * self._kldiv(dist1, half) + 0.5 * self._kldiv(dist2, half)

    def report(self):
        report = super().report()
        report['kldiv_humgen'] = self._kldiv(
            self.running_human, self.running_generation
        )
        report['kldiv_genhum'] = self._kldiv(
            self.running_generation, self.running_human
        )
        report['jsdiv'] = self._jsdiv(self.running_human, self.running_generation)
        return report


class SequenceVocabUnlikelihoodAgentTrait(_VocabUnlikelihoodTrait):
    """
    Abstract Trait.

    Applies unlikelihood loss to vocabulary distributiion by generating, calculating
    proportion of tokens per vocabulary frequency bin, and computing loss accordingly
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.NUM_STEPS = opt['queue_size']
        if shared is None:
            self._reset_running_histories()
            self._last_was_training = True
            self.truebins = {}
            counts_file = self.opt['counts_file']
            if counts_file is None:
                counts_file = os.path.join(
                    os.path.dirname(self.opt['model_file']), 'counts.txt'
                )
                if not PathManager.exists(counts_file):
                    raise RuntimeError(
                        'Please give a --counts-file to use vocab unlikelihood'
                    )
            with PathManager.open(counts_file) as f:
                for line in f:
                    record = json.loads(line)
                    self.truebins[record['word_id']] = record['bin']

    def reset(self):
        super().reset()
        self._reset_running_histories()

    def _reset_running_histories(self):
        self.generation_history = []
        self.running_generation = Counter()
        self.human_history = []
        self.running_human = Counter()

    @classmethod
    def add_cmdline_args(cls, argparser):
        print(super())
        grp = super().add_cmdline_args(argparser)
        grp.add_argument('--alpha', default=1.0, type=float)
        grp.add_argument('--queue-size', default=32, type=int)
        grp.add_argument(
            '--weighting', choices={'uniform', 'logdiff', 'kldiv'}, default='uniform'
        )
        grp.add_argument('--threshold', type=float, default=1e-3)
        grp.add_argument('--counts-file', type=str, default=None)

    def _init_cuda_buffer(self, *args, **kwargs):
        pass

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['num_penalize'] = 0
        self.metrics['steps'] = 0
        self.metrics['hum_toks'] = 0
        self.metrics['gen_toks'] = 0
        self.metrics['ul_weights'] = 0
        # better clean our statistics so not to leak test statistics into train

    def _get_bins(self, counts: Counter):
        c = Counter()
        for k, v in counts.items():
            c.update({self.truebins.get(k, 'never'): v})
        t = sum(c.values())
        return {k: round_sigfigs(v / t, 4) for k, v in c.items()}

    def _l2dist(self, bins):
        return (
            (bins.get('frequent', 0) - 0.4) ** 2
            + (bins.get('medium', 0) - 0.3) ** 2
            + (bins.get('rare', 0) - 0.2) ** 2
            + (bins.get('veryrare', 0) - 0.1) ** 2
            + (bins.get('never', 0) - 0.0) ** 2
        )

    def report(self):
        r = super().report()
        if self.running_generation and self.running_human:
            for k, v in self._get_bins(self.running_human).items():
                r[f'humdist_{k}'] = v
            gendist = self._get_bins(self.running_generation)
            for k, v in gendist.items():
                r[f'gendist_{k}'] = v
            r['dist_l2'] = self._l2dist(gendist)

        return r

    def compute_loss(self, batch, return_output=False):
        if self._last_was_training is not self.is_training:
            self._reset_running_histories()
            self._last_was_training = self.is_training

        nll_loss, model_output = super().compute_loss(batch, True)
        scores, preds, *_ = model_output  # scores is bsz x time x vocab
        targets = batch.label_vec
        notnull = targets != self.NULL_IDX

        with torch.no_grad():
            beam_pred_scores, _ = self._generate(
                batch, self.beam_size, self.opt['label_truncate']
            )

            # forward pass to create graph for beam search case
            generations = [g for (g, s) in beam_pred_scores]
            gentoks = torch.nn.utils.rnn.pad_sequence(
                generations, batch_first=True, padding_value=self.NULL_IDX
            )
            # strip the BOS tokens
            gentoks = gentoks[:, 1:]

        # find everything we oversampled
        gen_mask = gentoks != self.NULL_IDX
        self.generation_history.append(Counter(gentoks[gen_mask].view(-1).tolist()))
        self.human_history.append(Counter(targets[notnull].view(-1).tolist()))
        self.running_generation += self.generation_history[-1]
        self.running_human += self.human_history[-1]

        if len(self.generation_history) > self.NUM_STEPS:
            if not self.is_training:
                # we want a running history of word usage
                self.running_generation -= self.generation_history.pop(0)
                self.running_human -= self.human_history.pop(0)
        else:
            if return_output:
                return nll_loss, model_output
            else:
                return nll_loss

        gen_sum = sum(self.running_generation.values())
        hum_sum = sum(self.running_human.values())

        # what did we oversample?
        if self.opt['weighting'] == 'logdiff':
            to_penalize = {
                w: (v / gen_sum) - (self.running_human.get(w, 0) / hum_sum)
                for w, v in self.running_generation.items()
            }
            to_penalize = {
                w: v for w, v in to_penalize.items() if v >= self.opt['threshold']
            }
            to_penalize = {w: math.log(v / 0.001) for w, v in to_penalize.items()}
        elif self.opt['weighting'] == 'uniform':
            to_penalize = {
                w: (v / gen_sum) - (self.running_human.get(w, 0) / hum_sum)
                for w, v in self.running_generation.items()
            }
            to_penalize = {
                w: 1 for w, v in to_penalize.items() if v >= self.opt['threshold']
            }
        elif self.opt['weighting'] == 'kldiv':
            to_penalize = {
                w: (
                    self.running_generation[w] / gen_sum,
                    self.running_human[w] / hum_sum,
                )
                for w, v in self.running_human.items()
                if w in self.running_generation
            }
            to_penalize = {
                w: (p_gen, p_hum)
                for w, (p_gen, p_hum) in to_penalize.items()
                if p_gen > p_hum
            }
            to_penalize = {
                w: p_gen * (math.log(p_gen) - math.log(p_hum))
                for w, (p_gen, p_hum) in to_penalize.items()
            }
            to_penalize = {
                k: v for k, v in to_penalize.items() if v > self.opt['threshold']
            }
        else:
            raise ValueError

        self.global_metrics.add('num_penalize', SumMetric(len(to_penalize)))

        ul_weights = torch.zeros(gen_mask.shape)
        ul_mask = torch.zeros_like(gen_mask)
        for wordid, weight in to_penalize.items():
            ul_mask = ul_mask | (gentoks == wordid)
            ul_weights[gentoks == wordid] = weight
        ul_weights = ul_weights.to(gen_mask.device)
        self.global_metrics.add('ul_weights', AverageMetric(ul_weights[ul_mask].mean()))

        # and whack it
        model_output = self.model(*self._model_input(batch), ys=gentoks)
        scores, *_ = model_output
        downweight = gentoks[ul_mask]

        almost_scores = F.log_softmax(scores[ul_mask], dim=-1)
        ul_scores = almost_scores[torch.arange(len(downweight)), downweight]

        clamp_min = 1e-6 if self.opt['fp16'] else 1e-20

        ul_loss = (
            -(torch.log(torch.clamp(1 - ul_scores.exp(), min=clamp_min)))
            * ul_weights[ul_mask]
        ).sum()
        num_ul = ul_mask.sum()

        self.global_metrics.add('ul_loss', AverageMetric(ul_loss, num_ul))
        self.global_metrics.add('ul_num_tokens', SumMetric(num_ul))

        ul_loss = div(ul_loss, num_ul)

        if len(self.generation_history) < self.NUM_STEPS:
            loss = nll_loss
        else:
            loss = nll_loss + self.opt['alpha'] * ul_loss

        if return_output:
            return (loss, model_output)
        else:
            return loss


class TransformerSequenceVocabUnlikelihoodAgent(
    SequenceVocabUnlikelihoodAgentTrait, TransformerGeneratorAgent
):
    """
    Example usage:

    -t convai2 -m parlai_internal.projects.dontsaythat.agents:TransformerSequenceVocabUnlikelihoodAgent
    """

    pass


class TransformerUnlikelihoodAgent(
    RewardUnlikelihoodAgentTrait, TransformerGeneratorAgent
):
    """
    Example usage:

    -t convai2 -m parlai_internal.projects.dontsaythat.agents:TransformerUnlikelihoodAgent
    """

    pass


class RepetitionUnlikelihoodAgent(RepetitionUnlikelihoodAgentTrait, ImageSeq2seqAgent):
    """
    Example usage:

    -t convai2 -m parlai_internal.projects.dontsaythat.agents:RepetitionUnlikelihoodAgent
    """

    pass
