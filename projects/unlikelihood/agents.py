#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import json
from collections import defaultdict, Counter
from nltk import ngrams

from parlai.utils.misc import round_sigfigs
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.torch_agent import TorchAgent, Batch, Output

from parlai_internal.agents.parlall.agents import ParlallAgent


class RewardUnlikelihoodAgentTrait(object):
    """
    Abstract Trait.
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

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['ul_loss'] = 0.0
        self.metrics['ul_num_tokens'] = 0
        self.metrics['mask_loss'] = 0.0
        self.metrics['mask_num_tokens'] = 0
        self.metrics['minmask_loss'] = 0.0
        self.metrics['minmask_num_tokens'] = 0

    def report(self):
        report = super().report()
        if self.metrics['ul_num_tokens']:
            report['ul_loss'] = round_sigfigs(
                self.metrics['ul_loss'] / self.metrics['ul_num_tokens']
            )
        return report

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
            )
            * mle_notnull.view(-1).float()
        ).sum()

        # limit loss to only the positive rewards
        mle_target_tokens = mle_notnull.long().sum().item()
        correct = ((targets == preds) * mle_notnull).sum().item()
        self.metrics['correct_tokens'] += correct
        self.metrics['nll_loss'] += mle_loss.item()
        self.metrics['num_tokens'] += mle_target_tokens
        if mle_target_tokens > 0:
            mle_loss /= mle_target_tokens  # average loss per token

        if not self.is_training:
            if return_output:
                return (mle_loss, model_output)
            else:
                return mle_loss

        # and now we want the unlikelihood loss on the negative examples
        ul_notnull = notnull & (batch.rewards < 0).unsqueeze(1).expand_as(notnull)
        ul_target_tokens = ul_notnull.long().sum().item()
        range_ = torch.arange(targets_view.size(0)).to(batch.label_vec.device)
        ul_scores = scores_view[range_, targets_view]
        clamp_min = 1e-6 if self.opt['fp16'] else 1e-20
        ul_loss = (
            -torch.log((1 - ul_scores.exp()).clamp_(1e-6)) * ul_notnull.view(-1).float()
        ).sum()
        #print(ul_notnull)
        #import pdb; pdb.set_trace()
        self.metrics['ul_loss'] += ul_loss.item()
        self.metrics['ul_num_tokens'] += ul_target_tokens
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
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.pred_logsoftmax = torch.nn.LogSoftmax(dim=2)

    @classmethod
    def add_cmdline_args(cls, argparser):
        grp = super(RepetitionUnlikelihoodAgentTrait, cls).add_cmdline_args(argparser)
        grp.add_argument('--seq_ul_ratio', default=0.5, type=float)
        grp.add_argument('--seq_ul_n', default=4, type=int)
        grp.add_argument('--mask-n', default=100, type=int)
        grp.add_argument('--ctxt_beta', default=0.5, type=float)
        grp.add_argument('--train_to_convergence', default=True, type=bool)
        grp.add_argument('--crep_pen', default='crep', type=str)

    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        pass

    def receive_metrics(self, metrics_dict):
        super().receive_metrics(metrics_dict)
        if self.opt['train_to_convergence']:
            return
        if False:
            if self.opt['task'] == 'convai2':
                if metrics_dict['ppl'] > 12 or metrics_dict['f1'] < 0.19:
                    raise StopTrainException('Convai2 PPL is too high or f1 is too low')
            elif self.opt['task'] == 'internal:eli5':
                if metrics_dict['ppl'] > 30 or metrics_dict['f1'] < 0.10:
                    raise StopTrainException('ELI5 PPL is too high or f1 is too low')
            elif self.opt['task'] == 'wizard_of_wikipedia:GeneratorTeacher':
                if metrics_dict['ppl'] > 15 or metrics_dict['f1'] < 0.20:
                    raise StopTrainException('Wizard PPL is too high or f1 is too low')

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['ul_loss'] = 0.0
        self.metrics['ul_num_tokens'] = 0
        self.metrics['mask_loss'] = 0.0
        self.metrics['mask_num_tokens'] = 0
        self.metrics['minmask_loss'] = 0.0
        self.metrics['minmask_num_tokens'] = 0
        self.metrics['ppl_irep4'] = 0
        for n in range(1, 5):
            self.metrics['irep_%dgrams' % n] = 0.0
            self.metrics['ihrep_%dgrams' % n] = 0.0
            self.metrics['crep_%dgrams' % n] = 0.0
            self.metrics['wcrep_%dgrams' % n] = 0.0
            self.metrics['wocrep_%dgrams' % n] = 0.0
            self.metrics['hcrep_%dgrams' % n] = 0.0
            self.metrics['pred_%dgrams' % n] = 0.0
            self.metrics['label_%dgrams' % n] = 0.0

    def report(self):
        report = super().report()
        if self.metrics['ul_num_tokens']:
            report['ul_loss'] = round_sigfigs(
                self.metrics['ul_loss'] / self.metrics['ul_num_tokens']
            )
        if self.metrics['mask_num_tokens']:
            report['mask_loss'] = round_sigfigs(
                self.metrics['mask_loss'] / self.metrics['mask_num_tokens']
            )
        if self.metrics['minmask_num_tokens']:
            report['minmask_loss'] = round_sigfigs(
                self.metrics['minmask_loss'] / self.metrics['minmask_num_tokens']
            )
        if not self.is_training:
            for n in range(1, 5):
                report['irep_%dgrams' % n] = (
                    self.metrics['irep_%dgrams' % n] / self.metrics['pred_%dgrams' % n]
                )
                report['ihrep_%dgrams' % n] = (
                    self.metrics['ihrep_%dgrams' % n]
                    / self.metrics['label_%dgrams' % n]
                )
                report['crep_%dgrams' % n] = (
                    self.metrics['crep_%dgrams' % n] / self.metrics['pred_%dgrams' % n]
                )
                report['wcrep_%dgrams' % n] = (
                    self.metrics['wcrep_%dgrams' % n] / self.metrics['pred_%dgrams' % n]
                )
                report['wocrep_%dgrams' % n] = (
                    self.metrics['wocrep_%dgrams' % n] / self.metrics['pred_%dgrams' % n]
                )
                report['hcrep_%dgrams' % n] = (
                    self.metrics['hcrep_%dgrams' % n]
                    / self.metrics['label_%dgrams' % n]
                )
            report['ppl_irep4'] = report['ppl'] + 100 * report['irep_4grams']
        return report

    def compute_loss(self, batch, return_output=False):
        if not (self.is_training and torch.rand(1).item() < self.opt['seq_ul_ratio']):
            total_loss, model_output = super().compute_loss(batch, return_output=True)
        else:
            #total_loss, model_output = super().compute_loss(batch, return_output=True)
            # generate
            clamp_min = 1e-6 if self.opt['fp16'] else 1e-20
            maxlen = self.label_truncate or 256
            with torch.no_grad():
                beam_pred_scores, _ = self._generate(batch, self.beam_size, maxlen)

            # forward pass to create graph for beam search case
            generations = [g[1:] for (g, s) in beam_pred_scores]
            # pred_toks = [g[1:] for (g, s) in beam_pred_scores]
            # pred_toks = torch.nn.utils.rnn.pad_sequence(pred_toks, batch_first=True)
            pred_toks = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True)
            # import pdb;pdb.set_trace()
            #pred_toks =  pred_toks[:,1:]
            model_output = self.model(*self._model_input(batch), ys=pred_toks)
            # print("MODEL OUTPUT", model_output)
            logits, preds, _ = model_output

            # construct mask marking repeats
            n = self.opt['seq_ul_n']
            crep_mask = torch.zeros_like(pred_toks).type_as(logits)
            lrep_mask = torch.zeros_like(pred_toks).type_as(logits)
            lrep2_mask = torch.zeros_like(pred_toks).type_as(logits)
            mask_min = []
            mask_len = []
            for i, x in enumerate(generations):
                context_ngs = defaultdict(int)
                seen_ngs = defaultdict(int)
                seen_ngs_cnt = 0
                mmin = len(x)
                label_ngs = defaultdict(int)
                # Add context ngrams
                context_x = batch.text_vec[i]
                context_xl = context_x.tolist()
                for j in range(len(context_x) - n):
                    ng = tuple(context_xl[j : j + n])
                    context_ngs[ng] += 1
                label_x = batch.label_vec[i]
                label_xl = label_x.tolist()
                for j in range(len(label_xl) - n):
                    ng = tuple(label_xl[j : j + n])
                    label_ngs[ng] += 1

                xl = x.tolist()
                joined_ngrams = 0
                if  self.opt['crep_pen'] == 'bog':
                    # bog: make everything unlikely except the first
                    # occurrences of the bag of gold tokens
                    seen_ngs = defaultdict(int)
                    gold_pos = defaultdict(int)
                    for j in range(len(label_xl)):
                        ng = int(label_xl[j])
                        gold_pos[ng] += 1
                    for j in range(len(x)):
                        ng = int(x[j])
                        if seen_ngs[ng] >= gold_pos[ng]:
                            crep_mask[i, j] = 1
                            if seen_ngs_cnt == 0:
                                mmin = j
                            seen_ngs_cnt += 1
                        seen_ngs[ng] += 1
                else:
                    for j in range(len(x) - n + 1):
                        ng = tuple(xl[j : j + n])
                        if context_ngs[ng] > 0:
                            if self.opt['crep_pen'] == 'crep':
                                crep_mask[i, j : j + n] = 1
                            elif self.opt['crep_pen'] == 'wcrep':
                                if label_ngs[ng] == 0:
                                    crep_mask[i, j : j + n] = 1
                            elif self.opt['crep_pen'] == 'wocrep':
                                if seen_ngs[ng] >= label_ngs[ng]:
                                    crep_mask[i, j : j + n] = 1
                        if seen_ngs[ng] > 0:
                            lrep_mask[i, j : j + n] = 1
                            if seen_ngs_cnt == 0:
                                mmin = j
                            seen_ngs_cnt += 1
                            if seen_ngs_cnt <= self.opt['mask_n']:
                                lrep2_mask[i, j : j + n] = 1
                        seen_ngs[ng] += 1
                mask_min.append(mmin / len(x))
                mask_len.append(len(x))

            lprobs = self.pred_logsoftmax(logits)
            pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(
                1, pred_toks.view(-1, 1)
            )
            one_minus_probs = torch.clamp(
                (1.0 - pred_lprobs.exp()), min=clamp_min
            ).view(pred_toks.size(0), pred_toks.size(1))

            mask = ((1 - self.opt['ctxt_beta']) * lrep2_mask) + (
                self.opt['ctxt_beta'] * crep_mask
            )
            #mask = lrep2_mask

            loss = -torch.log(one_minus_probs) * mask
            ul_loss = loss.sum()
            ul_num_tokens = mask.sum()
            self.metrics['ul_loss'] += ul_loss.item()
            self.metrics['ul_num_tokens'] += ul_num_tokens.item()
            if ul_num_tokens > 0:
                total_loss = ul_loss / ul_num_tokens
            else:
                total_loss, model_output = super().compute_loss(
                    batch, return_output=True
                )

            mask_loss =  lrep_mask.sum()
            mask_num_tokens = sum(mask_len)
            self.metrics['mask_loss'] += mask_loss
            self.metrics['mask_num_tokens'] += mask_num_tokens

            minmask_loss = sum(mask_min)
            minmask_num_tokens = len(mask_min)
            self.metrics['minmask_loss'] += minmask_loss
            self.metrics['minmask_num_tokens'] += minmask_num_tokens

            #import pdb; pdb.set_trace()

        if return_output:
            return total_loss, model_output
        return total_loss

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        bsz = batch.text_vec.size(0)
        self.model.eval()
        cand_scores = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss = self.compute_loss(batch)  # noqa: F841  we need the side effects
            self.metrics['loss'] += loss.item()

        maxlen = self.label_truncate or 256
        beam_preds_scores, _ = self._generate(batch, self.beam_size, maxlen)
        preds, scores = zip(*beam_preds_scores)

        self._ngram_metrics(batch, preds)

        cand_choices = None
        # TODO: abstract out the scoring here
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(*self._model_input(batch))
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = padded_tensor(
                    batch.candidate_vecs[i], self.NULL_IDX, self.use_cuda
                )
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        text = [self._v2t(p) for p in preds] if preds is not None else None
        return Output(text, cand_choices)

    def _ngram_metrics(self, batch, preds):
        text_vecs_cpu = batch.text_vec.cpu()
        label_vecs_cpu = batch.label_vec.cpu()
        for n in range(1, 5):
            i_rep, i_human_rep, crep, wcrep, wocrep, hcrep = 0, 0, 0, 0, 0, 0
            total_pred_ngs, total_label_ngs = 0, 0
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
                i_rep += len(pred_ngs) - len(pred_counter)

                label_token_list = label_vecs_cpu[i].tolist()
                if self.END_IDX in label_token_list:
                    label_token_list = label_token_list[
                        : label_token_list.index(self.END_IDX)
                    ]  # remove possible padding
                label_ngs = [ng for ng in ngrams(label_token_list, n)]
                label_counter = Counter(label_ngs)
                total_label_ngs += len(label_ngs)
                i_human_rep += len(label_ngs) - len(label_counter)

                text_token_list = text_vecs_cpu[i].tolist()
                if self.NULL_IDX in text_token_list:
                    text_token_list = text_token_list[
                        : text_token_list.index(self.NULL_IDX)
                    ]  # remove possible padding
                context_counter = Counter([ng for ng in ngrams(text_token_list, n)])

                for ng in pred_counter:
                    if ng in context_counter:
                        crep += pred_counter[ng]
                        if pred_counter[ng] > label_counter[ng]:
                            wocrep += pred_counter[ng] - label_counter[ng]
                        if label_counter[ng] == 0:
                            wcrep += pred_counter[ng]

                for ng in label_counter:
                    if ng in context_counter:
                        hcrep += label_counter[ng]

            self.metrics['irep_%dgrams' % n] += i_rep
            self.metrics['ihrep_%dgrams' % n] += i_human_rep
            self.metrics['pred_%dgrams' % n] += total_pred_ngs
            self.metrics['crep_%dgrams' % n] += crep
            self.metrics['wcrep_%dgrams' % n] += wcrep
            self.metrics['wocrep_%dgrams' % n] += wocrep
            self.metrics['hcrep_%dgrams' % n] += hcrep
            self.metrics['label_%dgrams' % n] += total_label_ngs


class _VocabUnlikelihoodTrait(object):
    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['ul_loss'] = 0.0
        self.metrics['ul_num_tokens'] = 0
        self.metrics['high_freq'] = 0
        self.metrics['gold_high'] = 0
        self.metrics['words'] = Counter()
        self.metrics['human_words'] = Counter()
        self.metrics['ul_total_toks'] = 0

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
        if self.metrics['ul_num_tokens']:
            report['ul_loss'] = round_sigfigs(
                self.metrics['ul_loss'] / self.metrics['ul_num_tokens']
            )
        if self.metrics['ul_total_toks']:
            report['ul_pct'] = (
                self.metrics['ul_num_tokens'] / self.metrics['ul_total_toks']
            )

        report['most_common'] = self._freq2string(self.running_generation)
        report['human_common'] = self._freq2string(self.running_human)

        report['size_run_hum'] = sum(self.running_human.values())
        report['size_run_gen'] = sum(self.running_generation.values())

        report['kldiv_humgen'] = self._kldiv(
            self.running_human, self.running_generation
        )
        report['kldiv_genhum'] = self._kldiv(
            self.running_generation, self.running_human
        )
        report['jsdiv'] = self._kldiv(self.running_human, self.running_generation)
        return report

    def _freq2string(self, freqdict):
        s = []
        most_common = freqdict.most_common(10)
        total = sum(freqdict.values())
        for k, v in most_common:
            n = self.dict.ind2tok[k]
            r = v / total
            s.append(f'{n} {r:.3g}')
        return "  ".join(s)


class TokenVocabUnlikelihoodAgentTrait(_VocabUnlikelihoodTrait):
    """
    Abstract Trait.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        grp = super().add_cmdline_args(argparser)
        grp.add_argument('--alpha', default=1.0, type=float)
        grp.add_argument('--limit', default=64, type=int)
        return grp

    def _init_cuda_buffer(self, *args, **kwargs):
        pass

    def compute_loss(self, batch, return_output=False):
        # hardcode no special tokens
        min_limit = 0
        limit = self.opt['limit'] + min_limit

        nll_loss, model_output = super().compute_loss(batch, True)

        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output  # scores is bsz x time x vocab
        scores = F.log_softmax(scores, dim=-1)

        targets = batch.label_vec
        notnull = targets != self.NULL_IDX
        self.metrics['words'].update(preds[notnull].view(-1).tolist())

        high_freq = ((preds > min_limit) & (preds < limit) & notnull).sum().item()
        self.metrics['high_freq'] += high_freq
        gold_high_freq = (
            ((targets > min_limit) & (targets < limit) & notnull).sum().item()
        )
        self.metrics['gold_high'] += gold_high_freq

        ul_mask = (targets > limit) & notnull
        ul_scores = scores[:, :, 4:limit]
        ul_loss = (
            -torch.log((1 - ul_scores.exp()).clamp_(1e-6))
            * ul_mask.type_as(scores).unsqueeze(-1)
        ).sum()

        # indices = preds[ul_mask]
        # ul_scores = scores[ul_mask][torch.arange(num_ul).to(ul_mask.device), indices]

        # ul_loss = (-torch.log((1 - ul_scores.exp()).clamp_(1e-6))).sum()

        num_ul = ul_mask.sum().item()
        self.metrics['ul_loss'] += ul_loss.item()
        self.metrics['ul_num_tokens'] += num_ul

        if num_ul > 0:
            ul_loss /= num_ul

        loss = nll_loss + self.opt['alpha'] * ul_loss

        if return_output:
            return (loss, model_output)
        else:
            return loss


class SequenceVocabUnlikelihoodAgentTrait(_VocabUnlikelihoodTrait):
    """
    Abstract Trait.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.NUM_STEPS = opt['queue_size']
        if shared is None:
            self._reset_running_histories()
            self._last_was_training = True
            self.truebins = {}
            with open("/private/home/roller/working/parlai/convai2_counts.txt") as f:
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
        grp = super().add_cmdline_args(argparser)
        grp.add_argument('--alpha', default=1.0, type=float)
        grp.add_argument('--queue-size', default=32, type=int)
        grp.add_argument(
            '--weighting', choices={'uniform', 'logdiff', 'kldiv'}, default='uniform'
        )
        grp.add_argument('--threshold', type=float, default=1e-3)

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
        if self.metrics['steps']:
            r['num_penalize'] = self.metrics['num_penalize'] / self.metrics['steps']
            r['hum_toks'] = self.metrics['hum_toks'] / self.metrics['steps']
            r['gen_toks'] = self.metrics['gen_toks'] / self.metrics['steps']
            r['ul_weights'] = self.metrics['ul_weights'] / self.metrics['steps']
        if self.running_generation and self.running_human:
            gen_total = sum(self.running_generation.values())
            hum_total = sum(self.running_human.values())
            hum_pct_top14 = (
                sum(v for _, v in self.running_human.most_common(14)) / hum_total
            )
            gen_pct_top14 = (
                sum(
                    self.running_generation[w]
                    for w, _ in self.running_human.most_common(14)
                )
                / gen_total
            )
            r['hum_pct14'] = round_sigfigs(hum_pct_top14, 4)
            r['gen_pct14'] = round_sigfigs(gen_pct_top14, 4)

            r['humdist'] = self._get_bins(self.running_human)
            r['gendist'] = self._get_bins(self.running_generation)

            r['dist_l2'] = self._l2dist(r['gendist'])

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
        self.metrics['ul_total_toks'] += gen_mask.sum().item()

        self.generation_history.append(Counter(gentoks[gen_mask].view(-1).tolist()))
        self.human_history.append(Counter(targets[notnull].view(-1).tolist()))
        self.metrics['words'].update(self.generation_history[-1])
        self.metrics['human_words'].update(self.human_history[-1])
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

        if self.metrics['steps'] % 50 == 0:
            print(
                "weights:",
                " ".join(
                    "{}:{:.2g}".format(self.dict.ind2tok[i], v)
                    for i, v in to_penalize.items()
                ),
            )
        self.metrics['hum_toks'] += hum_sum
        self.metrics['gen_toks'] += gen_sum
        self.metrics['num_penalize'] += len(to_penalize)
        self.metrics['steps'] += 1

        ul_weights = torch.zeros(gen_mask.shape)
        ul_mask = torch.zeros_like(gen_mask)
        for wordid, weight in to_penalize.items():
            ul_mask = ul_mask | (gentoks == wordid)
            ul_weights[gentoks == wordid] = weight
        ul_weights = ul_weights.to(gen_mask.device)
        self.metrics['ul_weights'] += ul_weights[ul_mask].mean().item()

        # and whack it
        model_output = self.model(*self._model_input(batch), ys=gentoks)
        scores, *_ = model_output
        flat_scores = scores.view(-1, scores.size(2))
        flat_mask = ul_mask.view(-1)
        downweight = gentoks[ul_mask]

        almost_scores = F.log_softmax(scores[ul_mask], dim=-1)
        ul_scores = almost_scores[torch.arange(len(downweight)), downweight]

        ul_loss = (
            (-torch.log((1 - ul_scores.exp()).clamp_(1e-20))) * ul_weights[ul_mask]
        ).sum()

        num_ul = ul_mask.sum().item()
        self.metrics['ul_loss'] += ul_loss.item()
        self.metrics['ul_num_tokens'] += num_ul

        if num_ul > 0:
            ul_loss /= num_ul

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
    pass


class TransformerTokenVocabUnlikelihoodAgent(
    SequenceVocabUnlikelihoodAgentTrait, TransformerGeneratorAgent
):
    pass


class TransformerUnlikelihoodAgent(
    RewardUnlikelihoodAgentTrait, TransformerGeneratorAgent
):
    """
    Example usage:

        -t internal:generation_safety -m parlai_internal.projects.unlikelihood.agents:TransformerUnlikelihoodAgent
    """

    pass


class RepetitionUnlikelihoodParlallAgent(
    RepetitionUnlikelihoodAgentTrait, ParlallAgent
):
    """
    Example usage:

        -t internal:generation_safety -m parlai_internal.projects.unlikelihood.agents:RepetitionUnlikelihoodAgent
    """

    pass


class RepetitionUnlikelihoodAgent(
    RepetitionUnlikelihoodAgentTrait, TransformerGeneratorAgent
):
    """
    Example usage:

        -t internal:generation_safety -m parlai_internal.projects.unlikelihood.agents:RepetitionUnlikelihoodAgent
    """

    pass
