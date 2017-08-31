# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A set of classes that facilitate a dialogue between agents.
"""

import sys
from collections import defaultdict

import numpy as np
import torch
from torch import optim, autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from . import domain


class Agent(object):
    """Agent's interface.

    The dialogue should proceed in the following way:

    1) feed_context to each of the agent.
    2) randomly pick an agent who will start the conversation.
    3) the starting agent will write down her utterance.
    4) the other agent will read the pronounced utterance.
    5) unless the end of dialogue is pronounced, swap the agents and repeat the steps 3-4.
    6) once the conversation is over, generate choices for each agent and calculate the reward.
    7) pass back to the reward to the update function.


    See Dialogue.run in the dialog.py for more details.
    """

    def feed_context(self, context):
        """Feed context in to start new conversation.

        context: a list of context tokens.
        """
        pass

    def read(self, inpt):
        """Read an utterance from your partner.

        inpt: a list of English words describing a sentence.
        """
        pass

    def write(self):
        """Generate your own utterance."""
        pass

    def choose(self):
        """Call it after the conversation is over, to make the selection."""
        pass

    def update(self, agree, reward):
        """After end of each dialogue the reward will be passed back to update the parameters.

        agree: a boolean flag that specifies if the agents agreed on the deal.
        reward: the reward that the agent receives after the dialogue. 0 if there is no agreement.
        """
        pass


class LstmAgent(Agent):
    """An agent that uses DialogModel as an AI."""
    def __init__(self, model, args, name='Alice'):
        super(LstmAgent, self).__init__()
        self.model = model
        self.args = args
        self.name = name
        self.human = False
        self.domain = domain.get_domain(args.domain)

    def _encode(self, inpt, dictionary):
        """A helper function that encodes the passed in words using the dictionary.

        inpt: is a list of strings.
        dictionary: prebuild mapping, see Dictionary in data.py
        """
        encoded = torch.LongTensor(dictionary.w2i(inpt)).unsqueeze(1)
        if self.model.device_id is not None:
            encoded = encoded.cuda(self.model.device_id)
        return encoded

    def _decode(self, out, dictionary):
        """A helper function that decodes indeces into English words.

        out: variable that contains an encoded utterance.
        dictionary: prebuild mapping, see Dictionary in data.py
        """
        return dictionary.i2w(out.data.squeeze(1).cpu())

    def feed_context(self, context):
        # the hidden state of all the pronounced words
        self.lang_hs = []
        # all the pronounced words
        self.words = []
        self.context = context
        # encoded context
        self.ctx = self._encode(context, self.model.context_dict)
        # hidded state of context
        self.ctx_h = self.model.forward_context(Variable(self.ctx))
        # current hidden state of the language rnn
        self.lang_h = self.model.zero_hid(1)

    def read(self, inpt):
        inpt = self._encode(inpt, self.model.word_dict)
        lang_hs, self.lang_h = self.model.read(Variable(inpt), self.lang_h, self.ctx_h)
        # append new hidded states to the current list of the hidden states
        self.lang_hs.append(lang_hs.squeeze(1))
        # first add the special 'THEM:' token
        self.words.append(self.model.word2var('THEM:'))
        # then read the utterance
        self.words.append(Variable(inpt))
        assert (torch.cat(self.words).size()[0] == torch.cat(self.lang_hs).size()[0])

    def write(self):
        # generate a new utterance
        _, outs, self.lang_h, lang_hs = self.model.write(self.lang_h, self.ctx_h,
            100, self.args.temperature)
        # append new hidded states to the current list of the hidden states
        self.lang_hs.append(lang_hs)
        # first add the special 'YOU:' token
        self.words.append(self.model.word2var('YOU:'))
        # then append the utterance
        self.words.append(outs)
        assert (torch.cat(self.words).size()[0] == torch.cat(self.lang_hs).size()[0])
        # decode into English words
        return self._decode(outs, self.model.word_dict)

    def _choose(self, lang_hs=None, words=None, sample=False):
        # get all the possible choices
        choices = self.domain.generate_choices(self.context)
        # concatenate the list of the hidden states into one tensor
        lang_hs = lang_hs if lang_hs is not None else torch.cat(self.lang_hs)
        # concatenate all the words into one tensor
        words = words if words is not None else torch.cat(self.words)
        # logits for each of the item
        logits = self.model.generate_choice_logits(words, lang_hs, self.ctx_h)

        # construct probability distribution over only the valid choices
        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.from_numpy(np.array(idxs)))
            idxs = self.model.to_device(idxs)
            choices_logits.append(torch.gather(logits[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=False)
        # subtract the max to softmax more stable
        choice_logit = choice_logit.sub(choice_logit.max().data[0])
        prob = F.softmax(choice_logit)
        if sample:
            # sample a choice
            idx = prob.multinomial().detach()
            logprob = F.log_softmax(choice_logit).gather(0, idx)
        else:
            # take the most probably choice
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.data[0]]

        # Pick only your choice
        return choices[idx.data[0]][:self.domain.selection_length()], logprob, p_agree.data[0]

    def choose(self):
        choice, _, _ = self._choose()
        return choice


class LstmRolloutAgent(LstmAgent):
    """This agent uses planning by estimating potential scores via rollouts."""
    def __init__(self, model, args, name='Alice'):
        super(LstmRolloutAgent, self).__init__(model, args, name)
        # number of conversations to try out for planning
        self.ncandidate = 10
        # number of rollouts for each conversation to estimate the average reward.
        self.nrollout = 5
        # max len of each rollout
        self.rollout_len = 100

    def write(self):
        best_score = -1
        res = None

        for _ in range(self.ncandidate):
            # generate the beginning of the conversation
            _, move, move_lang_h, move_lang_hs = self.model.write(
                self.lang_h, self.ctx_h, 100, self.args.temperature)

            # if this is not the end of the conversation
            is_selection = len(move) == 1 and \
                self.model.word_dict.get_word(move.data[0][0]) == '<selection>'

            score = 0
            # try nrollout rollouts to estimate the reward
            for _ in range(self.nrollout):
                combined_lang_hs = self.lang_hs + [move_lang_hs]
                combined_words = self.words + [self.model.word2var('YOU:'), move]

                if not is_selection:
                    # complete the conversation with rollout_length samples
                    _, rollout, _, rollout_lang_hs = self.model.write(
                        move_lang_h, self.ctx_h, self.rollout_len, self.args.temperature,
                        stop_tokens=['<selection>'], resume=True)
                    combined_lang_hs += [rollout_lang_hs]
                    combined_words += [rollout]

                # choose items
                rollout_score = None

                combined_lang_hs = torch.cat(combined_lang_hs)
                combined_words = torch.cat(combined_words)
                rollout_choice, _, p_agree = self._choose(combined_lang_hs, combined_words, sample=False)
                rollout_score = self.domain.score(self.context, rollout_choice)
                score += p_agree * rollout_score

            # take the candidate with the max expected reward
            if score > best_score:
                res = (move, move_lang_h, move_lang_hs)
                best_score = score

        # store the best candidate and output the produced utterance
        outs, lang_h, lang_hs = res
        self.lang_h = lang_h
        self.lang_hs.append(lang_hs)
        self.words.append(self.model.word2var('YOU:'))
        self.words.append(outs)
        return self._decode(outs, self.model.word_dict)


class BatchedRolloutAgent(LstmRolloutAgent):
    """Similar to LstmRolloutAgent, but it uses batching to evaluate all the rollouts together."""
    def __init__(self, model, args, name='Alice'):
        super(BatchedRolloutAgent, self).__init__(model, args, name)
        self.eos = self.model.word_dict.get_idx('<eos>')
        self.eod = self.model.word_dict.get_idx('<selection>')

    def _find(self, seq, tokens):
        n = seq.size(0)
        for i in range(n):
            if seq[i] in tokens:
                return i
        return n

    def write(self):
        batch_outs, batch_lang_hs = self.model.write_batch(
            self.args.rollout_bsz, self.lang_h, self.ctx_h, self.args.temperature)

        counts, scores, states = defaultdict(float), defaultdict(int), defaultdict(list)
        for i in range(self.args.rollout_bsz):
            outs = batch_outs.narrow(1, i, 1).squeeze(1).data.cpu()
            lang_hs = batch_lang_hs.narrow(1, i, 1).squeeze(1)

            # find the end of the dialogue
            eod_pos = self._find(outs, [self.eod])
            if eod_pos == outs.size(0):
                # unfinished dialogue, don't count this
                continue

            # find the end of the first utterance
            first_turn_length = self._find(outs, [self.eos, self.eod]) + 1
            move = outs.narrow(0, 0, first_turn_length)
            sent = ' '.join(self.model.word_dict.i2w(move.numpy()))
            sent_lang_hs = lang_hs.narrow(0, 0, first_turn_length + 1)
            lang_h = lang_hs.narrow(0, first_turn_length + 1, 1).unsqueeze(0)

            dialog_lang_hs = lang_hs.narrow(0, 0, eod_pos + 1)
            dialog_words = Variable(self.model.to_device(outs.narrow(0, 0, eod_pos + 1)))
            choice, _, p_agree = self._choose(
                torch.cat(self.lang_hs + [dialog_lang_hs]),
                torch.cat(self.words + [dialog_words]).squeeze().unsqueeze(1), sample=False)

            # group by the first utterance
            counts[sent] += 1
            scores[sent] += self.domain.score(self.context, choice) * p_agree
            states[sent] = (lang_h, sent_lang_hs, move)

        # filter out the candidates that appeared less than 'threshold' times
        for threshold in range(self.args.rollout_count_threshold, -1, -1):
            cands = [k for k in counts if counts[k] >= threshold]
            if cands:
                sent = max(cands, key=lambda k: scores[k] / counts[k])
                lang_h, sent_lang_hs, move = states[sent]
                self.lang_h = lang_h
                self.lang_hs.append(sent_lang_hs)
                self.words.append(self.model.word2var('YOU:'))
                self.words.append(self.model.to_device(Variable(move)))
                assert (torch.cat(self.words).size()[0] == torch.cat(self.lang_hs).size()[0])

                return sent.split(' ')


class RlAgent(LstmAgent):
    """An Agent that updates the model parameters using REINFORCE to maximize the reward."""
    def __init__(self, model, args, name='Alice'):
        super(RlAgent, self).__init__(model, args, name=name)
        self.opt = optim.SGD(
            self.model.parameters(),
            lr=self.args.rl_lr,
            momentum=self.args.momentum,
            nesterov=(self.args.nesterov and self.args.momentum > 0))

        self.all_rewards = []

        if self.args.visual:
            self.model_plot = vis.ModulePlot(self.model, plot_weight=False, plot_grad=True)
            self.reward_plot = vis.Plot(['reward',], 'reward', 'reward')
            self.loss_plot = vis.Plot(['loss',], 'loss', 'loss')
        self.t = 0

    def feed_context(self, ctx):
        super(RlAgent, self).feed_context(ctx)
        # save all the log probs for each generated word,
        # so we can use it later to estimate policy gradient.
        self.logprobs = []

    def write(self):
        logprobs, outs, self.lang_h, lang_hs = self.model.write(self.lang_h, self.ctx_h,
            100, self.args.temperature)
        # append log probs from the generated words
        self.logprobs.extend(logprobs)
        self.lang_hs.append(lang_hs)
        # first add the special 'YOU:' token
        self.words.append(self.model.word2var('YOU:'))
        # then append the utterance
        self.words.append(outs)
        assert (torch.cat(self.words).size()[0] == torch.cat(self.lang_hs).size()[0])
        return self._decode(outs, self.model.word_dict)

    def choose(self):
        if self.args.eps < np.random.rand():
            choice, _, _ = self._choose(sample=False)
        else:
            choice, logprob, _ = self._choose(sample=True)
            # save log prob for the selection as well, if we sample it
            self.logprobs.append(logprob)
        return choice

    def update(self, agree, reward):
        self.t += 1
        reward = reward if agree else 0
        self.all_rewards.append(reward)
        # standardize the reward
        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        # compute accumulated discounted reward
        g = Variable(torch.zeros(1, 1).fill_(r))
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.rl_clip)
        if self.args.visual and self.t % 10 == 0:
            self.model_plot.update(self.t)
            self.reward_plot.update('reward', self.t, reward)
            self.loss_plot.update('loss', self.t, loss.data[0])
        self.opt.step()


class HumanAgent(Agent):
    """An agent that is used by a human to converse with AI."""
    def __init__(self, domain, name='Human'):
        self.name = name
        self.human = True
        self.domain = domain

    def feed_context(self, ctx):
        self.ctx = ctx

    def write(self):
        while True:
            try:
                return input('%s : ' % self.name).lower().strip().split() + ['<eos>']
            except KeyboardInterrupt:
                sys.exit()
            except:
                print('Your sentence is invalid! Try again.')

    def choose(self):
        while True:
            try:
                choice = input('%s choice: ' % self.name)
                return self.domain.parse_human_choice(self.ctx, choice)
            except KeyboardInterrupt:
                sys.exit()
            #except:
            #    print('Your choice is invalid! Try again.')
