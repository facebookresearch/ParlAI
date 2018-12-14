# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from parlai.tasks.talkthewalk.ttw.models.beam_search import SequenceGenerator
from parlai.tasks.talkthewalk.ttw.models.modules import GRUEncoder, CBoW, ControlStep, MASC, NoMASC
from parlai.tasks.talkthewalk.ttw.utils import get_collate_fn

def lengths_to_mask(lengths):
    t = torch.FloatTensor(lengths)
    return (torch.arange(max(t)).float().expand(t.size(0), max(t)) <=
            t.unsqueeze(1)).float()

class TouristLanguage(nn.Module):

    def __init__(self, act_emb_sz, act_hid_sz, num_actions, obs_emb_sz, obs_hid_sz, num_observations,
                 decoder_emb_sz, decoder_hid_sz, num_words, start_token=1, end_token=2):
        super(TouristLanguage, self).__init__()
        self.act_emb_sz = act_emb_sz
        self.act_hid_sz = act_hid_sz
        self.num_actions = num_actions

        self.obs_emb_sz = obs_emb_sz
        self.obs_hid_sz = obs_hid_sz
        self.num_observations = num_observations

        self.decoder_emb_sz = decoder_emb_sz
        self.decoder_hid_sz = decoder_hid_sz
        self.num_words = num_words

        self.act_encoder = GRUEncoder(act_emb_sz, act_hid_sz, num_actions)
        self.obs_encoder = GRUEncoder(obs_emb_sz, obs_hid_sz, num_observations, cbow=True)

        self.emb_fn = nn.Embedding(num_words, decoder_emb_sz)
        self.emb_fn.weight.data.normal_(0.0, 0.1)
        self.decoder = nn.GRU(2*decoder_emb_sz, decoder_hid_sz, batch_first=True)

        self.context_linear = nn.Linear(act_hid_sz+obs_hid_sz, decoder_emb_sz)
        self.out_linear = nn.Linear(decoder_hid_sz, num_words)

        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.start_token = start_token
        self.end_token = end_token


    def encode(self, observations, obs_seq_len, actions, act_seq_len):
        observation_emb = self.obs_encoder(observations, obs_seq_len)
        if act_seq_len.sum() > 0:
            action_emb = self.act_encoder(actions, act_seq_len)
        else:
            action_emb = Variable(torch.FloatTensor(act_seq_len.size(0), self.act_hid_sz).fill_(0.0))

        context_emb = torch.cat([observation_emb, action_emb], 1)
        context_emb = self.context_linear.forward(context_emb)

        return context_emb

    def forward(self, batch, decoding_strategy='greedy',
                max_sample_length=20, beam_width=4, train=True):

        batch = batch._asdict()
        print('batch', batch);
        batch_size = batch['text_vec'].size(0)
        obs_seq_len = batch['see_mask'][:, :, 0].sum(1).long()
        move_mask = lengths_to_mask(batch['move_lengths'])

        if move_mask.dim() > 1:
            act_seq_len = move_mask.sum(1).long()
        else:
            act_seq_len = Variable(torch.LongTensor(batch_size).fill_(0))

        act_seq_len = Variable(torch.LongTensor(batch_size).fill_(0))
        context_emb = self.encode(batch['see_vec'], obs_seq_len, batch['move_vec'], act_seq_len)

        if train:
            # teacher forcing
            assert('text_lengths' in batch.keys() and 'text_vec' in batch.keys())
            inp = batch['text_vec']
            tgt = batch['label_vec']

            inp_emb = self.emb_fn.forward(inp)

            # concatenate external emb
            context_emb = context_emb.view(batch_size, 1, self.decoder_emb_sz).repeat(1, inp_emb.size(1), 1)
            inp_emb = torch.cat([inp_emb, context_emb], 2)

            hs, _ = self.decoder(inp_emb)

            score = self.out_linear(hs)

            loss = 0.0

            #TODO: should this be label mask?
            mask = lengths_to_mask(batch['text_lengths'])[:, 1:]

            for j in range(tgt.size(1)):
                flat_mask = mask[:, j]
                flat_score = score[:, j, :]
                flat_tgt = tgt[:, j]
                nll = self.loss(flat_score, flat_tgt)
                loss += (flat_mask*nll).sum()

            out = {}
            out['loss'] = loss
        else:
            if decoding_strategy in ['greedy', 'sample']:
                preds = []
                probs = []

                input_ind = torch.LongTensor([self.start_token] * batch_size)
                hs = Variable(torch.FloatTensor(1, batch_size, self.decoder_hid_sz).fill_(0.0))
                mask = Variable(torch.FloatTensor(batch_size, max_sample_length).zero_())
                eos = torch.ByteTensor([0]*batch_size)
                if batch['see_vec'].is_cuda:
                    hs = hs.cuda()
                    eos = eos.cuda()
                    mask = mask.cuda()
                    input_ind = input_ind.cuda()

                for k in range(max_sample_length):
                    inp_emb = self.emb_fn.forward(input_ind.unsqueeze(-1))

                    context_emb = context_emb.view(batch_size, 1, self.decoder_emb_sz).repeat(1, inp_emb.size(1), 1)
                    inp_emb = torch.cat([inp_emb, context_emb], 2)

                    _, hs = self.decoder(inp_emb, hs)

                    prob = F.softmax(self.out_linear(hs.squeeze(0)), dim=-1)
                    if decoding_strategy == 'greedy':
                        _, samples = prob.max(1)
                        samples = samples.unsqueeze(-1)
                    else:
                        samples = prob.multinomial(1)
                    mask[:, k] = 1.0 - eos.float()

                    eos = eos | (samples == self.end_token).squeeze()

                    preds.append(samples)
                    probs.append(prob.unsqueeze(1))
                    input_ind = samples.squeeze(-1)

                out = {}
                out['text_vec'] = torch.cat(preds, 1)
                out['text_lengths'] = mask
                out['probs'] = torch.cat(probs, 1)
            elif decoding_strategy == 'beam_search':
                def _step_fn(input, hidden, context, k=4):
                    input = Variable(torch.LongTensor(input)).squeeze()
                    hidden = Variable(torch.FloatTensor(hidden)).unsqueeze(0)
                    context = Variable(torch.FloatTensor(context)).unsqueeze(1)
                    if batch['see_vec'].is_cuda:
                      input = input.cuda()
                      hidden = hidden.cuda()
                      context = context.cuda()


                    prob, hs = self.step(input, hidden, context)

                    logprobs = torch.log(prob)
                    logprobs, words = logprobs.topk(k, 1)
                    hs = hs.squeeze().cpu().data.numpy()

                    return words, logprobs, hs

                seq_gen = SequenceGenerator(_step_fn, self.end_token, max_sequence_length=max_sample_length,
                                            beam_size=beam_width, length_normalization_factor=0.5)
                start_tokens = [[self.start_token] for _ in range(batch_size)]
                hidden = [[0.0]*self.decoder_hid_sz]*batch_size
                beam_out = seq_gen.beam_search(start_tokens, hidden, context_emb.cpu().data.numpy())
                pred_tensor = torch.LongTensor(batch_size, max_sample_length).zero_()
                mask_tensor = torch.FloatTensor(batch_size, max_sample_length).zero_()

                for i, seq in enumerate(beam_out):
                    pred_tensor[i, :(len(seq.output)-1)] = torch.LongTensor(seq.output[1:])
                    mask_tensor[i, :(len(seq.output)-1)] = 1.0

                out = {}
                out['text'] = Variable(pred_tensor)
                out['text_vec'] = Variable(mask_tensor)

                if batch['see_vec'].is_cuda:
                    out['text_vec'] = out['text_vec'].cuda()
                    out['text_lengths'] = out['text_lengths'].cuda()

        return out


    def step(self, input_ind, hs, context_emb):
        inp_emb = self.emb_fn.forward(input_ind.unsqueeze(-1))
        inp_emb = torch.cat([inp_emb, context_emb], 2)

        _, hs = self.decoder(inp_emb, hs)

        prob = F.softmax(self.out_linear(hs.squeeze(0)), dim=-1)
        return prob, hs


    def save(self, path):
        state = dict()
        state['act_emb_sz'] = self.act_emb_sz
        state['act_hid_sz'] = self.act_hid_sz
        state['num_actions'] = self.num_actions
        state['obs_emb_sz'] = self.obs_emb_sz
        state['obs_hid_sz'] = self.obs_hid_sz
        state['num_observations'] = self.num_observations
        state['decoder_emb_sz'] = self.decoder_emb_sz
        state['decoder_hid_sz'] = self.decoder_hid_sz
        state['num_words'] = self.num_words
        state['start_token'] = self.start_token
        state['end_token'] = self.end_token
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)

        tourist = cls(state['act_emb_sz'], state['act_hid_sz'], state['num_actions'],
                      state['obs_emb_sz'], state['obs_hid_sz'], state['num_observations'],
                      state['decoder_emb_sz'], state['decoder_hid_sz'], state['num_words'],
                      start_token=state['start_token'], end_token=state['end_token'])
        tourist.load_state_dict(state['parameters'])
        return tourist

    def show_samples(self, dataset, num_samples=10, cuda=True, logger=None, decoding_strategy='sample',
                     indices=None, beam_width=4):
        if indices is None:
            indices = list()
            for _ in range(num_samples):
                indices.append(random.randint(0, len(dataset) - 1))

        collate_fn = get_collate_fn(cuda)

        data = [dataset[ind] for ind in indices]
        batch = collate_fn(data)

        out = self.forward(batch, decoding_strategy=decoding_strategy, train=False, beam_width=beam_width)

        generated_utterance = out['text_vec'].cpu().data
        logger_fn = print
        if logger:
            logger_fn = logger

        for i in range(len(indices)):
            o = ''
            for obs in data[i]['see_vec']:
                o += '(' + ','.join([dataset.map.landmark_dict.decode(o_ind) for o_ind in obs]) + ') ,'
            # a = ', '.join([i2act[a_ind] for a_ind in actions[i]])
            a = ','.join([dataset.act_dict.decode(a_ind) for a_ind in data[i]['actions']])

            logger_fn('Observations: ' + o)
            logger_fn('Actions: ' + a)
            logger_fn('GT: ' + dataset.dict.decode(batch['text_vec'][i, 1:]))
            logger_fn('Sample: ' + dataset.dict.decode(generated_utterance[i, :]))
            logger_fn('-' * 80)


class GuideLanguage(nn.Module):

    def __init__(self, inp_emb_sz, hidden_sz, num_tokens, apply_masc=True, T=1):
        super(GuideLanguage, self).__init__()
        self.hidden_sz = hidden_sz
        self.inp_emb_sz = inp_emb_sz
        self.num_tokens = num_tokens
        self.apply_masc = apply_masc
        self.T = T

        self.embed_fn = nn.Embedding(num_tokens, inp_emb_sz, padding_idx=0)
        self.encoder_fn = nn.LSTM(inp_emb_sz, hidden_sz//2, batch_first=True, bidirectional=True)
        self.cbow_fn = CBoW(11, hidden_sz)

        self.T_prediction_fn = nn.Linear(hidden_sz, T+1)

        self.feat_control_emb = nn.Parameter(torch.FloatTensor(hidden_sz).normal_(0.0, 0.1))
        self.feat_control_step_fn = ControlStep(hidden_sz)

        if apply_masc:
            self.act_control_emb = nn.Parameter(torch.FloatTensor(hidden_sz).normal_(0.0, 0.1))
            self.act_control_step_fn = ControlStep(hidden_sz)
            self.action_linear_fn = nn.Linear(hidden_sz, 9)

        self.landmark_write_gate = nn.ParameterList()
        self.obs_write_gate = nn.ParameterList()
        for _ in range(T + 1):
            self.landmark_write_gate.append(nn.Parameter(torch.FloatTensor(1, hidden_sz, 1, 1).normal_(0, 0.1)))
            self.obs_write_gate.append(nn.Parameter(torch.FloatTensor(1, hidden_sz).normal_(0.0, 0.1)))

        if apply_masc:
            self.masc_fn = MASC(self.hidden_sz)
        else:
            self.masc_fn = NoMASC(self.hidden_sz)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch, add_rl_loss=False):
        batch = batch._asdict()
        batch_size = batch['text_vec'].size(0)
        input_emb = self.embed_fn(batch['text_vec'])
        hidden_states, _ = self.encoder_fn(input_emb)

        text_mask = lengths_to_mask(batch['text_lengths'])
        last_state_indices = text_mask.sum(1).long() - 1

        last_hidden_states = hidden_states[torch.arange(batch_size).long(), last_state_indices, :]
        T_dist = F.softmax(self.T_prediction_fn(last_hidden_states))
        sampled_Ts = T_dist.multinomial(1).squeeze(-1)

        obs_msgs = list()
        feat_controller = self.feat_control_emb.unsqueeze(0).repeat(batch_size, 1)
        for step in range(self.T + 1):
            extracted_msg, feat_controller = self.feat_control_step_fn(
                    hidden_states, text_mask, feat_controller)
            obs_msgs.append(extracted_msg)

        tourist_obs_msg = []
        for i, (gate, emb) in enumerate(zip(self.obs_write_gate, obs_msgs)):
            include = (i <= sampled_Ts).float().unsqueeze(-1)
            tourist_obs_msg.append(include*F.sigmoid(gate)*emb)
        tourist_obs_msg = sum(tourist_obs_msg)


        landmark_emb = self.cbow_fn(batch['landmark_vec']).permute(0, 3, 1, 2)
        landmark_embs = [landmark_emb]

        if self.apply_masc:
            act_controller = self.act_control_emb.unsqueeze(0).repeat(batch_size, 1)
            for step in range(self.T):
                extracted_msg, act_controller = self.act_control_step_fn(
                        hidden_states, text_mask, act_controller)
                action_out = self.action_linear_fn(extracted_msg)
                out = self.masc_fn.forward(landmark_embs[-1], action_out, current_step=step, Ts=sampled_Ts)
                landmark_embs.append(out)
        else:
            for step in range(self.T):
                landmark_embs.append(self.masc_fn.forward(landmark_embs[-1]))

        landmarks = sum([F.sigmoid(gate)*emb for gate, emb in zip(self.landmark_write_gate, landmark_embs)])

        landmarks = landmarks.resize(batch_size, landmarks.size(1), 16).transpose(1, 2)

        out = dict()
        logits = torch.bmm(landmarks, tourist_obs_msg.unsqueeze(-1)).squeeze(-1)
        out['prob'] = F.softmax(logits, dim=1)
        y_true = (batch['target_location'][:, 0] * 4 + batch['target_location'][:, 1])

        out['sl_loss'] = -torch.log(torch.gather(out['prob'], 1, y_true.unsqueeze(-1)) + 1e-8)

        # add RL loss
        if add_rl_loss:
            advantage = -(out['sl_loss'] - out['sl_loss'].mean()).detach()

            log_prob = torch.log(torch.gather(T_dist, 1, sampled_Ts.unsqueeze(-1)) + 1e-8)
            out['rl_loss'] = log_prob*advantage

        out['acc'] = sum([1.0 for pred, target in zip(out['prob'].max(1)[1].data.cpu().numpy(), y_true.data.cpu().numpy()) if
                   pred == target]) / batch_size
        return out

    def save(self, path):
        state = dict()
        state['hidden_sz'] = self.hidden_sz
        state['embed_sz'] = self.inp_emb_sz
        state['num_tokens'] = self.num_tokens
        state['apply_masc'] = self.apply_masc
        state['T'] = self.T
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        guide = cls(state['embed_sz'], state['hidden_sz'], state['num_tokens'],
                    T=state['T'], apply_masc=state['apply_masc'])
        guide.load_state_dict(state['parameters'])
        return guide
