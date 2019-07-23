#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from copy import deepcopy


def mask_out(data, mask):
    return data.index_select(0, mask.nonzero().squeeze())


def normalize(data, p=2, dim=1, eps=1e-12):
    return data / torch.norm(data, p, dim).clamp(min=eps).expand_as(data)


class ObjectChecklistModel(nn.Module):
    def __init__(self, opt, data_agent):
        super().__init__()
        self.opt = opt

        self.input_emb = nn.Embedding(
            data_agent.wordcnt, opt['embedding_dim'], padding_idx=0
        )
        self.action_type_emb = nn.Embedding(
            data_agent.get_num_actions(), opt['action_type_emb_dim']
        )
        self.encoder = nn.GRU(
            opt['embedding_dim'],
            opt['rnn_h'],
            opt['rnn_layers'],
            batch_first=True,
            bidirectional=opt['bidir'],
        )
        self.decoder = nn.Sequential(nn.Linear(opt['rnn_h'], 1))
        self.log_softmax = nn.LogSoftmax()
        self.trans = nn.Sequential(
            nn.Linear(opt['rnn_h'] * (2 if opt['bidir'] else 1), opt['embedding_dim']),
            nn.Tanh(),
        )
        counter_emb = opt['counter_emb_dim']
        if opt['counter_ablation']:
            counter_emb = 0
        self.dec_gru = nn.GRU(
            opt['rnn_h'] * (2 if opt['bidir'] else 1)
            + counter_emb
            + (opt['embedding_dim'] if not opt['room_ablation'] else 0)
            + opt['action_type_emb_dim']
            + opt['action_type_emb_dim']
            + opt['embedding_dim']
            + opt['embedding_dim']
            + opt['rnn_h'] * (2 if opt['bidir'] else 1),
            opt['rnn_h'],
            opt['rnn_layers'],
            batch_first=True,
        )
        self.merge = nn.Sequential(nn.Linear(opt['rnn_h'] * 2, opt['rnn_h']), nn.Tanh())
        self.counter_emb = nn.Embedding(opt['counter_max'] + 1, opt['counter_emb_dim'])

    def forward_loss(
        self,
        x,
        action_key,
        second_action_key,
        action_type,
        current_room,
        checked,
        y,
        y_mask,
        counter_feat,
        average_=True,
    ):
        """
        x: [batch, seq_in], int
        action_key: [y_dim], int
        second_action_key: [y_dim], int
        action_type: [y_dim], int
        current_room: [batch, seq_out], int
        checked: [batch, seq_out + 1, y_dim], float, binary
        y: [batch, seq_out, y_dim], float, binary
        y_mask: [batch, seq_out, y_dim], float, binary
        counter_feat: [batch, seq_out, y_dim], int
        """

        opt = self.opt
        batch_size, seq_out, seq_in = x.size(0), y.size(1), x.size(1)
        h_0 = Variable(
            torch.zeros(
                opt['rnn_layers'] * (2 if opt['bidir'] else 1), batch_size, opt['rnn_h']
            )
        )
        if opt['cuda']:
            h_0 = h_0.cuda()

        emb_out = self.input_emb(x)  # [batch, seq_in, dim]
        enc_out, hidden = self.encoder(
            emb_out, h_0
        )  # [batch, seq_in, h], [layer, batch, h]

        action_emb_ori = self.input_emb(action_key.unsqueeze(1)).squeeze(
            1
        )  # [y_dim, dim]
        y_dim, emb_dim = action_emb_ori.size()
        action_emb = (
            action_emb_ori.unsqueeze(0)
            .expand(batch_size, y_dim, emb_dim)
            .transpose(1, 2)
        )  # [batch, dim, y_dim]

        second_action_emb_ori = self.input_emb(second_action_key.unsqueeze(1)).squeeze(
            1
        )  # [y_dim, dim]
        second_action_emb = (
            second_action_emb_ori.unsqueeze(0)
            .expand(batch_size, y_dim, emb_dim)
            .transpose(1, 2)
        )  # [batch, dim, y_dim]

        alpha = F.softmax(
            torch.bmm(emb_out, action_emb).transpose(1, 2).contiguous().view(-1, seq_in)
            + torch.bmm(
                self.trans(enc_out.contiguous().view(batch_size * seq_in, -1))
                .contiguous()
                .view(batch_size, seq_in, -1),
                action_emb,
            )
            .transpose(1, 2)
            .contiguous()
            .view(-1, seq_in)
        )  # [batch * y_dim, seq_in]
        attention = torch.bmm(
            alpha.view(batch_size, y_dim, seq_in), enc_out
        )  # [batch, y_dim, h]

        second_alpha = F.softmax(
            torch.bmm(emb_out, second_action_emb)
            .transpose(1, 2)
            .contiguous()
            .view(-1, seq_in)
            + torch.bmm(
                self.trans(enc_out.view(batch_size * seq_in, -1)).view(
                    batch_size, seq_in, -1
                ),
                second_action_emb,
            )
            .transpose(1, 2)
            .contiguous()
            .view(-1, seq_in)
        )
        second_attention = torch.bmm(
            second_alpha.view(batch_size, y_dim, seq_in), enc_out
        )  # [batch, y_dim, h]

        action_type_out_ori = self.action_type_emb(action_type)  # [y_dim, dim]
        action_type_out = action_type_out_ori.unsqueeze(0).expand(
            batch_size, y_dim, opt['action_type_emb_dim']
        )
        action_type_emb_dim = action_type_out.size(2)

        room_emb = self.input_emb(current_room)  # [batch, seq_out, emb_dim]

        loss = 0
        if not average_:
            loss = None

        hidden = (
            self.merge(hidden.view(batch_size, -1))
            .unsqueeze(1)
            .expand(batch_size, y_dim, opt['rnn_h'])
            .contiguous()
            .view(1, batch_size * y_dim, -1)
        )

        for i in range(seq_out):
            counter_in = self.counter_emb(counter_feat[:, i])  # [batch, y_dim, dim]
            room_in = room_emb[:, i].unsqueeze(1).expand(batch_size, y_dim, emb_dim)

            if i == 0:
                y_in = Variable(torch.zeros(batch_size, y_dim))
                if opt['cuda']:
                    y_in = y_in.cuda()
            else:
                y_in = y[:, i - 1]

            y_second_in = (
                torch.mm(y_in, second_action_emb_ori)
                .unsqueeze(1)
                .expand(batch_size, y_dim, emb_dim)
            )  # [batch, y_dim, dim]
            y_type_in = (
                torch.mm(y_in, action_type_out_ori)
                .unsqueeze(1)
                .expand(batch_size, y_dim, action_type_emb_dim)
            )  # [batch, y_dim, dim]
            y_in = (
                torch.mm(y_in, action_emb_ori)
                .unsqueeze(1)
                .expand(batch_size, y_dim, emb_dim)
            )  # [batch, y_dim, dim]

            dec_in_list = [attention]
            if not opt['counter_ablation']:
                dec_in_list.append(counter_in)
            if not opt['room_ablation']:
                dec_in_list.append(room_in)
            dec_in_list.append(action_type_out)
            dec_in_list.append(y_type_in)
            dec_in_list.append(y_second_in)
            dec_in_list.append(y_in)
            dec_in_list.append(second_attention)
            dec_in = torch.cat(dec_in_list, 2)
            dec_out, hidden = self.dec_gru(
                dec_in.view(batch_size * y_dim, 1, -1), hidden
            )  # [batch * y_dim, 1, h], [1, batch * y_dim, h]

            dec_out = dec_out.squeeze(1)  # [batch * y_dim, h]

            dec_out = self.log_softmax(self.decoder(dec_out).view(batch_size, y_dim))

            if not average_:
                new_loss = -(dec_out * y[:, i]).sum(1)
                if loss is None:
                    loss = new_loss
                else:
                    loss += new_loss
            else:
                loss += -(dec_out * y[:, i]).sum()

        if not average_:
            return loss

        loss /= y.sum()
        return loss

    def forward_predict(
        self,
        x,
        action_key,
        second_action_key,
        action_type,
        check_mapping,
        checked,
        graphs,
        data_agent,
        constrain_=True,
    ):
        """
        check_mapping: [y_dim, y_dim], float, binary
        """
        graphs = deepcopy(graphs)
        opt = self.opt
        batch_size, seq_out, seq_in = x.size(0), opt['max_seq_out'], x.size(1)
        h_0 = Variable(
            torch.zeros(
                opt['rnn_layers'] * (2 if opt['bidir'] else 1), batch_size, opt['rnn_h']
            ),
            volatile=True,
        )
        if opt['cuda']:
            h_0 = h_0.cuda()

        emb_out = self.input_emb(x)
        enc_out, hidden = self.encoder(emb_out, h_0)

        action_emb_ori = self.input_emb(action_key.unsqueeze(1)).squeeze(
            1
        )  # [y_dim, dim]
        y_dim, emb_dim = action_emb_ori.size()
        action_emb = (
            action_emb_ori.unsqueeze(0)
            .expand(batch_size, y_dim, emb_dim)
            .transpose(1, 2)
        )  # [batch, dim, y_dim]

        second_action_emb_ori = self.input_emb(second_action_key.unsqueeze(1)).squeeze(
            1
        )  # [y_dim, dim]
        second_action_emb = (
            second_action_emb_ori.unsqueeze(0)
            .expand(batch_size, y_dim, emb_dim)
            .transpose(1, 2)
        )  # [batch, dim, y_dim]

        alpha = F.softmax(
            torch.bmm(emb_out, action_emb).transpose(1, 2).contiguous().view(-1, seq_in)
            + torch.bmm(
                self.trans(enc_out.contiguous().view(batch_size * seq_in, -1))
                .contiguous()
                .view(batch_size, seq_in, -1),
                action_emb,
            )
            .transpose(1, 2)
            .contiguous()
            .view(-1, seq_in)
        )
        attention = torch.bmm(
            alpha.view(batch_size, y_dim, seq_in), enc_out
        )  # [batch, y_dim, h]

        second_alpha = F.softmax(
            torch.bmm(emb_out, second_action_emb)
            .transpose(1, 2)
            .contiguous()
            .view(-1, seq_in)
            + torch.bmm(
                self.trans(enc_out.view(batch_size * seq_in, -1)).view(
                    batch_size, seq_in, -1
                ),
                second_action_emb,
            )
            .transpose(1, 2)
            .contiguous()
            .view(-1, seq_in)
        )
        second_attention = torch.bmm(
            second_alpha.view(batch_size, y_dim, seq_in), enc_out
        )  # [batch, y_dim, h]

        action_type_out_ori = self.action_type_emb(action_type.unsqueeze(1)).squeeze(
            1
        )  # [y_dim, dim]
        action_type_out = action_type_out_ori.unsqueeze(0).expand(
            batch_size, y_dim, opt['action_type_emb_dim']
        )
        action_type_emb_dim = action_type_out.size(2)

        counter_feat = Variable(torch.zeros(batch_size, y_dim).long())
        if opt['cuda']:
            counter_feat = counter_feat.cuda()

        text_out = [[] for _ in range(batch_size)]

        hidden = (
            self.merge(hidden.view(batch_size, -1))
            .unsqueeze(1)
            .expand(batch_size, y_dim, opt['rnn_h'])
            .contiguous()
            .view(1, batch_size * y_dim, -1)
        )

        y_onehot = None
        for i in range(seq_out):
            room_in = torch.zeros(batch_size).long()
            for j in range(batch_size):
                room_in[j] = data_agent.get_room(graphs[j])
            if opt['cuda']:
                room_in = room_in.cuda()
            room_in = Variable(room_in, volatile=True)
            room_in = self.input_emb(room_in.unsqueeze(1)).expand(
                batch_size, y_dim, emb_dim
            )

            if i == 0:
                y_in = Variable(torch.zeros(batch_size, y_dim))
                if opt['cuda']:
                    y_in = y_in.cuda()
            else:
                y_in = y_onehot

            y_second_in = (
                torch.mm(y_in, second_action_emb_ori)
                .unsqueeze(1)
                .expand(batch_size, y_dim, emb_dim)
            )  # [batch, y_dim, dim]
            y_type_in = (
                torch.mm(y_in, action_type_out_ori)
                .unsqueeze(1)
                .expand(batch_size, y_dim, action_type_emb_dim)
            )  # [batch, y_dim, dim]
            y_in = (
                torch.mm(y_in, action_emb_ori)
                .unsqueeze(1)
                .expand(batch_size, y_dim, emb_dim)
            )  # [batch, y_dim, dim]

            counter_in = self.counter_emb(counter_feat)  # [batch, y_dim, dim]

            dec_in_list = [attention]
            if not opt['counter_ablation']:
                dec_in_list.append(counter_in)
            if not opt['room_ablation']:
                dec_in_list.append(room_in)
            dec_in_list.append(action_type_out)
            dec_in_list.append(y_type_in)
            dec_in_list.append(y_second_in)
            dec_in_list.append(y_in)
            dec_in_list.append(second_attention)
            dec_in = torch.cat(dec_in_list, 2)

            dec_out, hidden = self.dec_gru(
                dec_in.view(batch_size * y_dim, 1, -1), hidden
            )  # [batch * y_dim, 1, h], [1, batch * y_dim, h]

            y_mask = torch.zeros(batch_size, y_dim)
            for j in range(batch_size):
                data_agent.get_mask(graphs[j], y_mask[j])
            if opt['cuda']:
                y_mask = y_mask.cuda()
            y_mask = Variable(y_mask, volatile=True)

            dec_out = dec_out.squeeze(1)  # [batch * y_dim, h]

            dec_out = self.decoder(dec_out).view(batch_size, y_dim)

            if constrain_:
                dec_out = dec_out * y_mask + -1e7 * (1 - y_mask)
            y_out = torch.max(dec_out, 1, keepdim=True)[1].data
            y_onehot = torch.zeros(batch_size, y_dim)
            y_onehot.scatter_(1, y_out.cpu(), 1)
            if opt['cuda']:
                y_onehot = y_onehot.cuda()
            y_onehot = Variable(y_onehot, volatile=True)  # [batch, y_dim]

            y_out = y_out.squeeze()
            for j in range(batch_size):
                if len(text_out[j]) > 0 and text_out[j][-1] == 'STOP':
                    continue
                cur_tuple = data_agent.get_action_tuple(y_out[j])
                text_out[j].append(data_agent.reverse_parse_action(cur_tuple))
                if text_out[j][-1] != 'STOP':
                    exec_result = graphs[j].parse_exec(text_out[j][-1])
                    if constrain_:
                        assert exec_result, text_out[j][-1]
                    for action_name in data_agent.key_to_check[
                        data_agent.check_to_key[cur_tuple]
                    ]:
                        action_id = data_agent.get_action_id(action_name)
                        counter_feat[j, action_id] = counter_feat[j, action_id] + 1
                    counter_feat.data.clamp_(max=opt['counter_max'])

        return text_out


class Seq2SeqModel(nn.Module):
    def __init__(self, opt, data_agent):
        super().__init__()
        self.opt = opt

        self.y_dim = data_agent.y_dim

        self.input_emb = nn.Embedding(
            data_agent.wordcnt, opt['embedding_dim'], padding_idx=0
        )
        self.encoder = nn.GRU(
            opt['embedding_dim'], opt['rnn_h'], opt['rnn_layers'], batch_first=True
        )
        self.decoder = nn.GRU(
            self.y_dim, opt['rnn_h'], opt['rnn_layers'], batch_first=True
        )
        self.mapping = nn.Sequential(
            nn.Linear(opt['rnn_h'] * 2, self.y_dim), nn.LogSoftmax()
        )

    def forward_loss(self, x, y, average_=True):
        """
        x: [batch, seq_in], int
        y: [batch, seq_out, 3 * target], float, binary
        """

        opt = self.opt
        batch_size, seq_out = x.size(0), y.size(1)
        h_0 = Variable(torch.zeros(opt['rnn_layers'], batch_size, opt['rnn_h']))
        if opt['cuda']:
            h_0 = h_0.cuda()

        enc_out, hidden = self.encoder(
            self.input_emb(x), h_0
        )  # [batch, seq_in, h], [layer, batch, h]
        loss = 0 if average_ else None
        for i in range(seq_out):
            if i == 0:
                y_in = Variable(torch.zeros(batch_size, 1, y.size(2)))
                if opt['cuda']:
                    y_in = y_in.cuda()
            else:
                y_in = y[:, i - 1].unsqueeze(1)
            dec_out, hidden = self.decoder(
                y_in, hidden
            )  # [batch, 1, h], [layer, batch, h]
            alpha = F.softmax(
                torch.bmm(enc_out, hidden[-1].unsqueeze(2))
            )  # [batch, seq_in, 1]
            attention = torch.bmm(enc_out.transpose(1, 2), alpha).squeeze(
                2
            )  # [batch, h]
            dec_out = self.mapping(
                torch.cat([attention, dec_out.squeeze(1)], dim=1)
            )  # [batch, y_dim]
            if average_:
                loss += -(dec_out * y[:, i]).sum()
            else:
                new_loss = -(dec_out * y[:, i]).sum(1)
                if loss is None:
                    loss = new_loss
                else:
                    loss += new_loss

        if not average_:
            return loss
        loss /= y.sum()

        return loss

    def forward_predict(self, x, graphs, data_agent, constrain_=True):
        graphs = deepcopy(graphs)
        opt = self.opt
        batch_size = x.size(0)
        h_0 = Variable(torch.zeros(opt['rnn_layers'], batch_size, opt['rnn_h']))
        if opt['cuda']:
            h_0 = h_0.cuda()

        enc_out, hidden = self.encoder(
            self.input_emb(x), h_0
        )  # [batch, seq_in, h], [layer, batch, h]
        text_out = [[] for _ in range(batch_size)]
        y_onehot = None
        for i in range(opt['max_seq_out']):
            if i == 0:
                y_in = Variable(torch.zeros(batch_size, 1, self.y_dim))
                if opt['cuda']:
                    y_in = y_in.cuda()
            else:
                y_in = y_onehot.unsqueeze(1)

            dec_out, hidden = self.decoder(y_in, hidden)
            alpha = F.softmax(torch.bmm(enc_out, hidden[-1].unsqueeze(2)))
            attention = torch.bmm(enc_out.transpose(1, 2), alpha).squeeze(2)
            dec_out = self.mapping(
                torch.cat([attention, dec_out.squeeze(1)], dim=1)
            )  # [batch, y_dim]

            y_mask = torch.zeros(batch_size, self.y_dim)
            for j in range(batch_size):
                data_agent.get_mask(graphs[j], y_mask[j])
            if opt['cuda']:
                y_mask = y_mask.cuda()
            y_mask = Variable(y_mask, volatile=True)
            if constrain_:
                dec_out = dec_out * y_mask + -1e7 * (1 - y_mask)

            y_out = torch.max(dec_out, 1, keepdim=True)[1].data  # [batch, 1]
            y_onehot = torch.zeros(batch_size, self.y_dim)  # [batch, y_dim]
            y_onehot.scatter_(1, y_out.cpu(), 1)
            y_onehot = Variable(y_onehot)
            if opt['cuda']:
                y_onehot = y_onehot.cuda()

            y_out = y_out.squeeze()
            for j in range(batch_size):
                if len(text_out[j]) > 0 and text_out[j][-1] == 'STOP':
                    continue
                text_out[j].append(
                    data_agent.reverse_parse_action(
                        data_agent.get_action_tuple(y_out[j])
                    )
                )
                if text_out[j][-1] != 'STOP':
                    exec_result = graphs[j].parse_exec(text_out[j][-1])
                    if constrain_:
                        assert exec_result, text_out[j][-1]
        return text_out
