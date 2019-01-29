# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from parlai.core.torch_generator_agent import TorchGeneratorModel


def _normalize(tensor, norm_layer):
    """
    Broadcast layer norm
    """
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


def _create_embeddings(dictionary, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


def _build_encoder(opt, dictionary, embedding=None, padding_idx=None, reduction=True):
    return TransformerEncoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
        embeddings_scale=opt['embeddings_scale'],
        reduction=reduction,
    )


def _build_decoder(opt, dictionary, embedding=None, padding_idx=None):
    return TransformerDecoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
        embeddings_scale=opt['embeddings_scale'],
    )


class TransformerMemNetModel(nn.Module):
    """Model which takes context, memories, candidates and encodes them"""
    def __init__(self, opt, dictionary):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]
        self.scores_norm = opt['scores_norm']

        # set up embeddings
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.context_encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx
        )

        if opt.get('share_encoders'):
            self.cand_encoder = TransformerResponseWrapper(
                self.context_encoder, self.context_encoder.out_dim,
            )
        else:
            self.cand_encoder = _build_encoder(
                opt, dictionary, self.embeddings, self.pad_idx, reduction=True,
            )

        # build memory encoder
        if opt.get('wrap_memory_encoder', False):
            self.memory_transformer = TransformerResponseWrapper(
                self.context_encoder, self.context_encoder.out_dim
            )
        else:
            self.memory_transformer = self.context_encoder

        self.attender = BasicAttention(dim=2, attn=opt['memory_attention'])

    def encode_cand(self, words):
        if words is None:
            return None

        # flatten if there are many candidates
        if words.dim() == 3:
            oldshape = words.shape
            words = words.reshape(oldshape[0] * oldshape[1], oldshape[2])
        else:
            oldshape = None

        encoded = self.cand_encoder(words)

        if oldshape is not None:
            encoded = encoded.reshape(oldshape[0], oldshape[1], -1)

        return encoded

    def encode_context_memory(self, context_w, memories_w):
        # [batch, d]
        context_h = self.context_encoder(context_w)

        if memories_w is None:
            return [], context_h

        bsz = memories_w.size(0)
        memories_w = memories_w.view(-1, memories_w.size(-1))
        memories_h = self.memory_transformer(memories_w)
        memories_h = memories_h.view(bsz, -1, memories_h.size(-1))

        context_h = context_h.unsqueeze(1)
        context_h, weights = self.attender(context_h, memories_h)

        return weights, context_h

    def _score(self, output, cands):
        if cands.dim() == 2:
            return torch.matmul(output, cands.t())
        elif cands.dim() == 3:
            return torch.bmm(output.unsqueeze(1),
                             cands.transpose(1, 2)).squeeze(1)
        else:
            raise RuntimeError('Unexpected candidate dimensions {}'
                               ''.format(cands.dim()))

    def forward(self, xs, mems, cands):
        weights, context_h = self.encode_context_memory(xs, mems)
        cands_h = self.encode_cand(cands)

        if self.opt['normalize_sent_emb']:
            context_h = context_h / context_h.norm(2, dim=1, keepdim=True)
            cands_h = cands_h / cands_h.norm(2, dim=1, keepdim=True)

        scores = self._score(context_h, cands_h)
        if self.scores_norm == 'dot':
            pass
        elif self.scores_norm == 'sqrt':
            scores /= math.sqrt(self.opt['embedding_size'])
        elif self.scores_norm == 'dim':
            scores /= self.opt['embedding_size']
        else:
            raise ValueError('Invalid --scores-norm')

        return scores


def create_position_codes(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])

    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class TransformerResponseWrapper(nn.Module):
    """Transformer response rapper. Pushes input through transformer and MLP"""
    def __init__(self, transformer, hdim):
        super(TransformerResponseWrapper, self).__init__()
        dim = transformer.out_dim
        self.transformer = transformer
        self.mlp = nn.Sequential(
            nn.Linear(dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, dim)
        )

    def forward(self, *args):
        return self.mlp(self.transformer(*args))


class TransformerEncoder(nn.Module):
    """Transformer model"""
    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction=True,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.padding_idx = padding_idx

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'
        n_positions = 1024  # TODO: use truncate or sth

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            assert False
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(
                n_heads, embedding_size, ffn_size, attention_dropout, relu_dropout
            ))

    def forward(self, input):
        """
            input data is a FloatTensor of shape [batch, seq_len, dim]
            mask is a ByteTensor of shape [batch, seq_len], filled with 1 when
            inside the sequence and 0 outside.
        """
        mask = input != self.padding_idx
        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        tensor *= mask.unsqueeze(-1).float()
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)

        if self.reduction:
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1e-20)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return output, mask


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, dropout=relu_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, tensor, mask):
        tensor = tensor + self.attention(tensor, mask=mask)
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.ffn(tensor)
        tensor = _normalize(tensor, self.norm2)
        tensor *= mask.unsqueeze(-1).float()
        return tensor


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'
        n_positions = 1024  # TODO: use truncate or sth

        self.embeddings = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayer(
                n_heads, embedding_size, ffn_size, attention_dropout, relu_dropout
            ))

    def forward(self, input, encoder_state, incr_state=None):
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask)

        return tensor, None


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask):
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        # x = dropout(x)
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        # x = dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class TransformerGeneratorModel(TorchGeneratorModel):
    def __init__(self, opt, dictionary):
        super().__init__()
        self.pad_idx = dictionary[dictionary.null_token]
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False
        )
        self.decoder = _build_decoder(opt, dictionary, self.embeddings, self.pad_idx)

    def reorder_encoder_states(self, encoder_states, indices):
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        # no support for incremental decoding at this time
        return None

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        return output


class BasicAttention(nn.Module):
    def __init__(self, dim=1, attn='cosine'):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        if attn == 'cosine':
            self.cosine = nn.CosineSimilarity(dim=dim)
        self.attn = attn
        self.dim = dim

    def forward(self, xs, ys):
        if self.attn == 'cosine':
            l1 = self.cosine(xs, ys).unsqueeze(self.dim - 1)
        else:
            l1 = torch.bmm(xs, ys.transpose(1, 2))
            if self.attn == 'sqrt':
                d_k = ys.size(-1)
                l1 = l1 / math.sqrt(d_k)
        l2 = self.softmax(l1)
        lhs_emb = torch.bmm(l2, ys)
        # add back the query
        lhs_emb = lhs_emb.add(xs)

        return lhs_emb.squeeze(self.dim - 1), l2


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        # multi head is seen as one layer, dropout is only applied to the input
        self.dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        # Input is [B, query_len, dim]
        # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(
                batch_size * n_heads,
                seq_len,
                dim_per_head
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, key_len)
            .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, -float(1e20))

        attn_weights = F.softmax(dot_prod / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2).contiguous()
            .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out


class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, dropout=0):
        super(TransformerFFN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x
