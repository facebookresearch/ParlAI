import math

import torch
import torch.nn as nn

from parlai.agents.seq2seq.modules import RNNEncoder
from parlai.agents.transformer.modules import (
    TransformerEncoder,
    TransformerFFN,
    MultiHeadAttention,
)

class MetadialogModel(nn.Module):
    @classmethod
    def add_cmdline_args(cls, argparser):
        model = argparser.add_argument_group('MetadialogModel')

        model.add_argument('-shl', '--sen-head-layers', type=int, default=1,
            help="The number of linear layers to have in the sentiment task head")

        model.add_argument('-sexpemb', '--share-exp-embeddings', type='bool',
            default=True,
            help="If True, the explanation task will share the dialog embeddings")
        model.add_argument('-sexpxenc', '--share-exp-x-encoder', type='bool',
            default=True,
            help="If True, the explanation task will share the dialog x encoder")
        model.add_argument('-sexpyenc', '--share-exp-y-encoder', type='bool',
            default=True,
            help="If True, the explanation task will share the dialog y encoder")

        model.add_argument('-ssenemb', '--share-sen-embeddings', type='bool',
            default=False,
            help="If True, the sentiment task will share the dialog embeddings")
        model.add_argument('-ssenenc', '--share-sen-encoder', type='bool',
            default=False,
            help="If True, the sentiment task will share the dialog encoder")

    def __init__(self, opt, dictionary):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]
        self.vocab_size = len(dictionary)

        # Build dialog
        self.dia_embeddings = self.init_embeddings()
        self.x_dia_encoder = self.build_encoder(opt, self.dia_embeddings)
        # print("WARNING: Using linear heads for x and y!")
        # self.x_dia_head = self.build_head(opt, outdim=opt['embedding_size'], num_layers=1)
        self.x_dia_head = nn.Dropout(p=0)

        self.y_dia_encoder = self.build_encoder(opt, self.dia_embeddings)
        # print("WARNING: Using linear heads for x and y!")
        # self.y_dia_head = self.build_head(opt, outdim=opt['embedding_size'], num_layers=1)
        self.y_dia_head = nn.Dropout(p=0)
        # self.y_dia_head = self.x_dia_head

        # Only build the parts of the network you will be using
        # This saves space (nbd) and prevents conflicts when loading
        # Build explanation
        if 'explanation' in self.opt['subtasks']:
            if self.opt['share_exp_embeddings']:
                self.exp_embeddings = self.dia_embeddings
            else:
                self.exp_embeddings = self.init_embeddings()
            if self.opt['share_exp_x_encoder']:
                self.x_exp_encoder = self.x_dia_encoder
            else:
                self.x_exp_encoder = self.build_encoder(opt, self.exp_embeddings)
            # print("WARNING: Using linear heads for x and y!")
            # self.x_exp_head = self.build_head(opt, outdim=opt['embedding_size'], num_layers=1)
            self.x_exp_head = nn.Dropout(p=0)

            if self.opt['share_exp_y_encoder']:
                self.y_exp_encoder = self.y_dia_encoder
            else:
                self.y_exp_encoder = self.build_encoder(opt, self.exp_embeddings)
            # print("WARNING: Using linear heads for x and y!")
            # self.y_exp_head = self.build_head(opt, outdim=opt['embedding_size'], num_layers=1)
            # self.y_exp_head = self.x_exp_head
            self.y_exp_head = nn.Dropout(p=0)

        # Build sentiment
        if 'sentiment' in self.opt['subtasks']:
            if self.opt['share_sen_embeddings']:
                self.sen_embeddings = self.dia_embeddings
            else:
                self.sen_embeddings = self.init_embeddings()
            if self.opt['share_sen_encoder']:
                self.x_sen_encoder = self.x_dia_encoder
            else:
                self.x_sen_encoder = self.build_encoder(opt, self.sen_embeddings)
            self.x_sen_head = self.build_head(opt, outdim=1,
                num_layers=self.opt['sen_head_layers'])

    def forward(self):
        raise NotImplementedError

    def score_dialog(self, x_vecs, y_vecs):
        # x_mask = (x_vecs != self.pad_idx)
        x_enc = self.x_dia_head(self.x_dia_encoder(x_vecs))

        if y_vecs.dtype == torch.float32:
            # Assume candidates have already been encoded (e.g., in interactive mode)
            y_enc = y_vecs
        elif y_vecs.dtype == torch.int64:
            # Assume candidates have only been vectorized
            y_enc = self.encode_dia_y(y_vecs)
        else:
            raise Exception("Unsupported type for cands: {}".format(type(y_vecs)))

        return self.score_similarity(x_enc, y_enc)

    def encode_dia_y(self, y_vecs):
        """Encodes a tensor of vectorized candidates

        :param y_vecs: a [bs, seq_len] or [bs, num_cands, seq_len](?) of vectorized
            candidates
        """
        if y_vecs.dim() == 2:
            # y_mask = (y_vecs != self.pad_idx)
            y_enc = self.y_dia_head(self.y_dia_encoder(y_vecs))
        elif y_vecs.dim() == 3:
            oldshape = y_vecs.shape
            y_vecs = y_vecs.reshape(oldshape[0] * oldshape[1], oldshape[2])
            # y_mask = (y_vecs != self.pad_idx)
            y_enc = self.y_dia_head(self.y_dia_encoder(y_vecs))
            y_enc = y_enc.reshape(oldshape[0], oldshape[1], -1)
        return y_enc

    def score_explanation(self, x_vecs, y_vecs):
        x_mask = (x_vecs != self.pad_idx)
        x_enc = self.x_exp_head(self.x_exp_encoder(x_vecs, x_mask, task='exp'))

        y_mask = (y_vecs != self.pad_idx)
        y_enc = self.y_exp_head(self.y_exp_encoder(y_vecs, y_mask, task='exp'))
        return self.score_similarity(x_enc, y_enc)

    def score_sentiment(self, x_vecs):
        x_mask = (x_vecs != self.pad_idx)
        return torch.sigmoid(self.x_sen_head(self.x_sen_encoder(x_vecs, x_mask))).squeeze(1)

    def score_similarity(self, context_h, cand_h):
        """Returns the dot product of encoded contexts and encoded candidates"""
        if self.opt['normalize_sent_emb']:
            context_h /= context_h.norm(2, dim=1, keepdim=True)
            cand_h /= cand_h.norm(2, dim=1, keepdim=True)

        if cand_h.dim() == 2:
            scores = torch.matmul(context_h, cand_h.t())
        elif cand_h.dim() == 3:
            scores = torch.bmm(context_h.unsqueeze(1), cand_h.transpose(1, 2)).squeeze(1)
        else:
            raise RuntimeError('Unexpected candidate dimensions {}'
                               ''.format(cand_h.dim()))

        return self.normalize_scores(scores)

    def normalize_scores(self, scores):
        if self.opt['scores_norm'] == 'dot':
            return scores
        elif self.opt['scores_norm'] == 'sqrt':
            return scores / math.sqrt(self.opt['embedding_size'])
        elif self.opt['scores_norm'] == 'dim':
            return scores / self.opt['embedding_size']
        else:
            raise ValueError

    def init_embeddings(self):
        embeddings = nn.Embedding(
            self.vocab_size,
            self.opt['embedding_size'],
            padding_idx=self.pad_idx
        )
        nn.init.normal_(
            embeddings.weight, mean=0, std=self.opt['embedding_size'] ** -0.5
        )
        nn.init.constant_(embeddings.weight[self.pad_idx], 0)
        return embeddings

    def build_encoder(self, opt, embeddings):
        return TransformerEncoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=self.vocab_size,
            embedding=embeddings,
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=self.pad_idx,
            learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
            embeddings_scale=opt['embeddings_scale'],
        )

    def build_head(self, opt, outdim=1, num_layers=1):
        dim = self.opt['embedding_size']
        modules = []
        for i in range(num_layers - 1):
            modules.append(nn.Linear(dim, dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(dim, outdim))
        # TEMP
        # print("WARNING: Hardcoded setting linear head to identity")
        # modules[0].weight.data.copy_(torch.eye(dim))
        # TEMP
        return nn.Sequential(*modules)

    # TEMP
    # def build_transf_layer(self, opt):
    #     return TransformerLayer(
    #         n_heads=opt['n_heads'],
    #         n_layers=opt['n_layers'],
    #         embedding_size=opt['embedding_size'],
    #         ffn_size=opt['ffn_size'],
    #         vocabulary_size=self.vocab_size,
    #         embedding=None,
    #         attention_dropout=opt['attention_dropout'],
    #         relu_dropout=opt['relu_dropout'],
    #         padding_idx=self.pad_idx,
    #         learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
    #         embeddings_scale=opt['embeddings_scale'],
    #     )


class Identity(nn.Module):
    def forward(self, x):
        return x



### TEMP ###
# class TransformerLayer(nn.Module):
#     """Transformer model"""
#     def __init__(self,
#                  n_heads,
#                  n_layers,
#                  embedding_size,
#                  ffn_size,
#                  vocabulary_size,
#                  embedding=None,
#                  attention_dropout=0.0,
#                  relu_dropout=0.0,
#                  padding_idx=None,
#                  learn_positional_embeddings=False,
#                  embeddings_scale=False,
#                  ):
#         super(TransformerLayer, self).__init__()

#         self.embedding_size = embedding_size
#         self.ffn_size = ffn_size
#         self.n_layers = n_layers
#         self.n_heads = n_heads
#         self.dim = embedding_size
#         self.embeddings_scale = embeddings_scale

#         self.out_dim = embedding_size
#         assert embedding_size % n_heads == 0, \
#             'Transformer embedding size must be a multiple of n_heads'
#         n_positions = 1024  # TODO: use truncate or sth

#         assert(embedding is None)

#         # build the model
#         self.attentions = nn.ModuleList()
#         self.layer_norm1 = nn.ModuleList()
#         self.ffns = nn.ModuleList()
#         self.layer_norm2 = nn.ModuleList()

#         for _ in range(self.n_layers):
#             self.attentions.append(
#                 MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
#             )
#             self.layer_norm1.append(nn.LayerNorm([embedding_size]))
#             self.ffns.append(
#                 TransformerFFN(embedding_size, ffn_size, dropout=relu_dropout)
#             )
#             self.layer_norm2.append(nn.LayerNorm([embedding_size]))

#     def forward(self, input, mask):
#         """A derivative of TransformerEncoder, stripping away embeddings-related stuff"""
#         tensor = input
#         tensor *= mask.unsqueeze(-1).float()
#         for i in range(self.n_layers):
#             tensor = tensor + self.attentions[i](tensor, mask=mask)
#             tensor = self.normalize(tensor, self.layer_norm1[i])
#             tensor = tensor + self.ffns[i](tensor, mask)
#             tensor = self.normalize(tensor, self.layer_norm2[i])

#             tensor *= mask.unsqueeze(-1).float()

#         # divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1e-20)
#         # output = tensor.sum(dim=1) / divisor
#         output = tensor.sum(dim=1)

#         return output

#     def normalize(self, tensor, norm_layer):
#         size = tensor.size()
#         return norm_layer(tensor.view(-1, self.dim)).view(size)
