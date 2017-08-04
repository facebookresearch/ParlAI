import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import softmax

from functools import lru_cache


class MemNN(nn.Module):
    def __init__(self, opt, freqs):
        super(MemNN, self).__init__()
        self.opt = opt

        # Prepare features
        self.num_time_features = opt['mem_size']
        num_features = freqs.numel()
        self.extra_features_slots = 0
        if opt['time_features']:
            self.time_features = torch.LongTensor(range(num_features,
                num_features + self.num_time_features))
            num_features += self.num_time_features
            self.extra_features_slots += 1

        def embedding():
            return Embed(num_features, opt['embedding_size'],
                position_encoding=opt['position_encoding'], padding_idx=0)

        self.query_embedder = embedding()
        self.answer_embedder = embedding()
        self.in_memory_embedder = embedding()
        self.out_memory_embedder = embedding()
        self.memory_hop = Hop(opt['embedding_size'])

        self.score = DotScore()

        if opt['cuda']:
            self.score.cuda()
            if hasattr(self, 'memory_hop'):
                self.memory_hop.cuda()

        self.original_cuda_params = [(p, p.data) for p in self.parameters() if p.data.is_cuda]

    def time_feature(self, t):
        return self.time_features[min(t, self.num_time_features - 1)]

    def update_memories_with_extra_features_(self, memory_lengths, memories):
        memory_lengths = memory_lengths.data
        memories = memories.data
        if self.extra_features_slots > 0:
            num_nonempty_memories = memory_lengths.ne(0).sum()
            updated_memories = memories.new(memories.numel() + num_nonempty_memories * self.extra_features_slots)
            src_offset = 0
            dst_offset = 0
            for i in range(memory_lengths.size(0)):
                for j in range(self.opt['mem_size']):
                    length = memory_lengths[i, j]
                    if length > 0:
                        if self.opt['time_features']:
                            updated_memories[dst_offset] = self.time_feature(j)
                            dst_offset += 1
                        updated_memories[dst_offset:dst_offset + length] = memories[src_offset:src_offset + length]
                        src_offset += length
                        dst_offset += length
            memory_lengths += memory_lengths.ne(0).long() * self.extra_features_slots
            memories.set_(updated_memories)

    def forward(self, memories, queries, answers,
                memory_lengths, query_lengths, answer_lengths):
        self.update_memories_with_extra_features_(memory_lengths, memories)

        in_memory_embeddings = self.in_memory_embedder(memory_lengths, memories)
        out_memory_embeddings = self.out_memory_embedder(memory_lengths, memories)
        query_embeddings = self.query_embedder(query_lengths, queries)
        answer_embeddings = None
        if answer_lengths.numel() > 0:
            answer_embeddings = self.answer_embedder(answer_lengths, answers)
        attention_mask = Variable(memory_lengths.data.ne(0), requires_grad=False)

        if self.opt['cuda']:
            in_memory_embeddings = in_memory_embeddings.cuda(async=True)
            out_memory_embeddings = out_memory_embeddings.cuda(async=True)
            query_embeddings = query_embeddings.cuda(async=True)
            if answer_lengths.numel() > 0:
                answer_embeddings = answer_embeddings.cuda(async=True)
            attention_mask = attention_mask.cuda(async=True)

        for _ in range(self.opt['hops']):
            query_embeddings = self.memory_hop(query_embeddings,
                    in_memory_embeddings, out_memory_embeddings, attention_mask)

        return query_embeddings, answer_embeddings


class Embed(nn.Embedding):
    def __init__(self, *args, position_encoding=False, **kwargs):
        self.position_encoding = position_encoding
        super().__init__(*args, **kwargs)

    def forward(self, lengths, indices):
        lengths_mat = lengths.data
        indices = indices.data
        if lengths.dim() == 1 or lengths.size(1) == 1:
            lengths_mat = lengths_mat.squeeze().unsqueeze(0)

        input = torch.LongTensor(lengths_mat.size(0), lengths_mat.size(1), torch.max(lengths_mat))
        pad = self.padding_idx if self.padding_idx is not None else 0
        input.fill_(pad)
        emb_list = []
        offset = 0
        for i, row in enumerate(lengths_mat):
            for j, length in enumerate(row):
                if length > 0:
                    input[i, j, :length] = indices[offset:offset+length]
                offset += length
        input = Variable(input)

        for i, row in enumerate(lengths_mat):
            emb = super().forward(input[i, :, :])
            if self.position_encoding:
                emb = emb * Variable(self.position_tensor(row, emb))
            emb = torch.sum(emb, dim=1).squeeze(1)
            for j, length in enumerate(row):
                if length > 0:
                    emb[j] /= length
            emb_list.append(emb)
        embs = torch.stack(emb_list)

        if lengths.dim() == 1:
            embs = embs.squeeze(0)
        elif lengths.size(1) == 1:
            embs = embs.squeeze().unsqueeze(1)
        return embs

    @staticmethod
    @lru_cache(maxsize=32)
    def position_matrix(J, d):
        m = torch.Tensor(J, d)
        for k in range(1, d+1):
            for j in range(1, J+1):
                m[j-1, k-1] = (1 - j/J) - (k/d) * (1 - 2 * j/J)
        return m

    @staticmethod
    def position_tensor(sentence_lengths, embeddings):
        t = torch.zeros(embeddings.size())
        embedding_dim = embeddings.size()[-1]
        for i, length in enumerate(sentence_lengths):
            if length > 0:
                t[i, :length, :] = Embed.position_matrix(length, embedding_dim)
        return t


class Hop(nn.Module):
    def __init__(self, embedding_size):
        super(Hop, self).__init__()
        self.embedding_size = embedding_size
        self.linear = nn.Linear(embedding_size, embedding_size, bias=False)

    def forward(self, query_embeddings, in_memory_embeddings, out_memory_embeddings, attention_mask=None):
        attention = torch.bmm(in_memory_embeddings, query_embeddings.unsqueeze(2)).squeeze(2)
        if attention_mask is not None:
            # exclude masked elements from the softmax
            attention = attention_mask.float() * attention + (1 - attention_mask.float()) * -1e20
        probs = softmax(attention).unsqueeze(1)
        memory_output = torch.bmm(probs, out_memory_embeddings).squeeze(1)
        query_embeddings = self.linear(query_embeddings)
        output = memory_output + query_embeddings
        return output


class DotScore(nn.Module):
    def one_to_one(self, query_embeddings, answer_embeddings, reply_embeddings=None):
        return (query_embeddings * answer_embeddings).sum(dim=1).squeeze(1)

    def one_to_many(self, query_embeddings, answer_embeddings, reply_embeddings=None):
        return query_embeddings.mm(answer_embeddings.t())
