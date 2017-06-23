import logging
import numpy as np
import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse

MAX_TIME_FEATURES = 20


class MemNN(nn.Module):
    def __init__(self, opt, freqs):
        logging.info('Initializing model')
        super(MemNN, self).__init__()

        self.opt = opt

        # Prepare features
        num_features = freqs.numel()
        self.extra_features_slots = 0
        if opt['speaker_features']:
            self.speaker_features = torch.LongTensor([num_features, num_features + 1])
            num_features += 2
            self.extra_features_slots += 1
        if opt['time_features']:
            self.time_features = torch.LongTensor(range(num_features, num_features + MAX_TIME_FEATURES))
            num_features += MAX_TIME_FEATURES
            self.extra_features_slots += 1

        # Initialize feature weights
        freqs = freqs.float()
        if opt['weighting_scheme'] is not None:
            if opt['weighting_scheme'] == 'log':
                feature_weights = (freqs + 1).log().reciprocal()
            elif opt['weighting_scheme'] == 'smooth':
                feature_weights = 0.001 / (0.001 + freqs / freqs.sum())
            else:
                raise NotImplementedError
            self.feature_weights = torch.cat([feature_weights, torch.ones(num_features - freqs.numel())])
        else:
            self.feature_weights = None

        def embedding():
            layer = Embed(
                num_features, opt['embedding_size'], padding_idx=0)
            return layer

        self.B = embedding()
        self.L = embedding()
        self.A = embedding()
        self.C = embedding()
        self.memory_hop = Hop(
            opt['embedding_size'], gating=opt['gating'], transform=False, temperature=opt['mem_temp'])

        if opt['forward_time'] > 1 and opt['score'] != 'triple':
            self.forward_hop = Hop(
                opt['embedding_size'], gating=False, transform=True, temperature=opt['forward_temperature'])

        if opt['score'] == 'dot':
            self.score = DotScore()
        elif opt['score'] == 'concat':
            self.score = ConcatScore(opt['embedding_size'], opt['score_hidden_size'])
        elif opt['score'] == 'triple':
            # if self.opt.forward_time < 2:
            #     raise NotImplementedError
            self.score = ConcatTripleScore(opt['embedding_size'], opt['score_hidden_size'])
        else:
            raise NotImplementedError

        if opt['cuda']:
            self.score.cuda()
            if hasattr(self, 'memory_hop'):
                self.memory_hop.cuda()
            if hasattr(self, 'forward_hop'):
                self.forward_hop.cuda()
                self.cuda()

        self.original_cuda_params = [(p, p.data) for p in self.parameters() if p.data.is_cuda]

    def copy_cuda_from_original(self):
        for p, orig_p_data in self.original_cuda_params:
            p.data = orig_p_data.cuda(async=True)

    def speaker_feature(self, t):
        return self.speaker_features[(self.opt['mem_size'] - t) % 2]

    def time_feature(self, t):
        return self.time_features[min(t, MAX_TIME_FEATURES - 1)]

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
                        if self.opt['speaker_features']:
                            updated_memories[dst_offset] = self.speaker_feature(j)
                            dst_offset += 1
                        updated_memories[dst_offset:dst_offset + length] = memories[src_offset:src_offset + length]
                        src_offset += length
                        dst_offset += length
            memory_lengths += memory_lengths.ne(0).long() * self.extra_features_slots
            memories.set_(updated_memories)

    def forward(self, memories, queries, answers,
                memory_lengths, query_lengths, answer_lengths):
        self.update_memories_with_extra_features_(memory_lengths, memories)

        feature_weights_var = Variable(self.feature_weights, requires_grad=False)
        in_memory_embeddings = self.A(memory_lengths, memories, feature_weights_var)
        out_memory_embeddings = self.C(memory_lengths, memories, feature_weights_var)
        query_embeddings = self.B(query_lengths, queries, feature_weights_var)
        answer_embeddings = None
        if answer_lengths.numel() > 0:
            answer_embeddings = self.L(answer_lengths, answers, feature_weights_var)
        attention_mask = Variable(memory_lengths.data.ne(0), requires_grad=False)

        if self.opt['cuda']:
            in_memory_embeddings = in_memory_embeddings.cuda(async=True)
            out_memory_embeddings = out_memory_embeddings.cuda(async=True)
            query_embeddings = query_embeddings.cuda(async=True)
            if answer_lengths.numel() > 0:
                answer_embeddings = answer_embeddings.cuda(async=True)
            attention_mask = attention_mask.cuda(async=True)

        for _ in range(self.opt['hops']):
            query_embeddings = self.memory_hop(
                query_embeddings, in_memory_embeddings, out_memory_embeddings, attention_mask)

        query_embeddings = normalize(query_embeddings)
        return query_embeddings, answer_embeddings

    def embed(self, lookup_name, lengths, indices):
        feature_weights_var = Variable(self.feature_weights, volatile=True)
        if not isinstance(lengths, Variable):
            lengths = Variable(lengths, volatile=True)
        if not isinstance(indices, Variable):
            indices = Variable(indices, volatile=True)
        return getattr(self, lookup_name)(lengths, indices, feature_weights_var)


def idxattr(name, idx):
    return '{}_{}'.format(name, idx)


def setidxattr(module, name, idx, value):
    return module.__setattr__(idxattr(name, idx), value)


def getidxattr(module, name, idx):
    return module.__getattr__(idxattr(name, idx))


class SafeNorm(Function):
    """The default Norm function doesn't behave well with 0 vectors"""

    def __init__(self, dim=None):
        super(SafeNorm, self).__init__()
        self.dim = dim

    def forward(self, input):
        output = input.norm(2, self.dim)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        big_grad_output = grad_output.expand_as(input)
        pos_mask = output.ne(0).float()
        neg_mask = output.eq(0).float()
        safe_output = pos_mask * output + neg_mask
        big_safe_output = safe_output.expand_as(input)
        return input.mul(big_grad_output).div(big_safe_output)


def safe_norm(input, dim):
    return SafeNorm(dim=dim)(input)


class Normalize(nn.Module):

    def forward(self, input):
        norm = safe_norm(input, input.dim() - 1)
        pos_mask = Variable(norm.data.ne(0).float(), requires_grad=False)
        zero_mask = Variable(norm.data.eq(0).float(), requires_grad=False)
        safe_input_norm = pos_mask * norm + zero_mask
        return input / safe_input_norm.expand_as(input)


def normalize(x):
    return Normalize()(x)


class SparseWeightedAverage(Function):

    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def forward(self, embeddings, lengths, inputs, weights):
        if inputs.numel() == 0:
            result = torch.zeros(lengths.numel(), embeddings.size(1))
            inv_sum_weights = weights.new()
            example_indices = inputs.new()
        else:
            num_examples = lengths.size(0)
            num_embeddings = embeddings.size(0)
            indices = inputs.new(2, inputs.numel())
            input_weights = weights.gather(0, inputs)
            inv_sum_weights = torch.Tensor(inputs.numel())
            offset = 0
            for i, length in enumerate(lengths):
                if length > 0:
                    indices[0, offset:offset + length].fill_(i)
                    indices[1, offset:offset + length] = inputs[offset:offset + length].sort()[0]
                    inv_sum_weights[offset:offset + length].fill_(1. / input_weights[offset:offset + length].sum())
                offset += length
            values = weights.gather(0, indices[1]).mul_(inv_sum_weights)
            sp_inputs = sparse.FloatTensor(indices, values, torch.Size(
                [num_examples, num_embeddings])).coalesce()
            result = torch.dsmm(sp_inputs, embeddings)
            example_indices = indices[0]
        self.mark_non_differentiable(inv_sum_weights, example_indices)
        self.save_for_backward(embeddings, lengths, inputs, weights, inv_sum_weights, example_indices)
        return result, inv_sum_weights, example_indices

    def backward(self, grad_output, grad_inv_sum_weights, grad_example_indices):
        embeddings, lengths, inputs, weights, inv_sum_weights, example_indices = self.saved_tensors
        if inputs.numel() == 0:
            return sparse.FloatTensor(), None, None, None
        num_examples = lengths.size(0)
        unique, inverse = np.unique(inputs.numpy(), return_inverse=True)
        order = torch.from_numpy(inverse.argsort(kind='mergesort'))  # stable sort
        unique = torch.from_numpy(unique)
        inverse = torch.from_numpy(inverse)
        num_changed = unique.size(0)
        sorted_inverse = inverse.gather(0, order)
        sorted_inputs = inputs.gather(0, order)
        indices = inputs.new(2, inputs.numel())
        indices[0].copy_(sorted_inverse)
        indices[1] = example_indices.gather(0, order)
        values = weights.gather(0, sorted_inputs).mul_(inv_sum_weights.gather(0, order))
        sp_inputs = sparse.FloatTensor(indices, values, torch.Size(
            [num_changed, num_examples])).coalesce()
        grad_values = torch.dsmm(sp_inputs, grad_output)
        if self.padding_idx is not None:
            grad_values[inverse[self.padding_idx]].zero_()
        result = sparse.FloatTensor(unique.unsqueeze(0), grad_values, embeddings.size())
        return result, None, None, None


class Embed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(Embed, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        # TODO remove when pytorch updates
        try:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim), sparse_grad=True)
        except TypeError:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-0.1, 0.1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, lengths, indices, weights=None):
        embeddings_size = [x for x in lengths.size()] + [self.embedding_dim]
        lengths = lengths.view(-1)
        embeddings = SparseWeightedAverage(self.padding_idx)(
            self.weight, lengths, indices, weights)[0].view(-1, self.embedding_dim)
        embeddings = normalize(embeddings)
        return embeddings.view(*embeddings_size)


def _violator_counts(pairwise_losses):
    return Variable(pairwise_losses.data.gt(0).float().sum(dim=1), requires_grad=False)


class DotScore(nn.Module):

    def one_to_one(self, query_embeddings, answer_embeddings, reply_embeddings=None):
        return (query_embeddings * answer_embeddings).sum(dim=1).squeeze(1)

    def one_to_many(self, query_embeddings, answer_embeddings, reply_embeddings=None):
        return query_embeddings.mm(answer_embeddings.t())


class ConcatScore(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(ConcatScore, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # By default the hidden layer size is the same as the embedding size
        # We could make this configurable
        self.A = nn.Linear(input_size, hidden_size, bias=False)
        self.B = nn.Linear(input_size, hidden_size, bias=False)
        for i in range(self.num_layers):
            setidxattr(self, 'C', i, nn.Linear(hidden_size, hidden_size, bias=False))
        self.non_linearity = nn.LeakyReLU()
        # self.non_linearity = nn.Tanh()
        self.D = nn.Linear(hidden_size, 1, bias=False)

    def mlp(self, hidden):
        for i in range(self.num_layers):
            hidden = self.non_linearity(getidxattr(self, 'C', i)(hidden))
        return self.D(hidden).squeeze(1)

    def one_to_one(self, query_embeddings, answer_embeddings, reply_embeddings=None):
        hidden = self.non_linearity(self.A(query_embeddings) + self.B(answer_embeddings))
        return self.mlp(hidden)

    def one_to_many(self, query_embeddings, answer_embeddings, reply_embeddings=None):
        num_q = query_embeddings.size(0)
        num_a = answer_embeddings.size(0)
        q = self.A(query_embeddings).unsqueeze(1)
        a = self.B(answer_embeddings).unsqueeze(0)
        size = [num_q, num_a, self.hidden_size]
        hidden = self.non_linearity((q.expand(*size) + a.expand(*size)).view(num_q * num_a, -1))
        return self.mlp(hidden).view(num_q, num_a)


class ConcatTripleScore(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(ConcatTripleScore, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.A = nn.Linear(input_size, hidden_size, bias=False)
        self.B = nn.Linear(input_size, hidden_size, bias=False)
        self.C = nn.Linear(input_size, hidden_size, bias=False)
        for i in range(self.num_layers):
            setidxattr(self, 'D', i, nn.Linear(hidden_size, hidden_size, bias=False))
        self.non_linearity = nn.LeakyReLU(inplace=False)
        # self.non_linearity = nn.Tanh(inplace=False)
        self.E = nn.Linear(hidden_size, 1, bias=False)

    def mlp(self, hidden):
        for i in range(self.num_layers):
            hidden = self.non_linearity(getidxattr(self, 'D', i)(hidden))
        return self.E(hidden).squeeze(1)

    def one_to_one(self, query_embeddings, answer_embeddings, reply_embeddings=None):
        hidden = self.A(query_embeddings)
        if answer_embeddings is not None:
            hidden = hidden + self.B(answer_embeddings)
        if reply_embeddings is not None:
            hidden = hidden + self.C(reply_embeddings)
        return self.mlp(hidden)

    def one_to_many(self, query_embeddings, answer_embeddings, reply_embeddings=None):
        num_q = query_embeddings.size(0)
        num_a = answer_embeddings.size(0) if answer_embeddings is not None else 1
        num_r = reply_embeddings.size(0) if reply_embeddings is not None else 1
        size = [num_q, num_a, num_r, self.hidden_size]
        expansion = self.A(query_embeddings).unsqueeze(1).unsqueeze(2).expand(*size)
        if answer_embeddings is not None:
            expansion = expansion + self.B(answer_embeddings).unsqueeze(0).unsqueeze(2).expand(*size)
        if reply_embeddings is not None:
            expansion = expansion + self.C(reply_embeddings).unsqueeze(0).unsqueeze(1).expand(*size)
        return self.mlp(expansion.contiguous().view(-1, self.hidden_size)).view(num_q, -1)


class Hop(nn.Module):

    def __init__(self, embedding_size, gating, transform, temperature=1):
        super(Hop, self).__init__()
        self.embedding_size = embedding_size
        self.inv_temperature = 1. / temperature
        self.gating = nn.Linear(embedding_size, embedding_size, bias=False) if gating else None
        self.w = nn.Linear(embedding_size, embedding_size, bias=False) if transform else None

    def forward(self, query_embeddings, in_memory_embeddings, out_memory_embeddings, attention_mask=None):
        attention = torch.bmm(in_memory_embeddings, query_embeddings.unsqueeze(2)).squeeze(2) * self.inv_temperature
        if attention_mask is not None:
            # exclude masked elements from the softmax
            attention = attention_mask.float() * attention + (1 - attention_mask.float()) * -1e20
        probs = F.softmax(attention).unsqueeze(1)
        memory_output = torch.bmm(probs, out_memory_embeddings).squeeze(1)
        if self.gating is not None:
            memory_output = self.gating(memory_output).sigmoid_() * memory_output
        output = memory_output + query_embeddings
        if self.w is not None:
            output = self.w(output)
        return output


class ForwardPrediction(nn.Module):

    def __init__(self, hop):
        super(ForwardPrediction, self).__init__()
        self.hop = hop

    def forward(self, query_embeddings, answer_embeddings):
        # for each time step, do one hop over all answers from the corresponding time step
        batch_size = query_embeddings.size(0)
        embedding_size = query_embeddings.size(1)
        for i in range(answer_embeddings.size(1)):
            memory_embeddings = answer_embeddings[:, i].unsqueeze(0).expand(batch_size, batch_size, embedding_size)
            query_embeddings = self.hop(query_embeddings, memory_embeddings, memory_embeddings)

        return query_embeddings


class WarpLossWithBatch(nn.Module):

    def __init__(self, margin, score):
        super(WarpLossWithBatch, self).__init__()
        self.margin = margin
        self.score = score

    def warp_losses(self, dots, pos_dots):
        losses = dots - pos_dots.unsqueeze(1).expand_as(dots) + self.margin
        violator_counts = _violator_counts(losses)
        # The loss for each example is the average of the violator's pairwise
        # losses. We use clamp to avoid division by 0
        # We need to cancel the effect of the opt.margin loss on the diagonal
        return (losses.clamp(min=0).sum(dim=1) - self.margin) / (violator_counts - 1).clamp(min=1)

    def forward(self, query_embeddings, answer_embeddings, reply_embeddings=None):
        if reply_embeddings is not None:
            dots = torch.cat([
                self.score.one_to_many(query_embeddings, answer_embeddings, reply_embeddings),
                self.score.one_to_many(query_embeddings, answer_embeddings, None),
                self.score.one_to_many(query_embeddings, None, reply_embeddings),
            ], 1).repeat(2, 1)
            pos_dots = torch.cat([
                self.score.one_to_one(query_embeddings, answer_embeddings, reply_embeddings),
                self.score.one_to_one(query_embeddings, answer_embeddings, None),
            ], 0)
        else:
            dots = self.score.one_to_many(query_embeddings, answer_embeddings, reply_embeddings)
            pos_dots = self.score.one_to_one(query_embeddings, answer_embeddings, reply_embeddings)
        return self.warp_losses(dots, pos_dots).mean()
