import torch
import torch.nn as nn
from torch.autograd import Variable

class HistoryEncoder(nn.Module):
    def __init__(self, opt, dictionary):
        
        emb = opt['embeddingsize']
        hsz = opt['hiddensize']
        nlayer = opt['numlayers']

        self.embedding_size = emb
        self.hidden_size = hsz
        self.num_layers = nlayer

        self.embedding = nn.Embedding(len(dictionary), emb)
        self.h_encoder = nn.LSTM(emb, hsz, nlayer)
        super().__init__(opt, dictionary)

    def forward(self, input, lengths):
        # input: [batch_size, memory_length, max_sentence_length]
        # output:
        #   history: [batch_size, memory_length, hidden_size]
        memory_embeddings = []
        for i in range(input.size(0)):
            memory = self.embedding(input[i])
            memory = memory.unsqueeze(0)
            memory_embeddings.append(memory)
        memory_embeddings = torch.cat(memory_embeddings)

        memory_length = memory_embeddings.size(1)
        batch_size = memory_embeddings.size(0)
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

        # lengths [batch_size, memory_length]
        s_idx = lengths.numpy()[0].nonzero()[0][0]
        lengths = lengths[:, s_idx:]
        history = []

        for i in range(memory_length):
            temp = memory_embeddings[:, i]
            data = sort_embeddings(temp, lengths[:, i])
            
            output, _ = self.h_encoder(data, (h0, c0))
            output, o_len = nn.utils.rnn.pad_packed_sequence(output)
            output_embedding = torch.cat([output.transpose(0, 1)[i][o_len[i]-1].unsqueeze(0) for i in range(batch_size)])
            output_embedding = output_embedding.unsqueeze(1)
            history.append(output_embedding)

        history = torch.cat(history, 1)

        return history

def sort_embeddings(embeddings, lens):
    lens, perm_idx = lens.sort(0, descending=True)
    embeddings = embeddings[perm_idx]
    data = nn.utils.rnn.pack_padded_sequence(embeddings, lens.tolist(), batch_first=True)
    return data


class QueryEncoder(nn.Module):
    def __init__(self, opt, dictionary):
        super().__init__(
        emb = opt['embeddingsize']
        hsz = opt['hiddensize']
        nlayer = opt['numlayers']

        self.embedding = nn.Embedding(len(dictionary), emb)
        self.q_encoder = nn.LSTM(emb, hsz, nlayer)

    def forward(self, input, lengths):
        batch_size = input.size(0)

        queries = self.embedding(input)
        data = sort_embeddings(queries, lengths)

        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

        output, _ = self.q_encoder(data, (h0, c0))
        output, o_len = nn.utils.rnn.pad_packed_sequence(output)
        output_embedding = torch.cat([output.transpose(0, 1)[i][o_len[i]-1].unsqueeze(0) for i in range(batch_size)])

        return output_embedding

