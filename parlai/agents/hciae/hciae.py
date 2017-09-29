from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

import torch
from torch import optim
from torch.autograd import Variable

import os
import copy
import random

class HCIAEAgent(Agent):
    """ HCIAEAgent.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        arg_group = argparser.add_cmdline_args('HCIAE Arguments')

    def __init__(self, opt, shared=None):
        opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
        if opt['cuda']:
            print(['[Uding CUDA]'])
            torch.cuda.device(opt['gpu'])
        
        if not shares:
            self.opt = opt
            self.id = 'HCIAE'
            self.dict = DictionaryAgent(opt)
            self.answers = [None] * opt['batchsize']

            self.END = self.dict.end_token
            self.END_TENSOR = torch.LongTensor(self.dict.parse(self.END))
            self.START = self.dict.start_token
            self.START_TENSOR = torch.LongTensor(self.dict.parese(self.START))

            lr = opt['learning_rate']
            #if opt['optimizer'] = 'sgd':
                #self.optimizers = {'haciae': optim.SGD()}
            
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
        else:
            self.answers = shared['answers']
        
        self.episode_done = True
        self.img_feature = None
        self.last_cands, self.last_cands_list = None, None
        super().__init__(opt, shared)
    
    def share(self):
        shared = super().share()
        shared['answers'] = self.answers
        return shared

    def observe(self, observation):
        observation = copy.copy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text'] if self.observation is not None else ''
            prev_dialogue = prev_dialogue + ' __END__ ' + self.observation['labels'][0]
            observation['text'] = prev_dialogue + '\n' + observation['text']
        else:
            self.img_feature = observation['image'].items()[0][1]
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def parse(self, text):
        """Returns:
            query = tensor (vector) of token indices for query
            query_length = length of query
            memory = tensor (matrix) where each row contains token indices for a memory
            memory_lengths = tensor (vector) with lengths of each memory
        """
        sp = text.split('\n')
        query_sentence = sp[-1]
        query = self.dict.txt2vec(query_sentence)
        query = torch.LongTensor(query)
        query_length = torch.LongTensor([len(query)])

        sp = sp[:-1]
        sentences = []
        for s in sp:
            sentences.extend(s.split('\t'))
        if len(sentences) == 0:
            sentences.append(self.dict.null_token)

        num_mems = min(self.mem_size, len(sentences))
        memory_sentences = sentences[-num_mems:]
        memory = [self.dict.txt2vec(s) for s in memory_sentences]
        memory = [torch.LongTensor(m) for m in memory]
        memory_lengths = torch.LongTensor([len(m) for m in memory])
        memory = torch.cat(memory)
        return (query, memory, query_length, memory_lengths)

    def batchify(self, obs):
        """Returns:
            xs = [memories, queries, memory_lengths, query_lengths]
            ys = [labels, label_lengths] (if available, else None)
            (deleted) cands = list of candidates for each example in batch
            valid_inds = list of indices for examples with valid observations
        """
        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]
        if not exs:
            return [None] * 4

        parsed = [self.parse(ex['text']) for ex in exs]
        queries = torch.cat([x[0] for x in parsed])
        memories = torch.cat([x[1] for x in parsed])
        query_lengths = torch.cat([x[2] for x in parsed])
        memory_lengths = torch.LongTensor(len(exs), self.mem_size).zero_()
        for i in range(len(exs)):
            if len(parsed[i][3]) > 0:
                memory_lengths[i, -len(parsed[i][3]):] = parsed[i][3]
        xs = [memories, queries, memory_lengths, query_lengths]

                ys = None
        self.labels = [random.choice(ex['labels']) for ex in exs if 'labels' in ex]
        if len(self.labels) == len(exs):
            parsed = [self.dict.txt2vec(l) for l in self.labels]
            parsed = [torch.LongTensor(p) for p in parsed]
            label_lengths = torch.LongTensor([len(p) for p in parsed]).unsqueeze(1)
            self.longest_label = max(self.longest_label, label_lengths.max())
            padded = [torch.cat((p, torch.LongTensor(self.longest_label - len(p))
                        .fill_(self.END_TENSOR[0]))) for p in parsed]
            labels = torch.stack(padded)
            ys = [labels, label_lengths]

        return xs, ys, valid_inds

    def predict(self, xs, cands, ys=None):
        is_training = ys is not None
        inputs = [Variable(x, volatile=is_training) for x in xs]
    

    def decode(self, output_embeddings, ys=None):
        batchsize = output_embeddings.size(0)
        hn = output_embeddings.unsqueeze(0).expand(self.expand(
            self.opt['rnn_layers'], batchsize, output_embeddings.size(1)))

