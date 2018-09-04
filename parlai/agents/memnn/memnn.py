# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.thread_utils import SharedTable
from parlai.core.utils import round_sigfigs, padded_tensor

import torch
from torch import nn

import os

from .modules import MemNN, opt_to_kwargs


class MemnnAgent(TorchAgent):
    """Memory Network agent."""

    @staticmethod
    def add_cmdline_args(argparser):
        arg_group = argparser.add_argument_group('MemNN Arguments')
        arg_group.add_argument(
            '--init-model', type=str, default=None,
            help='load dict/model/opts from this path')
        arg_group.add_argument(
            '-esz', '--embedding-size', type=int, default=128,
            help='size of token embeddings')
        arg_group.add_argument(
            '-hops', '--hops', type=int, default=3,
            help='number of memory hops')
        arg_group.add_argument(
            '--memsize', type=int, default=32,
            help='size of memory')
        arg_group.add_argument(
            '-mtf', '--time-features', type='bool', default=True,
            help='use time features for memory embeddings')
        arg_group.add_argument(
            '--position-encoding', type='bool', default=False,
            help='use position encoding instead of bag of words embedding')
        arg_group.add_argument(
            '--output', type=str, default='rank', choices=('rank', 'generate'),
            help='type of output (rank|generate)')
        arg_group.add_argument(
            '--rnn-layers', type=int, default=2,
            help='number of hidden layers in RNN decoder for generative output')
        arg_group.add_argument(
            '--dropout', type=float, default=0.1,
            help='dropout probability for RNN decoder training')
        TorchAgent.add_cmdline_args(argparser)
        MemnnAgent.dictionary_class().add_cmdline_args(argparser)
        return arg_group

    def __init__(self, opt, shared=None):
        init_model = None
        if not shared:  # only do this on first setup
            # first check load path in case we need to override paths
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                # check first for 'init_model' for loading model from file
                init_model = opt['init_model']

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # next check for 'model_file', this would override init_model
                init_model = opt['model_file']

            if init_model is not None:
                # if we are loading a model, should load its dict too
                if (os.path.isfile(init_model + '.dict') or
                        opt['dict_file'] is None):
                    opt['dict_file'] = init_model + '.dict'
        super().__init__(opt, shared)

        # all instances may need some params
        self.id = 'MemNN'
        self.memsize = opt['memsize']
        self.use_time_features = opt['time_features']

        self.model_cuda = self.use_cuda
        self.use_cuda = False  # override parent

        if shared:
            # set up shared properties
            self.model = shared['model']
            self.metrics = shared['metrics']
            self.decoder = shared['decoder']
        else:
            self.metrics = {'loss': 0.0, 'batches': 0, 'rank': 0}

            if opt['time_features']:
                for i in range(self.memsize):
                    self.dict[self._time_feature(i)] = 100000000 + i

            # initialize model from scratch
            self._init_model()
            if init_model is not None:
                print('Loading existing model parameters from ' + init_model)
                self.load(init_model)

        # set up criteria
        self.rank_loss = nn.CrossEntropyLoss()  # TODO: rank loss option?
        # self.rank_loss = nn.MultiMarginLoss(margin=1)
        # self.gen_loss = nnCrossEntropyLoss(ignore_index=self.NULL_IDX)
        if self.use_cuda:
            self.rank_loss.cuda()
            # self.gen_loss.cuda()

        if 'train' in self.opt.get('datatype', ''):
            # set up optimizer
            optim_params = [p for p in self.model.parameters() if
                            p.requires_grad]
            # if self.decoder is not None:
            #     optim_params.extend([p for p in self.decoder.parameters() if
            #                          p.requires_grad])
            self._init_optim(optim_params)

    def _init_model(self):
        opt = self.opt
        kwargs = opt_to_kwargs(opt)
        self.model = MemNN(
            len(self.dict), opt['embedding_size'], use_cuda=self.model_cuda,
            padding_idx=self.NULL_IDX,
            **kwargs)

        self.decoder = None
        # if opt['output'] == 'generate':
        #     self.decoder = Decoder(opt['embedding_size'], opt['embedding_size'],
        #                            opt['rnn_layers'], opt, self.dict)
        # elif opt['output'] != 'rank' and opt['output'] != 'r':
        #     raise NotImplementedError('Output type not supported.')

        # if self.use_cuda and self.decoder is not None:
        #     # don't call cuda on self.model, it is split cuda and cpu
        #     self.decoder.cuda()

    def _time_feature(self, i):
        return '__TF{}__'.format(i)

    def share(self):
        shared = super().share()
        shared['model'] = self.model
        shared['decoder'] = self.decoder
        if self.opt.get('numthreads', 1) > 1 and isinstance(self.metrics, dict):
            torch.set_num_threads(1)
            # move metrics and model to shared memory
            self.metrics = SharedTable(self.metrics)
            self.model.share_memory()
            if self.decoder is not None:
                self.decoder.share_memory()
        shared['metrics'] = self.metrics
        return shared

    def update_params(self):
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            if self.decoder is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.decoder.parameters(), self.clip)
        self.optimizer.step()

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        super().reset_metrics()
        self.metrics['loss'] = 0.0
        self.metrics['batches'] = 0
        self.metrics['rank'] = 0

    def report(self):
        """Report loss and mean_rank from model's perspective."""
        m = {}
        batches = self.metrics['batches']
        if batches > 0:
            if self.metrics['loss'] > 0:
                m['loss'] = self.metrics['loss']
            if self.metrics['rank'] > 0:
                m['mean_rank'] = self.metrics['rank'] / batches
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

    def vectorize(self, *args, **kwargs):
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        kwargs['split_lines'] = True
        obs = args[0]
        if 'labels' in obs and 'label_candidates' in obs:
            # we aren't going to rank them, don't waste time
            del obs['label_candidates']
        return super().vectorize(*args, **kwargs)

    def get_dialog_history(self, *args, **kwargs):
        kwargs['add_p1_after_newln'] = True  # will only happen if -pt True
        return super().get_dialog_history(*args, **kwargs)

    def _build_mems(self, mems):
        bsz = len(mems)
        num_mems = max(len(mem) for mem in mems)
        if num_mems > self.memsize:
            raise RuntimeError('TODO: truncate max mem size')

        seqlen = max(len(m) for mem in mems for m in mem)
        if self.use_time_features:
            seqlen += 1
        padded = torch.LongTensor(bsz, num_mems, seqlen).fill_(0)

        for i, mem in enumerate(mems):
            for j, m in enumerate(mem):
                padded[i, j, :len(m)] = m

        if self.use_time_features:
            for i in range(num_mems):
                padded[:, i, -1] = self.dict[self._time_feature(i)]

        # if self.use_cuda:
        #     padded = padded.cuda()

        return padded

    def train_step(self, batch):
        if batch.text_vec is None:
            return
        batchsize = batch.text_vec.size(0)
        self.model.train()
        self.optimizer.zero_grad()
        mems = self._build_mems(batch.memory_vecs)

        cands, label_inds = batch.label_vec.unique(return_inverse=True)
        cands.unsqueeze_(1)
        label_inds.squeeze_(1)

        scores = self.model(batch.text_vec, mems, cands)
        loss = self.rank_loss(scores, label_inds)

        self.metrics['loss'] += loss.item()
        self.metrics['batches'] += batchsize
        _, ranks = scores.sort(1, descending=True)
        for b in range(batchsize):
            self.metrics['rank'] += 1 + (ranks[b] == label_inds[b]).nonzero().item()
        loss.backward()
        self.update_params()

        # get predictions but not full rankings--too slow to get hits@1 score
        preds = [self.dict[cands[row[0]].item()] for row in ranks]
        return Output(preds)

    def _build_cands(self, batch):
        if not batch.candidates:
            return None, None
        cand_inds = [i for i in range(len(batch.candidates))
                     if batch.candidates[i]]
        cands = [batch.candidate_vecs[i] for i in cand_inds]
        for i, c in enumerate(cands):
            cands[i] = padded_tensor(c, use_cuda=self.use_cuda)[0]
        return cands, cand_inds

    def eval_step(self, batch):
        if batch.text_vec is None:
            return
        batchsize = batch.text_vec.size(0)
        self.model.eval()

        mems = self._build_mems(batch.memory_vecs)
        cands, cand_inds = self._build_cands(batch)
        scores = self.model(batch.text_vec, mems, cands)
        # label_inds = batch.label_vec.new(range(batch.label_vec.size(0)))
        # loss = self.rank_loss(scores, label_inds)
        # self.metrics['loss'] += loss
        self.metrics['batches'] += batchsize
        _, ranks = scores.sort(1, descending=True)
        # for b in range(batchsize):
        #     self.metrics['rank'] += 1 + (ranks[b] == b).nonzero().item()

        preds, cand_preds = None, None
        if batch.candidates:
            cand_preds = [[batch.candidates[b][i.item()] for i in row]
                          for b, row in enumerate(ranks)]
            preds = [row[0] for row in cand_preds]
        else:
            cand_preds = [[self.dict[i.item()] for i in row]
                          for row in ranks]
            preds = [row[0] for row in cand_preds]

        return Output(preds, cand_preds)

    def score(self, cands, output_embeddings):
        last_cand = None
        max_len = max([len(c) for c in cands])
        scores = output_embeddings.data.new(len(cands), max_len)
        for i, cand_list in enumerate(cands):
            if last_cand != cand_list:
                candidate_lengths, candidate_indices = to_tensors(cand_list, self.dict)
                candidate_embeddings = self.model.answer_embedder(candidate_lengths, candidate_indices)
                if self.use_cuda:
                    candidate_embeddings = candidate_embeddings.cuda()
                last_cand = cand_list
            scores[i, :len(cand_list)] = self.model.score.one_to_many(output_embeddings[i].unsqueeze(0), candidate_embeddings).squeeze(0)
        return scores

    def ranked_predictions(self, cands, scores):
        # return [' '] * len(self.answers)
        _, inds = scores.sort(descending=True, dim=1)
        return [[cands[i][j] for j in r if j < len(cands[i])]
                for i, r in enumerate(inds)]

    def decode(self, output_embeddings, ys=None):
        batchsize = output_embeddings.size(0)
        hn = output_embeddings.unsqueeze(0).expand(self.opt['rnn_layers'], batchsize, output_embeddings.size(1))
        x = self.model.answer_embedder(torch.LongTensor([1]), self.START_TENSOR)
        xes = x.unsqueeze(1).expand(x.size(0), batchsize, x.size(1))

        loss = 0
        output_lines = [[] for _ in range(batchsize)]
        done = [False for _ in range(batchsize)]
        total_done = 0
        idx = 0
        while(total_done < batchsize) and idx < self.longest_label:
            # keep producing tokens until we hit END or max length for each ex
            if self.use_cuda:
                xes = xes.cuda()
                hn = hn.contiguous()
            preds, scores = self.decoder(xes, hn)
            if ys is not None:
                y = ys[0][:, idx]
                temp_y = y.cuda() if self.use_cuda else y
                loss += self.gen_loss(scores, temp_y)
            else:
                y = preds
            # use the true token as the next input for better training
            xes = self.model.answer_embedder(torch.LongTensor(preds.numel()).fill_(1), y).unsqueeze(0)

            for b in range(batchsize):
                if not done[b]:
                    token = self.dict.vec2txt(preds[b])
                    if token == self.END:
                        done[b] = True
                        total_done += 1
                    else:
                        output_lines[b].append(token)
            idx += 1
        return output_lines, loss


    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        if path:
            checkpoint = {}
            checkpoint['memnn'] = self.model.state_dict()
            checkpoint['memnn_optim'] = self.optimizers['memnn'].state_dict()
            if self.decoder is not None:
                checkpoint['decoder'] = self.decoder.state_dict()
                checkpoint['decoder_optim'] = self.optimizers['decoder'].state_dict()
                checkpoint['longest_label'] = self.longest_label
            with open(path, 'wb') as write:
                torch.save(checkpoint, write)

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda cpu, _: cpu)
        self.model.load_state_dict(checkpoint['memnn'])
        self.optimizers['memnn'].load_state_dict(checkpoint['memnn_optim'])
        if self.decoder is not None:
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.optimizers['decoder'].load_state_dict(checkpoint['decoder_optim'])
            self.longest_label = checkpoint['longest_label']


def to_tensors(sentences, dictionary):
    lengths = []
    indices = []
    for sentence in sentences:
        tokens = dictionary.txt2vec(sentence)
        lengths.append(len(tokens))
        indices.extend(tokens)
    lengths = torch.LongTensor(lengths)
    indices = torch.LongTensor(indices)
    return lengths, indices


def build_cands(exs, dict):
    dict_list = list(dict.tok2ind.keys())[1:]  # skip NULL
    cands = []
    for ex in exs:
        if 'label_candidates' in ex:
            cands.append(ex['label_candidates'])
        else:
            cands.append(dict_list)
            if 'labels' in ex:
                cands[-1] += [l for l in ex['labels'] if l not in dict.tok2ind]
    return cands
