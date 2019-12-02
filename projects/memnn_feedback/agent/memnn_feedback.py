#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

import os
import copy
import random

from .modules import MemNN, Decoder, to_tensors


class MemnnFeedbackAgent(Agent):
    """
    Memory Network agent for question answering that supports reward-based learning
    (RBI), forward prediction (FP), and imitation learning (IM).

    For more details on settings see: https://arxiv.org/abs/1604.06045.

    Models settings 'FP', 'RBI', 'RBI+FP', and 'IM_feedback' assume that
    feedback and reward for the current example immediatly follow the query
    (add ':feedback' argument when specifying task name).

    python examples/train_model.py --setting 'FP'
    -m "projects.memnn_feedback.agent.memnn_feedback:MemnnFeedbackAgent"
    -t "projects.memnn_feedback.tasks.dbll_babi.agents:taskTeacher:3_p0.5:feedback"
    """

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        arg_group = argparser.add_argument_group('MemNN Arguments')
        arg_group.add_argument(
            '-lr', '--learning-rate', type=float, default=0.01, help='learning rate'
        )
        arg_group.add_argument(
            '--embedding-size', type=int, default=128, help='size of token embeddings'
        )
        arg_group.add_argument(
            '--hops', type=int, default=3, help='number of memory hops'
        )
        arg_group.add_argument(
            '--mem-size', type=int, default=100, help='size of memory'
        )
        arg_group.add_argument(
            '--time-features',
            type='bool',
            default=True,
            help='use time features for memory embeddings',
        )
        arg_group.add_argument(
            '--position-encoding',
            type='bool',
            default=False,
            help='use position encoding instead of bag of words embedding',
        )
        arg_group.add_argument(
            '-clip',
            '--gradient-clip',
            type=float,
            default=0.2,
            help='gradient clipping using l2 norm',
        )
        arg_group.add_argument(
            '--output', type=str, default='rank', help='type of output (rank|generate)'
        )
        arg_group.add_argument(
            '--rnn-layers',
            type=int,
            default=2,
            help='number of hidden layers in RNN decoder for generative output',
        )
        arg_group.add_argument(
            '--dropout',
            type=float,
            default=0.1,
            help='dropout probability for RNN decoder training',
        )
        arg_group.add_argument(
            '--optimizer', default='sgd', help='optimizer type (sgd|adam)'
        )
        arg_group.add_argument(
            '--no-cuda',
            action='store_true',
            default=False,
            help='disable GPUs even if available',
        )
        arg_group.add_argument(
            '--gpu', type=int, default=-1, help='which GPU device to use'
        )
        arg_group.add_argument(
            '--setting',
            type=str,
            default='IM',
            help='choose among IM, IM_feedback, RBI, FP, RBI+FP',
        )
        arg_group.add_argument(
            '--num-feedback-cands',
            type=int,
            default=6,
            help='number of feedback candidates',
        )
        arg_group.add_argument(
            '--single_embedder',
            type='bool',
            default=False,
            help='number of embedding matrices in the model',
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
        if opt['cuda']:
            print('[ Using CUDA ]')
            torch.cuda.device(opt['gpu'])

        if not shared:
            self.id = 'MemNN'
            self.dict = DictionaryAgent(opt)
            self.decoder = None
            if opt['output'] == 'generate' or opt['output'] == 'g':
                self.decoder = Decoder(
                    opt['embedding_size'],
                    opt['embedding_size'],
                    opt['rnn_layers'],
                    opt,
                    self.dict,
                )
            elif opt['output'] != 'rank' and opt['output'] != 'r':
                raise NotImplementedError('Output type not supported.')

            if 'FP' in opt['setting']:
                # add extra beta-word to indicate learner's answer
                self.beta_word = 'betaword'
                self.dict.add_to_dict([self.beta_word])

            self.model = MemNN(opt, self.dict)

            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            lr = opt['learning_rate']
            if opt['optimizer'] == 'sgd':
                self.optimizers = {'memnn': optim.SGD(optim_params, lr=lr)}
                if self.decoder is not None:
                    self.optimizers['decoder'] = optim.SGD(
                        self.decoder.parameters(), lr=lr
                    )
            elif opt['optimizer'] == 'adam':
                self.optimizers = {'memnn': optim.Adam(optim_params, lr=lr)}
                if self.decoder is not None:
                    self.optimizers['decoder'] = optim.Adam(
                        self.decoder.parameters(), lr=lr
                    )
            else:
                raise NotImplementedError('Optimizer not supported.')

            if opt['cuda']:
                self.model.share_memory()
                if self.decoder is not None:
                    self.decoder.cuda()

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
                self.load(opt['model_file'])
        else:
            if 'model' in shared:
                # model is shared during hogwild
                self.model = shared['model']
                self.dict = shared['dict']
                self.decoder = shared['decoder']
                self.optimizers = shared['optimizer']
                if 'FP' in opt['setting']:
                    self.beta_word = shared['betaword']

        if hasattr(self, 'model'):
            self.opt = opt
            self.mem_size = opt['mem_size']
            self.loss_fn = CrossEntropyLoss()
            self.gradient_clip = opt.get('gradient_clip', 0.2)

            self.model_setting = opt['setting']
            if 'FP' in opt['setting']:
                self.feedback_cands = set([])
                self.num_feedback_cands = opt['num_feedback_cands']

            self.longest_label = 1
            self.END = self.dict.end_token
            self.END_TENSOR = torch.LongTensor(self.dict.parse(self.END))
            self.START = self.dict.start_token
            self.START_TENSOR = torch.LongTensor(self.dict.parse(self.START))

        self.reset()
        self.last_cands, self.last_cands_list = None, None

    def share(self):
        # Share internal states between parent and child instances
        shared = super().share()

        if self.opt.get('numthreads', 1) > 1:
            shared['model'] = self.model
            self.model.share_memory()
            shared['optimizer'] = self.optimizers
            shared['dict'] = self.dict
            shared['decoder'] = self.decoder
            if 'FP' in self.model_setting:
                shared['betaword'] = self.beta_word
        return shared

    def observe(self, observation):
        observation = copy.copy(observation)

        # extract feedback for forward prediction
        # IM setting - no feedback provided in the dataset
        if self.opt['setting'] != 'IM':
            if 'text' in observation:
                split = observation['text'].split('\n')
                feedback = split[-1]
                observation['feedback'] = feedback
                observation['text'] = '\n'.join(split[:-1])

        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = (
                self.observation['text'] if self.observation is not None else ''
            )

            # append answer and feedback (if available) given in the previous example to the previous dialog
            if 'eval_labels' in self.observation:
                prev_dialogue += '\n' + random.choice(self.observation['eval_labels'])
            elif 'labels' in self.observation:
                prev_dialogue += '\n' + random.choice(self.observation['labels'])
            if 'feedback' in self.observation:
                prev_dialogue += '\n' + self.observation['feedback']

            observation['text'] = prev_dialogue + '\n' + observation['text']

        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def reset(self):
        # reset observation and episode_done
        self.observation = None
        self.episode_done = True

    def backward(self, loss, retain_graph=False):
        # zero out optimizer and take one optimization step
        for o in self.optimizers.values():
            o.zero_grad()
        loss.backward(retain_graph=retain_graph)

        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clip)
        for o in self.optimizers.values():
            o.step()

    def parse_cands(self, cand_answers):
        """Returns:
            cand_answers = tensor (vector) of token indices for answer candidates
            cand_answers_lengths = tensor (vector) with lengths of each answer candidate
        """
        parsed_cands = [to_tensors(c, self.dict) for c in cand_answers]
        cand_answers_tensor = torch.cat([x[1] for x in parsed_cands])
        max_cands_len = max([len(c) for c in cand_answers])
        cand_answers_lengths = torch.LongTensor(
            len(cand_answers), max_cands_len
        ).zero_()
        for i in range(len(cand_answers)):
            if len(parsed_cands[i][0]) > 0:
                cand_answers_lengths[i, -len(parsed_cands[i][0]) :] = parsed_cands[i][0]
        cand_answers_tensor = Variable(cand_answers_tensor)
        cand_answers_lengths = Variable(cand_answers_lengths)
        return cand_answers_tensor, cand_answers_lengths

    def get_cand_embeddings_with_added_beta(self, cands, selected_answer_inds):
        # add beta_word to the candidate selected by the learner to indicate learner's answer
        cand_answers_with_beta = copy.deepcopy(cands)

        for i in range(len(cand_answers_with_beta)):
            cand_answers_with_beta[i][selected_answer_inds[i]] += ' ' + self.beta_word

        # get candidate embeddings after adding beta_word to the selected candidate
        (
            cand_answers_tensor_with_beta,
            cand_answers_lengths_with_beta,
        ) = self.parse_cands(cand_answers_with_beta)
        cands_embeddings_with_beta = self.model.answer_embedder(
            cand_answers_lengths_with_beta, cand_answers_tensor_with_beta
        )
        if self.opt['cuda']:
            cands_embeddings_with_beta = cands_embeddings_with_beta.cuda()
        return cands_embeddings_with_beta

    def predict(self, xs, answer_cands, ys=None, feedback_cands=None):
        is_training = ys is not None
        if is_training and 'FP' not in self.model_setting:
            # Subsample to reduce training time
            answer_cands = [
                list(set(random.sample(c, min(32, len(c))) + self.labels))
                for c in answer_cands
            ]
        else:
            # rank all cands to increase accuracy
            answer_cands = [list(set(c)) for c in answer_cands]

        self.model.train(mode=is_training)

        # Organize inputs for network (see contents of xs and ys in batchify method)
        inputs = [Variable(x, volatile=is_training) for x in xs]

        if self.decoder:
            output_embeddings = self.model(*inputs)
            self.decoder.train(mode=is_training)
            output_lines, loss = self.decode(output_embeddings, ys)
            predictions = self.generated_predictions(output_lines)
            self.backward(loss)
            return predictions

        scores = None
        if is_training:
            label_inds = [
                cand_list.index(self.labels[i])
                for i, cand_list in enumerate(answer_cands)
            ]

            if 'FP' in self.model_setting:
                if len(feedback_cands) == 0:
                    print(
                        'FP is not training... waiting for negative feedback examples'
                    )
                else:
                    cand_answers_embs_with_beta = self.get_cand_embeddings_with_added_beta(
                        answer_cands, label_inds
                    )
                    scores, forward_prediction_output = self.model(
                        *inputs, answer_cands, cand_answers_embs_with_beta
                    )
                    fp_scores = self.model.get_score(
                        feedback_cands, forward_prediction_output, forward_predict=True
                    )
                    feedback_label_inds = [
                        cand_list.index(self.feedback_labels[i])
                        for i, cand_list in enumerate(feedback_cands)
                    ]
                    if self.opt['cuda']:
                        feedback_label_inds = Variable(
                            torch.cuda.LongTensor(feedback_label_inds)
                        )
                    else:
                        feedback_label_inds = Variable(
                            torch.LongTensor(feedback_label_inds)
                        )
                    loss_fp = self.loss_fn(fp_scores, feedback_label_inds)
                    if loss_fp.data[0] > 100000:
                        raise Exception(
                            "Loss might be diverging. Loss:", loss_fp.data[0]
                        )
                    self.backward(loss_fp, retain_graph=True)

            if self.opt['cuda']:
                label_inds = Variable(torch.cuda.LongTensor(label_inds))
            else:
                label_inds = Variable(torch.LongTensor(label_inds))

        if scores is None:
            output_embeddings = self.model(*inputs)
            scores = self.model.get_score(answer_cands, output_embeddings)

        predictions = self.ranked_predictions(answer_cands, scores)

        if is_training:
            update_params = True
            # don't perform regular training if in FP mode
            if self.model_setting == 'FP':
                update_params = False
            elif 'RBI' in self.model_setting:
                if len(self.rewarded_examples_inds) == 0:
                    update_params = False
                else:
                    self.rewarded_examples_inds = torch.LongTensor(
                        self.rewarded_examples_inds
                    )
                    if self.opt['cuda']:
                        self.rewarded_examples_inds = self.rewarded_examples_inds.cuda()
                    # use only rewarded examples for training
                    loss = self.loss_fn(
                        scores[self.rewarded_examples_inds, :],
                        label_inds[self.rewarded_examples_inds],
                    )
            else:
                # regular IM training
                loss = self.loss_fn(scores, label_inds)

            if update_params:
                self.backward(loss)
        return predictions

    def ranked_predictions(self, cands, scores):
        _, inds = scores.data.sort(descending=True, dim=1)
        return [
            [cands[i][j] for j in r if j < len(cands[i])] for i, r in enumerate(inds)
        ]

    def decode(self, output_embeddings, ys=None):
        batchsize = output_embeddings.size(0)
        hn = output_embeddings.unsqueeze(0).expand(
            self.opt['rnn_layers'], batchsize, output_embeddings.size(1)
        )
        x = self.model.answer_embedder(
            Variable(torch.LongTensor([1])), Variable(self.START_TENSOR)
        )
        xes = x.unsqueeze(1).expand(x.size(0), batchsize, x.size(1))

        loss = 0
        output_lines = [[] for _ in range(batchsize)]
        done = [False for _ in range(batchsize)]
        total_done = 0
        idx = 0
        while (total_done < batchsize) and idx < self.longest_label:
            # keep producing tokens until we hit END or max length for each ex
            if self.opt['cuda']:
                xes = xes.cuda()
                hn = hn.contiguous()
            preds, scores = self.decoder(xes, hn)
            if ys is not None:
                y = Variable(ys[0][:, idx])
                temp_y = y.cuda() if self.opt['cuda'] else y
                loss += self.loss_fn(scores, temp_y)
            else:
                y = preds
            # use the true token as the next input for better training
            xes = self.model.answer_embedder(
                Variable(torch.LongTensor(preds.numel()).fill_(1)), y
            ).unsqueeze(0)

            for b in range(batchsize):
                if not done[b]:
                    token = self.dict.vec2txt(preds.data[b])
                    if token == self.END:
                        done[b] = True
                        total_done += 1
                    else:
                        output_lines[b].append(token)
            idx += 1
        return output_lines, loss

    def generated_predictions(self, output_lines):
        return [
            [' '.join(c for c in o if c != self.END and c != self.dict.null_token)]
            for o in output_lines
        ]

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
            cands = list of candidates for each example in batch
            valid_inds = list of indices for examples with valid observations
        """
        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]
        if not exs:
            return [None] * 5

        if 'RBI' in self.model_setting:
            self.rewarded_examples_inds = [
                i
                for i, ex in enumerate(obs)
                if 'text' in ex and ex.get('reward', 0) > 0
            ]

        parsed = [self.parse(ex['text']) for ex in exs]
        queries = torch.cat([x[0] for x in parsed])
        memories = torch.cat([x[1] for x in parsed])
        query_lengths = torch.cat([x[2] for x in parsed])
        memory_lengths = torch.LongTensor(len(exs), self.mem_size).zero_()
        for i in range(len(exs)):
            if len(parsed[i][3]) > 0:
                memory_lengths[i, -len(parsed[i][3]) :] = parsed[i][3]
        xs = [memories, queries, memory_lengths, query_lengths]

        ys = None
        self.labels = [random.choice(ex['labels']) for ex in exs if 'labels' in ex]

        if len(self.labels) == len(exs):
            parsed = [self.dict.txt2vec(l) for l in self.labels]
            parsed = [torch.LongTensor(p) for p in parsed]
            label_lengths = torch.LongTensor([len(p) for p in parsed]).unsqueeze(1)
            self.longest_label = max(self.longest_label, label_lengths.max())
            padded = [
                torch.cat(
                    (
                        p,
                        torch.LongTensor(self.longest_label - len(p)).fill_(
                            self.END_TENSOR[0]
                        ),
                    )
                )
                for p in parsed
            ]
            labels = torch.stack(padded)
            ys = [labels, label_lengths]

        feedback_cands = []
        if 'FP' in self.model_setting:
            self.feedback_labels = [
                ex['feedback']
                for ex in exs
                if 'feedback' in ex and ex['feedback'] is not None
            ]
            self.feedback_cands = self.feedback_cands | set(self.feedback_labels)

            if (
                len(self.feedback_labels) == len(exs)
                and len(self.feedback_cands) > self.num_feedback_cands
            ):
                feedback_cands = [
                    list(
                        set(
                            random.sample(self.feedback_cands, self.num_feedback_cands)
                            + [feedback]
                        )
                    )
                    for feedback in self.feedback_labels
                ]

        cands = [ex['label_candidates'] for ex in exs if 'label_candidates' in ex]
        # Use words in dict as candidates if no candidates are provided
        if len(cands) < len(exs):
            cands = build_cands(exs, self.dict)
        # Avoid rebuilding candidate list every batch if its the same
        if self.last_cands != cands:
            self.last_cands = cands
            self.last_cands_list = [list(c) for c in cands]
        cands = self.last_cands_list
        return xs, ys, cands, valid_inds, feedback_cands

    def batch_act(self, observations):
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        xs, ys, cands, valid_inds, feedback_cands = self.batchify(observations)

        if xs is None or len(xs[1]) == 0:
            return batch_reply

        # Either train or predict
        predictions = self.predict(xs, cands, ys, feedback_cands)

        for i in range(len(valid_inds)):
            batch_reply[valid_inds[i]]['text'] = predictions[i][0]
            batch_reply[valid_inds[i]]['text_candidates'] = predictions[i]
        return batch_reply

    def act(self):
        return self.batch_act([self.observation])[0]

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
        with open(path, 'rb') as read:
            checkpoint = torch.load(read)
        self.model.load_state_dict(checkpoint['memnn'])
        self.optimizers['memnn'].load_state_dict(checkpoint['memnn_optim'])
        if self.decoder is not None:
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.optimizers['decoder'].load_state_dict(checkpoint['decoder_optim'])
            self.longest_label = checkpoint['longest_label']


def build_cands(exs, dict):
    dict_list = list(dict.tok2ind.keys())
    cands = []
    for ex in exs:
        if 'label_candidates' in ex:
            cands.append(ex['label_candidates'])
        else:
            cands.append(dict_list)
            if 'labels' in ex:
                cands[-1] += [l for l in ex['labels'] if l not in dict.tok2ind]
    return cands
