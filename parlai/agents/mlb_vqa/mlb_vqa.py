#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent

import torch.nn as nn
import torch
import os
from collections import defaultdict, Counter
import numpy as np
import re

from .mlb_modules import MlbAtt, MlbNoAtt

'''
    An implementation of the Multimodal Low-rank Bilinear Attention Network
    outlined in https://arxiv.org/abs/1610.04325, that can be used with the
    VQA V1 and VQA V2 datasets.

    The model currently only supports `image_mode = resnet152_spatial`,
    `image_size = 448`, and `image_cropsize = 448`

    While this model can be used normally with the VQA task, it is best to
    make use of the `PytorchDataTeacher` for improved image feature loading,
    which substantially improves speed of training.

    To train the model using the `PytorchDataTeacher` on VQA V1, use the
    following command:

        `python examples/train_model.py -m mlb_vqa -pytd vqa_v1  \
        -mf <model_file> -bs <batchsize> \
        -im resnet152_spatial --image-size 448 --image-cropsize 448`

    Where you fill in `<model_file>` and `<batchsize>` with
    your own values; e.g.:

        `python examples/train_model.py -m mlb_vqa -pytd vqa_v1 -mf mlb \
        -bs 512 -im resnet152_spatial --image-size 448 --image-cropsize 448`

    This will also download and extract the image features on the fly.

    If you would like to extract the image features prior to training, run the
    following command (where `-dt` can be either 'train', 'valid', or 'test'):
        `python examples/extract_image_feature.py -pytd vqa_v1\
        -im resnet152_spatial --image-size 448 --image-cropsize 448 \
        -dt <datatype>`

    For faster training, specify '--no-metrics,' which prevents computation
    of f1 score and accuracy

    Finally, it should be noted that the default attention strategy in this
    implementation uses less parameters than the original model (though it
    has performed better) - to use the original attention method, specify
    '--original_att' on the command line.
'''


def escape(s):
    """Replace potential special characters with escaped version.
    For example, newline => \\n and tab => \\t
    """
    return s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')


def unescape(s):
    """Revert escaped characters back to their special version.
    For example, \\n => newline and \\t => tab
    """
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')


class VqaDictionaryAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        dictionary = argparser.add_argument_group('Dictionary Arguments')
        dictionary.add_argument(
            '--dict-file',
            help='if set, the dictionary will automatically save to this path' +
                 ' during shutdown')
        dictionary.add_argument(
            '--dict-initpath',
            help='path to a saved dictionary to load tokens / counts from to ' +
                 'seed the dictionary with initial tokens and/or frequencies')
        dictionary.add_argument(
            '--dict-maxexs', default=300000, type=int,
            help='max number of examples to build dict on')
        dictionary.add_argument('-smp', '--samplingans', type='bool', default=True)
        dictionary.add_argument('--nans', type=int, default=2000)
        dictionary.add_argument('--maxlength', type=int, default=16)
        dictionary.add_argument('--minwcount', type=int, default=0)
        dictionary.add_argument('--nlp', default='mcb')

    def __init__(self, opt, shared=None):
        super(VqaDictionaryAgent, self).__init__(opt)
        self.id = 'VqaDictionary'
        self.null_token = '__NULL__'
        self.unk_token = '__UNK__'

        if shared:
            self.freq = shared.get('freq', {})
            self.tok2ind = shared.get('tok2ind', {})
            self.ind2tok = shared.get('ind2tok', {})
            self.ans2ind = shared.get('ans2ind', {})
            self.ind2ans = shared.get('ind2ans', {})
        else:
            self.freq = defaultdict(int)
            self.ansfreq = defaultdict(int)
            self.ans2ques = defaultdict(list)
            self.tok2ind = {}
            self.ind2tok = {}
            self.ans2ind = {}
            self.ind2ans = {}

            if self.null_token:
                self.tok2ind[self.null_token] = 0
                self.ind2tok[0] = self.null_token

            if self.unk_token:
                # set special unknown word token
                index = len(self.tok2ind)
                self.tok2ind[self.unk_token] = index
                self.ind2tok[index] = self.unk_token

        if opt.get('dict_file') and os.path.isfile(opt['dict_file']):
            # load pre-existing dictionary
            self.load(opt['dict_file'])

        if not shared:

            if self.null_token:
                # fix count for null token to one billion and two
                self.freq[self.null_token] = 1000000002

            if self.unk_token:
                # fix count for unknown token to one billion
                self.freq[self.unk_token] = 1000000000

            if opt.get('dict_file'):
                self.save_path = opt['dict_file']

    def __len__(self):
        return len(self.tok2ind)

    def add_to_ques_dict(self, tokens):
        """Builds dictionary from the list of provided tokens.
        Only adds words contained in self.embedding_words, if not None.
        """
        for token in tokens:
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

    def add_to_ans_dict(self, token):
        """Builds dictionary from the list of provided tokens.
        Only adds words contained in self.embedding_words, if not None.
        """
        self.ansfreq[token] += 1
        if token not in self.ans2ind:
            index = len(self.ans2ind)
            self.ans2ind[token] = index
            self.ind2ans[index] = token

    def tokenize_mcb(self, s):
        t_str = s.lower()
        for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(',
                  r'\)', r'\,', r'\.', r'\;']:
            t_str = re.sub(i, '', t_str)
        for i in [r'\-', r'\/']:
            t_str = re.sub(i, ' ', t_str)
        q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
        q_list = list(filter(lambda x: len(x) > 0, q_list))
        return q_list

    def split_tokenize(self, s):
        return (s.lower().replace('.', ' . ').replace('. . .', '...')
                .replace(',', ' , ').replace(';', ' ; ').replace(':', ' : ')
                .replace('!', ' ! ').replace('?', ' ? ')
                .split())

    def act(self):
        """Add any words passed in the 'text' field of the observation to this
        dictionary.
        """
        mc_label = self.observation.get('mc_label', self.observation.get('labels', []))
        for text in mc_label:
            self.ansfreq[text] += 1
            self.ans2ques[text].append(self.tokenize_mcb(self.observation.get('text')))
        return {'id': 'Dictionary'}

    def encode_question(self, examples, training):
        minwcount = self.opt.get('minwcount', 0)
        maxlength = self.opt.get('maxlength', 16)
        for ex in examples:
            words = self.tokenize_mcb(ex['text'])
            if training:
                words_unk = [
                    w if self.freq.get(w, 0) > minwcount else self.unk_token
                    for w in words
                ]
            else:
                words_unk = [w if w in self.tok2ind else self.unk_token for w in words]
            ex['question_wids'] = [self.tok2ind[self.null_token]] * maxlength
            for k, w in enumerate(words_unk):
                if k < maxlength:
                    ex['question_wids'][k] = self.tok2ind[w]
        return examples

    def encode_answer(self, examples):
        for ex in examples:
            if self.opt.get('samplingans', True):
                labels = ex.get('labels', ex.get('eval_labels'))
                ans_count = Counter(labels).most_common()
                valid_ans = []
                valid_count = []
                for ans in ans_count:
                    if ans[0] in self.ans2ind:
                        valid_ans.append(self.ans2ind[ans[0]])
                        valid_count.append(ans[1])
                if not valid_ans:
                    ex['answer_aid'] = 0
                else:
                    probs = valid_count / np.sum(valid_count)
                    ex['answer_aid'] = int(np.random.choice(valid_ans, p=probs))
            else:
                ex['answer_aid'] = self.ans2ind[ex['mc_label'][0]]
        return examples

    def decode_answer(self, examples):
        txt_answers = []
        for ex in examples:
            txt_answers.append(self.ind2ans[ex])
            # print("Predicted output ex:", i, ex)
        return txt_answers

    def load(self, filename):
        """Load pre-existing dictionary in 'token[<TAB>count]' format.
        Initialize counts from other dictionary, or 0 if they aren't included.
        """
        print('Dictionary: loading dictionary from {}'.format(filename))
        with open(filename) as read:
            for line in read:
                split = line.strip().split('\t')
                token = unescape(split[0])
                cnt = int(split[1]) if len(split) > 1 else 0
                self.freq[token] = cnt
                if token not in self.tok2ind:
                    index = len(self.tok2ind)
                    self.tok2ind[token] = index
                    self.ind2tok[index] = token
        print('[ num ques words =  %d ]' % len(self.ind2tok))

        with open(filename[:-5] + "_ans.dict") as read:
            for line in read:
                split = line.strip().split('\t')
                token = unescape(split[0])
                cnt = int(split[1]) if len(split) > 1 else 0
                self.ansfreq[token] = cnt
                if token not in self.ans2ind:
                    index = len(self.ans2ind)
                    self.ans2ind[token] = index
                    self.ind2ans[index] = token

        print('[ num ans words =  %d ]' % len(self.ind2ans))

    def save(self, filename=None, append=False, sort=True):
        """Save dictionary to file.
        Format is 'token<TAB>count' for every token in the dictionary, sorted
        by count with the most frequent words first.

        If ``append`` (default ``False``) is set to ``True``, appends instead
        of overwriting.

        If ``sort`` (default ``True``), then first sort the dictionary before
        saving.
        """
        cw = sorted([(count, w) for w, count in self.ansfreq.items()], reverse=True)
        final_exs = cw[:self.opt.get('nans', 2000)]
        final_list = dict([(w, c) for c, w in final_exs])
        self.ansfreq = defaultdict(int)
        for ans, ques in self.ans2ques.items():
            if ans in final_list:
                for que in ques:
                    self.add_to_ques_dict(que)
                self.add_to_ans_dict(ans)

        filename = self.opt['dict_file'] if filename is None else filename
        print('Dictionary: saving dictionary to {}'.format(filename))
        # if sort:
        #     self.sort()

        with open(filename, 'a' if append else 'w') as write:
            for i in range(len(self.ind2tok)):
                tok = self.ind2tok[i]
                cnt = self.freq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))

        with open(filename[:-5] + "_ans.dict", 'a' if append else 'w') as write:
            for i in range(len(self.ind2ans)):
                tok = self.ind2ans[i]
                cnt = self.ansfreq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))

    def shutdown(self):
        """Save on shutdown if ``save_path`` is set."""
        if hasattr(self, 'save_path'):
            self.save(self.save_path)


class MlbVqaAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Mlb Arguments')
        agent.add_argument('--dim_q', type=int, default=2400)
        agent.add_argument('--dim_v', type=int, default=2048)
        agent.add_argument('--dim_h', type=int, default=1200)
        agent.add_argument('--dim_att_h', type=int, default=1200)
        agent.add_argument('--dropout_st', type=float, default=0.25)
        agent.add_argument('--dropout_v', type=float, default=0.5)
        agent.add_argument('--dropout_q', type=float, default=0.5)
        agent.add_argument('--dropout_cls', type=float, default=0.5)
        agent.add_argument('--dropout_att_v', type=float, default=0.5)
        agent.add_argument('--dropout_att_q', type=float, default=0.5)
        agent.add_argument('--dropout_att_mm', type=float, default=0.5)
        agent.add_argument('--activation_att_v', default='tanh')
        agent.add_argument('--activation_att_q', default='tanh')
        agent.add_argument('--activation_att_mm', default='tanh')
        agent.add_argument('--activation_v', default='tanh')
        agent.add_argument('--activation_q', default='tanh')
        agent.add_argument('--activation_cls', default='tanh')
        agent.add_argument('-at', '--attention', action='store_true')
        agent.add_argument('--use-bayesian', type='bool', default=True)
        agent.add_argument('--num_glimpses', type=int, default=4)
        agent.add_argument('--original_att', action='store_true')
        agent.add_argument('--lr', type=float, default=0.0001)
        agent.add_argument('--no-cuda', action='store_true',
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=0,
                           help='which GPU device to use')
        agent.add_argument('--no-data-parallel', action='store_true',
                           help='disable pytorch parallel data processing')
        agent.add_argument('--use-hdf5', type='bool', default=False,
                           help='specify whether to use a single hdf5 file to load \
                           images')
        agent.add_argument('--no-metrics', action='store_true',
                           help='specify to not compute f1 or accuracy during \
                           training (speeds up training)')
        MlbVqaAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return VqaDictionaryAgent

    def __init__(self, opt, shared=None):
        super(MlbVqaAgent, self).__init__(opt, shared)
        if not shared:
            # check for cuda
            self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
            self.use_data_parallel = not opt.get('no_data_parallel', False)
            self.compute_metrics = not opt.get('no_metrics', False)

            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            states = None
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                states = self.load(opt['model_file'])
                # override options with stored ones
                self.opt = self.override_opt(states['opt'])

            self.id = 'Mlb'

            self.dict = MlbVqaAgent.dictionary_class()(opt)
            self.vocab = len(self.dict.tok2ind)
            self.num_classes = len(self.dict.ans2ind)
            self.training = self.opt.get('datatype').startswith('train')
            self.testing = self.opt.get('datatype').startswith('test')
            self.batchsize = self.opt.get('batchsize', 1)
            if self.opt['attention']:
                self.model = MlbAtt(self.opt, self.dict, states)
            else:
                self.model = MlbNoAtt(self.opt, self.dict, states)

            self.criterion = nn.CrossEntropyLoss()
            if self.use_cuda:
                if self.use_data_parallel:
                    self.model = nn.DataParallel(self.model).cuda()
                else:
                    self.model.cuda()
                self.criterion.cuda()
            if self.use_cuda and self.use_data_parallel:
                self.optim = self.model.module.get_optim()
            else:
                self.optim = self.model.get_optim()

        self.reset()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def observe(self, observation):
        """Save observation for act."""
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    @staticmethod
    def static_vis_noatt(img_feat, use_att, use_hdf5=False):
        if use_att or use_hdf5:
            return img_feat
        nb_regions = img_feat.size(2) * img_feat.size(3)
        img_feat = img_feat.sum(3).sum(2).div(nb_regions).view(-1, 2048)
        return img_feat

    @staticmethod
    def collate(batch):
        # Get appropriate dims
        first_ex = batch[0][1][0]

        # If we are building the dictionary
        if 'image' not in first_ex or first_ex['image'] is None:
            new_batch = []
            for b in batch:
                if type(b[1]) is list:
                    ep = b[1][0]
                else:
                    ep = b[1]
                new_batch.append(ep)
            return new_batch
        img_var = torch.FloatTensor(first_ex['image'])
        use_att = first_ex['use_att']
        use_hdf5 = first_ex['use_hdf5']
        if use_hdf5:
            dim_v = img_var.size(0)
            if use_att:
                height = img_var.size(1)
                width = img_var.size(2)
        else:
            dim_v = img_var.size(1)
            height = img_var.size(2)
            width = img_var.size(3)

        max_len = len(first_ex['question_wids'])

        # Everything else
        batchsize = len(batch)
        if use_hdf5 and not use_att:
            '''As we are using hdf5 dataset,
               the input is already in noatt form'''
            input_v = torch.FloatTensor(batchsize, dim_v).fill_(0)
        else:
            input_v = torch.FloatTensor(batchsize, dim_v, height, width).fill_(0)
        input_q = torch.LongTensor(batchsize, max_len).fill_(0)
        answer = torch.LongTensor(batchsize).fill_(0)
        valid_inds = []
        labels = []
        ep_dones = []
        testing = True
        for i, (_, ex) in enumerate(batch):
            ex = ex[0]
            input_v[i] = ex['image']
            input_q[i] = torch.LongTensor(ex['question_wids'])
            if 'answer_aid' in ex:
                answer[i] = ex['answer_aid']
                testing = False
            if 'valid' in ex and ex['valid']:
                valid_inds.append(i)
            if 'labels' in ex:
                labels.append(ex['labels'])
            else:
                labels.append(None)
            ep_dones.append(ex['episode_done'])

        data = {
            'input_v': MlbVqaAgent.static_vis_noatt(
                input_v, use_att, use_hdf5=use_hdf5
            ),
            'input_q': input_q,
            'valid_inds': valid_inds,
            'batchsize': batchsize,
            'labels': labels[0],
            'episode_done': ep_dones[0],
            'preprocessed': True
        }
        if not testing:
            data['answer'] = answer
        return [
            data
        ] + [
            {
                'labels': ex_label,
                'episode_done': ep_done,
                'preprocessed': True
            }
            for ex_label, ep_done in zip(labels[1:], ep_dones[1:])
        ]

    def process_obs(self, observations):
        if any('text' not in ex for ex in observations):
            observations = [ex for ex in observations if 'text' in ex]
        if not observations:
            return None
        new_obs = []
        valid_inds = []
        mc = False
        for i, ex in enumerate(observations):
            if self.use_cuda:
                ex['image'] = ex['image'].cuda()
            if 'mc_label' in ex:
                self.training = True
                if ex['mc_label'][0] in self.dict.ans2ind:
                    mc = True
                    new_obs.append(ex)
                    valid_inds.append(i)

        if not self.training or not mc:
            new_obs = observations.copy()
            valid_inds = range(len(new_obs))

        if not self.testing:
            new_obs = self.dict.encode_question(new_obs, self.training)
            if self.training:
                new_obs = self.dict.encode_answer(new_obs)
        else:
            new_obs = self.dict.encode_question(new_obs, False)
            answer = None

        input_v = torch.stack([ex['image'][0] for ex in new_obs])
        input_q = torch.stack([torch.LongTensor(ex['question_wids']) for ex in new_obs])
        if not self.testing:
            answer = torch.LongTensor([ex['answer_aid'] for ex in new_obs])

        return {
            'input_v': MlbVqaAgent.static_vis_noatt(input_v, self.opt['attention']),
            'input_q': input_q,
            'answer': answer,
            'valid_inds': valid_inds,
        }

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        if observations[0].get('preprocessed'):
            data = observations[0]
        else:
            data = self.process_obs(observations)
        if data is None:
            return None, None, None, None

        input_v = data['input_v']
        input_q = data['input_q']
        answer = data.get('answer', None)
        if answer is None:
            self.testing = True
            self.training = False
        valid_inds = data['valid_inds']

        if self.use_cuda:
            if not self.use_data_parallel:
                input_v = input_v.cuda()
                input_q = input_q.cuda()
            if not self.testing:
                answer = answer.cuda()

        return input_v, input_q, answer, valid_inds

    def predict(self, visual, text, label):
        out = self.model(visual, text)
        loss = None
        if label is not None:
            torch.cuda.synchronize()
            loss = self.criterion(out, label)
            self.optim.zero_grad()
            loss.backward()
            torch.cuda.synchronize()
            self.optim.step()
            torch.cuda.synchronize()

        return loss, out

    def batch_act(self, observations):
        # initialize a table of replies with this agent's id
        input_v, input_q, answer, valid_inds = self.batchify(observations)

        batch_reply = [{'id': self.getID()} for _ in range(self.batchsize)]
        if input_v is None:
            return batch_reply

        loss, predictions = self.predict(input_v, input_q, answer)
        if loss is not None:
            batch_reply[0]['metrics'] = {'loss': loss.item()}
        if not self.training or self.compute_metrics:
            _, predictions = predictions.max(1)
            if predictions.size(0) > 1:
                predictions.squeeze_(0)
            tpreds = self.dict.decode_answer(predictions.tolist())
            for i in range(len(tpreds)):
                # map the predictions back to non-empty examples in the batch
                curr = batch_reply[valid_inds[i]]
                curr['text'] = tpreds[i]

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'dim_v', 'dim_q', 'dim_h', 'dim_att_h',
                      'dropout_cls', 'dropout_st', 'dropout_q',
                      'dropout_v', 'dropout_att_mm',
                      'dropout_att_q', 'dropout_att_v',
                      'activation_cls', 'activation_q',
                      'activation_v', 'activation_att_mm',
                      'activation_att_q', 'activation_att_v',
                      'num_glimpses', 'use_bayesian', 'attention'}
        for k, v in new_opt.items():
            if k not in model_args:
                # skip non-model args
                continue
            if k not in self.opt:
                print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('Overriding option [ {k}: {old} => {v}]'.format(
                      k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        return self.opt

    def save(self, path=None):
        if hasattr(self, 'model'):
            if self.use_cuda and self.use_data_parallel:
                self.model.module.save(path)
            else:
                self.model.save(path)

    def load(self, path):
        """Return opt and model states."""
        model = torch.load(path, map_location=lambda cpu, _: cpu)
        return model

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()
