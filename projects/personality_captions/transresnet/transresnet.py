#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent code for the model described in (https://arxiv.org/abs/1811.00945).
"""

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from .modules import TransresnetModel
from parlai.tasks.personality_captions.build import build


import os
import random
import json
import numpy as np
import torch
import tqdm


class TransresnetAgent(Agent):
    """
    Model described in (https://arxiv.org/abs/1811.00945).

    A model for producing engaging captions about an image. Given an image and
    this model will attempt to predict an appropriate
    next utterance in the dialog, in the context of a given personality.

    See the paper linked above for more information.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add command line args.
        """
        arg_group = argparser.add_argument_group('Transresnet Arguments')
        TransresnetModel.add_cmdline_args(argparser)
        argparser.add_argument(
            '--freeze-patience',
            type=int,
            default=-1,
            help='How long to freeze text encoders',
        )
        argparser.add_argument(
            '--one-cand-set',
            type='bool',
            default=False,
            help='True if each example has one set of shared ' 'label candidates',
        )
        argparser.add_argument(
            '--fixed-cands-path',
            type=str,
            default=None,
            help='path to text file with candidates',
        )
        argparser.add_argument(
            '--pretrained', type='bool', default=False, help='True if pretrained model'
        )
        DictionaryAgent.add_cmdline_args(argparser)
        return arg_group

    def __init__(self, opt, shared=None):
        if opt.get('numthreads', 1) > 1:
            raise RuntimeError(
                'Warning: You cannot use multithreading with '
                'this agent, as the current metrics do not '
                'support sharing of lists (for median rank '
                'calculation). Please set --numthreads to 1'
            )
        self.metrics = {
            'hits@1/100': 0.0,
            'loss': 0.0,
            'num_samples': 0,
            'med_rank': [],
        }
        self.blank_image_features = torch.FloatTensor(
            opt.get('image_features_dim')
        ).fill_(0)
        self.opt = opt
        self.model_file = opt['model_file']
        self.id = 'TransresnetAgent'
        self.one_cand_set = opt.get('one_cand_set', False)
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        self.fcp = None
        if opt.get('fixed_cands_path') is not None:
            self.fcp = opt['fixed_cands_path']
        self.episode_done = True

        if not shared:
            # setup dict
            self._setup_dict()
            # load the list of personalities
            self.personalities_list = self.load_personalities()
            # possibly load the model from a model file
            self._build_model()
            # load candidates if specified
            self._setup_cands()
            self.freeze_patience = self.opt['freeze_patience']
            if self.freeze_patience != -1:
                # For fine-tuning
                self.model.freeze_text_encoder()
                self.freeze_impatience = 0
                self.freeze_best_metric = 0
                self.is_frozen = True
        else:
            self.dict = shared['dict']
            self.model = shared['model']
            self.personalities_list = shared['personalities_list']
            self.fixed_cands = shared['fixed_cands']
            self.fixed_cands_enc = shared['fixed_cands_enc']

        super().__init__(opt, shared)

    def share(self):
        """
        Share appropriate attributes.
        """
        shared = super().share()
        shared['dict'] = self.dict
        shared['model'] = self.model
        shared['personalities_list'] = self.personalities_list
        shared['fixed_cands'] = self.fixed_cands
        shared['fixed_cands_enc'] = self.fixed_cands_enc
        return shared

    def _build_model(self, path=None):
        init_model_path = None
        if self.opt.get('init_model') and os.path.isfile(self.opt['init_model']):
            init_model_path = self.opt['init_model']
        elif self.opt.get('model_file') and os.path.isfile(self.opt['model_file']):
            init_model_path = self.opt['model_file']
        elif path is not None:
            init_model_path = path
        print('Creating or loading model')
        self.model = TransresnetModel(self.opt, self.personalities_list, self.dict)
        if init_model_path is not None:
            self.load(init_model_path)
        if self.use_cuda:
            self.model.cuda()

    def _setup_cands(self):
        self.fixed_cands = None
        self.fixed_cands_enc = None
        if self.fcp is not None:
            with open(self.fcp) as f:
                self.fixed_cands = [c.replace('\n', '') for c in f.readlines()]
            cands_enc_file = '{}.cands_enc'.format(self.fcp)
            print('loading saved cand encodings')
            if os.path.isfile(cands_enc_file):
                self.fixed_cands_enc = torch.load(
                    cands_enc_file, map_location=lambda cpu, _: cpu
                )
            else:
                print('Extracting cand encodings')
                self.model.eval()
                pbar = tqdm.tqdm(
                    total=len(self.fixed_cands),
                    unit='cand',
                    unit_scale=True,
                    desc='Extracting candidate encodings',
                )
                fixed_cands_enc = []
                for _, batch in enumerate(
                    [
                        self.fixed_cands[i : i + 50]
                        for i in range(0, len(self.fixed_cands) - 50, 50)
                    ]
                ):
                    embedding = self.model(None, None, batch)[1].detach()
                    fixed_cands_enc.append(embedding)
                    pbar.update(50)
                self.fixed_cands_enc = torch.cat(fixed_cands_enc, 0)
                torch.save(self.fixed_cands_enc, cands_enc_file)

    def load_personalities(self):
        """
        Load and return the list of personalities.
        """
        personality_path = os.path.join(
            self.opt['datapath'], 'personality_captions/personalities.txt'
        )
        if 'yfcc_path' not in self.opt:
            self.opt['yfcc_path'] = 'temp_path'
        build(self.opt)
        del self.opt['yfcc_path']
        perss = []
        with open(personality_path) as f:
            for line in f:
                if 'Trait' not in line:
                    perss.append(line[0:-1])
        return perss

    def observe(self, observation):
        """
        Observe.
        """
        self.observation = observation
        return observation

    def act(self):
        """
        Act.
        """
        return self.batch_act([self.observation])[0]

    def train_step(self, valid_obs, image_feats, personalities):
        """
        Model train step.

        :param valid_obs:
            list of valid observations

        :param image_feats:
            list of image features, one per example

        :param personalities:
            list of personalities, one per example

        :return:
            the total loss, number of correct examples, and total number of
            examples evaluated
        """
        comments = [random.choice(v['labels']) for v in valid_obs]
        loss, num_correct, num_examples = self.model.train_batch(
            image_feats, personalities, comments
        )
        return loss, num_correct, num_examples

    def eval_step(self, valid_obs, image_feats, personalities):
        """
        Model eval step.

        :param valid_obs:
            list of valid observations

        :param image_feats:
            list of image features, one per example

        :param personalities:
            list of personalities, one per example

        :return:
            the total loss, number of correct examples, the total number of
            examples evaluated, the ranked position of each correct caption,
            and the ranked lists of candidates (one per example)
        """
        med_rank = None
        chosen_captions = None
        if 'label_candidates' in valid_obs[0] or self.fixed_cands is not None:
            # User provides candidates, used as negatives for evaluation
            candidates_encoded = None
            if self.fixed_cands is not None:
                candidates_encoded = self.fixed_cands_enc
                candidates = self.fixed_cands
            else:
                candidates = [v['label_candidates'] for v in valid_obs]
                if self.one_cand_set:
                    candidates_encoded = self.model(None, None, candidates[0])[
                        1
                    ].detach()
            chosen_captions = self.model.choose_best_caption(
                image_feats,
                personalities,
                candidates,
                candidates_encoded=candidates_encoded,
                k=-1 if self.fixed_cands is None else 100,
            )
            # calculate median ranks
            num_examples = len(chosen_captions)
            loss = -1
            if self.fixed_cands is not None:
                num_correct = 0
            else:
                comments = [v['eval_labels'] for v in valid_obs]
                med_rank = []
                for i, c_list in enumerate(chosen_captions):
                    lowest_rank = len(c_list) + 1
                    for _, c in enumerate(comments[i]):
                        lowest_rank = min(lowest_rank, c_list.index(c) + 1)
                    med_rank.append(lowest_rank)
                num_correct = sum(
                    [
                        1 if chosen_captions[i][0] in chosen_captions[i] else 0
                        for i in range(len(chosen_captions))
                    ]
                )
        else:
            comments = [random.choice(v['eval_labels']) for v in valid_obs]
            loss, num_correct, num_examples = self.model.eval_batch(
                image_feats, personalities, comments
            )

        return loss, num_correct, num_examples, med_rank, chosen_captions

    def batch_act(self, observations):
        """
        Act on a batch of observations.

        :param observations:
            list of observations

        :return:
            A list of acts, one for each observation
        """
        is_training = any(['labels' in obs for obs in observations])
        valid_obs, valid_indexes = self.filter_valid_obs(observations, is_training)
        image_feats = self.extract_image_feats(valid_obs)
        personalities = [v.get('text', '') for v in valid_obs]

        chosen_captions = None
        med_rank = None
        if is_training:
            loss, num_correct, num_examples = self.train_step(
                valid_obs, image_feats, personalities
            )
        else:
            loss, num_correct, num_examples, med_rank, chosen_captions = self.eval_step(
                valid_obs, image_feats, personalities
            )

        self.update_metrics(loss, num_correct, num_examples, med_rank)
        result = [
            {'text': 'No Response During Training'} for _ in range(len(observations))
        ]
        if chosen_captions is not None:
            for i, index_obs in enumerate(valid_indexes):
                result[index_obs]['text'] = chosen_captions[i][0]
                result[index_obs]['text_candidates'] = chosen_captions[i]
        return result

    def extract_image_feats(self, obs):
        """
        Extract image features from the observations.

        :param obs:
            list of observations

        :return:
            list of image features
        """
        tmp_image_feats = [v.get('image') for v in obs]
        for i, im in enumerate(tmp_image_feats):
            try:
                # Check if given img features of form [1, <dim>, 1, 1]
                if len(im.size()) == 4:
                    tmp_image_feats[i] = im[0, :, 0, 0]
            except TypeError:  # No Image Feats Given
                tmp_image_feats[i] = self.blank_image_features
        image_feats = []
        for img in tmp_image_feats:
            image_feats.append(img.detach())
        return image_feats

    def filter_valid_obs(self, observations, is_training):
        """
        Filter out invalid observations.
        """
        label_key = 'labels' if is_training else 'eval_labels'
        valid_obs = []
        valid_indexes = []
        seen_texts = set()
        for i in range(len(observations)):
            if 'image' in observations[i]:
                if self.fixed_cands is not None:
                    valid_obs.append(observations[i])
                    valid_indexes.append(i)
                else:
                    text = observations[i][label_key][0]
                    if text not in seen_texts:
                        seen_texts.add(text)
                        valid_obs.append(observations[i])
                        valid_indexes.append(i)
        return valid_obs, valid_indexes

    def update_metrics(self, loss, num_correct, num_samples, med_rank=None):
        """
        Update Metrics.

        :param loss:
            float loss
        :param num_correct:
            number of examples for which chosen caption is correct
        :param num_samples:
            total number of examples
        :param med_rank:
            rank of correct caption for each example
        """
        self.metrics['hits@1/100'] += num_correct
        self.metrics['loss'] += loss
        self.metrics['num_samples'] += num_samples
        if med_rank:
            self.metrics['med_rank'] += med_rank

    def _setup_dict(self):
        """
        Set up the dictionary.

        The pretrained model used a separate dictionary from the standard ParlAI one.
        """
        self.dict = DictionaryAgent(self.opt)
        if self.opt.get('pretrained', False):
            new_tok2ind = {}
            new_ind2tok = {}
            for key in self.dict.tok2ind:
                val = self.dict.tok2ind[key]
                if val - 4 >= 0:
                    new_tok2ind[key] = val - 4
                    new_ind2tok[val - 4] = key
            self.dict.null_token = '<PAD>'
            self.dict.unk_token = '<UNK>'
            self.dict.tok2ind = new_tok2ind
            self.dict.ind2tok = new_ind2tok

    def receive_metrics(self, metrics_dict):
        """
        Receive the metrics from validation.

        Unfreeze text encoder weights after a certain number of rounds without improvement.

        :param metrics_dict:
            the metrics dictionary
        """
        if 'tasks' in metrics_dict:
            metrics_dict = metrics_dict['tasks']['personality_captions']
        if self.freeze_patience != -1 and self.is_frozen:
            m = metrics_dict['hits@1/100']
            if m > self.freeze_best_metric:
                self.freeze_impatience = 0
                self.freeze_best_metric = m
                print('performance not good enough to unfreeze the model.')
            else:
                self.freeze_impatience += 1
                print('Growing impatience for unfreezing')
                if self.freeze_impatience >= self.freeze_patience:
                    self.is_frozen = False
                    print(
                        'Reached impatience for fine tuning. '
                        'Reloading the best model so far.'
                    )
                    self._build_model(self.model_file)
                    if self.use_cuda:
                        self.model = self.model.cuda()
                    print('Unfreezing.')
                    self.model.unfreeze_text_encoder()
                    print('Done')

    def reset(self):
        """
        Reset metrics.
        """
        super().reset()
        self.reset_metrics()

    def reset_metrics(self):
        """
        Reset the metrics.
        """
        self.metrics['hits@1/100'] = 0.0
        self.metrics['loss'] = 0.0
        self.metrics['num_samples'] = 0.0
        if 'med_rank' in self.metrics:
            self.metrics['med_rank'] = []

    def report(self):
        """
        Report the current metrics.

        :return:
            a metrics dict
        """
        m = {}
        if self.metrics['num_samples'] > 0:
            m['hits@1/100'] = round_sigfigs(
                self.metrics['hits@1/100'] / self.metrics['num_samples'], 4
            )
            m['loss'] = round_sigfigs(
                self.metrics['loss'] / self.metrics['num_samples'], 4
            )
            if 'med_rank' in self.metrics:
                m['med_rank'] = np.median(self.metrics['med_rank'])
        return m

    def save(self, path=None):
        """
        Save the model.

        :param path:
            path for saving model
        """
        path = self.opt.get('model_file', None) if path is None else path
        self.dict.save(path + '.dict', sort=False)
        print('Saving best model')
        states = {}
        states['model'] = self.model.state_dict()
        torch.save(states, path)

        with open(path + '.opt', 'w') as handle:
            json.dump(self.opt, handle)
            handle.write('\n')

    def load(self, path):
        """
        Load a model.

        :param path:
            path from which to load model
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        if 'model' in states:
            self.model.load_state_dict(states['model'])
