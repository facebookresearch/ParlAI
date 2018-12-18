#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher
from parlai.core.teachers import FixedDialogTeacher
from .build import build
from .worlds import Simulator, is_action

import copy
import json
import os

from collections import namedtuple
from torch.utils.data.dataset import Dataset
from torch import nn, optim
import torch

from parlai.tasks.talkthewalk.ttw.dict import Dictionary, LandmarkDictionary, ActionAgnosticDictionary, ActionAwareDictionary, TextrecogDict, \
    START_TOKEN, END_TOKEN
from parlai.tasks.talkthewalk.ttw.utils import list_to_tensor
from parlai.tasks.talkthewalk.ttw.models.language import TouristLanguage, GuideLanguage

from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.core.utils import padded_tensor, set_namedtuple_defaults

Batch = namedtuple('Batch', ['text_vec', 'text_lengths', 'label_vec',
                             'label_lengths', 'labels', 'valid_indices',
                             'candidates', 'candidate_vecs', 'image',
                             'memory_vecs', 'move_lengths', 'move_vec',
                             'see_vec', 'see_mask', 'target_location',
                             'landmark_vec'])

set_namedtuple_defaults(Batch, default=None)

def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    opt['ttw_data'] = os.path.join(opt['datapath'], 'TalkTheWalk')
    return opt['ttw_data'], os.path.join(opt['ttw_data'], 'talkthewalk.' + dt + '.json')

class TTWAgent(TorchAgent):
    def is_tourist(self):
        return self.id == 'tourist'

    def zero_grad(self):
        """Zero out optimizer."""
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        for optimizer in self.optims.values():
            optimizer.step()

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared['model'] = self.model
        return shared


class GuideAgent(TTWAgent):

    #TODO: switch to using some of the default args here
    @staticmethod
    def add_cmdline_args(argparser):
        TorchAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Talk the Walk Guide Arguments')
        agent.add_argument('--apply-masc', action='store_true', help='If true, use MASC mechanism in the models')
        agent.add_argument('--T', type=int, default=2, help='Maximum predicted length of trajectory taken by the tourist')
        agent.add_argument('--hidden-sz', type=int, default=256, help='Number of hidden units of language encoder')
        agent.add_argument('--embed-sz', type=int, default=128, help='Word embedding size')
        agent.add_argument('--last-turns', type=int, default=1,
                            help='Specifies how many utterances from the dialogue are included to predict the location. '
                                 'Note that guide utterances will be included as well.')

        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        self.id = 'guide'

        data_path,_ = _path(opt)

        #TODO: switch to built in dict
        self.dict = Dictionary(file=os.path.join(data_path, 'dict.txt'), min_freq=3)
        self.landmark_dict = LandmarkDictionary()

        if not shared:
            self.model = GuideLanguage(opt['embed_sz'], opt['hidden_sz'],
                    len(self.dict), apply_masc=opt['apply_masc'], T=opt['T'])
            if self.use_cuda:  # set in parent class
                self.model.cuda()

        elif 'model' in shared:
            self.model = shared['model']

        # set up the criterion
        self.criterion = nn.CrossEntropyLoss(reduce=False)

        # set up optims for each module
        lr = opt['learningrate']
        self.optims = {
            'model': optim.Adam(self.model.parameters(), lr=lr),
        }

        self.reset()

    def train_step(self, batch):
        """Train model to produce ys given xs.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.

        Return estimated responses, with teacher forcing on the input sequence
        (list of strings of length batchsize).
        """
        if batch.valid_indices == None:
            return

        self.zero_grad()
        out = self.model.forward(batch)

        #TODO: move the loss function out of the model
        out['sl_loss'].backward()
        self.update_params()

    def eval_step(self, batch):

        if  batch.valid_indices == None \
            or torch.sum(batch.move_vec) == 0 \
            or torch.sum(batch.see_vec) == 0 \
            or sum(batch.move_vec.size()) == 0 \
            or sum(batch.see_vec.size()) == 0 \
            or batch.see_vec.size(0) < 120:
            return

        out = self.model.forward(batch)
        predictions = {'text':out['text'] for x in out}
        return Output(self.v2t(predictions))

    def batchify(self, obs_batch, **kwargs):
        batch = super().batchify(obs_batch, **kwargs)

        if batch.valid_indices == None:
            return batch

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if i in
                batch.valid_indices]

        valid_inds, exs = zip(*valid_obs)

        ls = None
        if any('landmark_vec' in ex for ex in exs):
            lands = [ex.get('landmark_vec') or [] for ex in exs]
            ls, _ = list_to_tensor(lands)

        tl = None
        if any('target_location' in ex for ex in exs):
            _tl = [ex.get('target_location', self.EMPTY) for ex in exs]
            tl, _ = padded_tensor(_tl, self.NULL_IDX, self.use_cuda)

        batch = batch._asdict()
        batch['landmark_vec'] = ls
        batch['target_location'] = tl
        return Batch(**batch)

    def vectorize(self, obs, *args, **kwargs):
        obs = super().vectorize(obs, *args, **kwargs)
        if obs.get('landmarks'):
            obs['landmark_vec'] = self._vectorize_landmarks(obs['landmarks'])
        return obs

    def _vectorize_landmarks(self, landmarks):
        for i, row in enumerate(landmarks):
            for j, col in enumerate(row):
                landmarks[i][j] = [self.landmark_dict.encode(x) for x in col]
        return landmarks

class TouristAgent(TTWAgent):

    @staticmethod
    def add_cmdline_args(argparser):
        TorchAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Talk The Walk Tourist Arguments')
        agent.add_argument('--act-emb-sz', type=int, default=128, help='Dimensionality of action embedding')
        agent.add_argument('--act-hid-sz', type=int, default=128, help='Dimensionality of action encoder')
        agent.add_argument('--obs-emb-sz', type=int, default=128, help='Dimensionality of observation embedding')
        agent.add_argument('--obs-hid-sz', type=int, default=128, help='Dimensionality of observation encoder')
        agent.add_argument('--decoder-emb-sz', type=int, default=128, help='Dimensionality of word embeddings')
        agent.add_argument('--decoder-hid-sz', type=int, default=1024, help='Hidden size of decoder RNN')
        TouristAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        self.id = 'tourist'

        data_path,_ = _path(opt)

        self.dict = Dictionary(file=os.path.join(data_path, 'dict.txt'), min_freq=3)
        self.act_dict = ActionAgnosticDictionary()
        self.landmark_dict = LandmarkDictionary()

        self.START_IDX = self.dict.tok2i[START_TOKEN]
        self.END_IDX = self.dict.tok2i[END_TOKEN]
        if not shared:
            self.model = TouristLanguage(opt['act_emb_sz'], opt['act_hid_sz'],
                    4, opt['obs_emb_sz'],
                              opt['obs_hid_sz'], 11,
                              opt['decoder_emb_sz'], opt['decoder_hid_sz'], len(self.dict),
                              start_token=self.START_IDX,
                              end_token=self.END_IDX)

            if self.use_cuda:  # set in parent class
                self.model.cuda()

        elif 'model' in shared:
            # copy initialized data from shared table
            self.model = shared['model']

        # set up the criterion
        self.criterion = nn.CrossEntropyLoss(reduce=False)

        # set up optims for each module
        lr = opt['learningrate']
        self.optims = {
            'model': optim.Adam(self.model.parameters(), lr=lr),
        }

        self.reset()

    def observe(self, obs):
        if obs.get('see'):
            self.see_memory.append(obs.get('see'))
        if obs.get('location') and self.location != obs.get('location'):
            self.location = obs['location']
            self.move_memory.append(self.location)
        super(TouristAgent, self).observe(obs)
        if obs.get('episode_done'):
            self.see_memory = []
            self.move_memory = []
            self.location = None
        return obs

    def reset(self):
        super(TouristAgent, self).reset()
        self.see_memory = []
        self.move_memory = []
        self.location = None


    def vectorize(self, obs, *args, **kwargs):
        obs = super().vectorize(obs, *args, **kwargs)
        obs['move_vec'] = self._vectorize_move()
        obs['see_vec'] = self._vectorize_see()
        return obs

    def _vectorize_move(self, steps=4):
        if len(self.move_memory) < 2:
            return []
        moves = self.move_memory[-steps:]
        move_vec = []

        #feed act_dict pairs of locations, most recent first
        moves.reverse()
        for i in range(len(moves)-1):
            move_vec.append(self.act_dict.encode_from_location(*(moves[i:i+2])))
        return move_vec

    def _vectorize_see(self, steps=3):
        if len(self.see_memory) == 0:
            return []
        saw = self.see_memory[-steps:]
        return [[self.landmark_dict.encode(y) for y in x] for x in saw]


    def batchify(self, obs_batch, **kwargs):
        batch = super().batchify(obs_batch, **kwargs)

        if batch.valid_indices == None:
            return batch

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if i in
                batch.valid_indices]

        valid_inds, exs = zip(*valid_obs)

        # MOVE
        ms, m_lens = None, None
        if any('move_vec' in ex for ex in exs):
            _ms = [ex.get('move_vec', self.EMPTY) for ex in exs]
            ms, m_lens = padded_tensor(_ms, self.NULL_IDX, self.use_cuda)

        # SEE
        ss, s_lens = None, None
        if any('see_vec' in ex for ex in exs):
            _ss = [ex.get('see_vec', self.EMPTY) for ex in exs]
            ss, s_lens = list_to_tensor(_ss)


        batch = batch._asdict()
        batch['move_vec'] = ms
        batch['move_lengths'] = m_lens
        batch['see_vec'] = ss
        batch['see_mask'] = s_lens

        return Batch(**batch)

    def train_step(self, batch):
        """Train model to produce ys given xs.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.

        Return estimated responses, with teacher forcing on the input sequence
        (list of strings of length batchsize).
        """

        #TODO: clean up

        # we need to wait to build up some memory so batch sizes
        # are reasonable.
        if  batch.valid_indices == None \
            or torch.sum(batch.move_vec) == 0 \
            or torch.sum(batch.see_vec) == 0 \
            or sum(batch.move_vec.size()) == 0 \
            or sum(batch.see_vec.size()) == 0 \
            or batch.see_vec.size(0) < 120:
            return

        self.zero_grad()
        out = self.model.forward(batch, train=True)

        #TODO: move the loss function out of the model
        out['loss'].backward()
        self.update_params()

    def eval_step(self, batch):

        if  batch.valid_indices == None \
            or torch.sum(batch.move_vec) == 0 \
            or torch.sum(batch.see_vec) == 0 \
            or sum(batch.move_vec.size()) == 0 \
            or sum(batch.see_vec.size()) == 0 \
            or batch.see_vec.size(0) < 120:
            return

        out = self.model.forward(batch, train=False)
        predictions = {'text':out['text'] for x in out}
        return Output(self.v2t(predictions))


class TTWTeacher(FixedDialogTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Talk The Walk Teacher Arguments')
        agent.add_argument('--train-actions', type=bool, default=False, help='Train model to take actions')


    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        self.datatype = self.opt.get('datatype')
        self.training = self.datatype.startswith('train')
        self.num_epochs = self.opt.get('num_epochs', 0)
        data_path, datafile = _path(opt)

        if shared:
            self.data = shared['data']
            self.sim = shared['sim']
        else:
            self.sim = Simulator(opt)
            self._setup_data(datafile)
        self.reset()


    def _setup_data(self, datafile):
        self.episodes = json.load(open(datafile))
        self.data = []
        self.examples_count = 0

        for episode in self.episodes:
            init = {x:y for x,y in episode.items() if x in ['start_location',
                'neighborhood', 'boundaries', 'target_location']}
            self.sim.init_sim(**init)
            if episode:
                episode = self._setup_episode(episode)
                if episode:
                    self.data.append(episode)
                    self.examples_count += len(episode)


    def get(self, episode_idx, entry_idx=0):
        return self.data[episode_idx][entry_idx]

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['sim'] = self.sim
        return shared

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return self.examples_count


class TouristTeacher(TTWTeacher):
    def _setup_episode(self, episode):
        ep = []
        example = {'episode_done': False}
        for msg in episode['dialog']:
            text = msg['text']
            if msg['id'] == 'Tourist':
                if self.opt['train_actions'] or not is_action(text):
                    example['labels'] = [text]
                    ep.append(example)
                    example = {'episode_done': False}
                # add movements to text history if not training on them
                if not self.opt['train_actions'] and is_action(text):
                    example['text'] = example.get('text', '') + text + '\n'
            elif msg['id'] == 'Guide':
                example['text'] = example.get('text', '') + text + '\n'

            self.sim.execute(text)
            self.sim.add_view_to_text(example, text)

        if len(ep):
            ep[-1]['episode_done'] = True
        return ep



class GuideTeacher(TTWTeacher):
    def _setup_episode(self, episode):
        ep = []
        example = {'episode_done': False,
                'target_location':self.sim.target_location,
                'landmarks':self.sim.landmarks, 'text':
                self.sim.get_text_map()}
        for msg in episode['dialog']:
            text = msg['text']
            if msg['id'] == 'Guide':
                if self.opt['train_actions'] or not text.startswith('EVALUATE'):
                    example['labels'] = [text]
                    ep.append(example)
                    example = {'episode_done': False}
            elif msg['id'] == 'Tourist' and not is_action(text):
                example['text'] = example.get('text', '') + text + '\n'

            self.sim.execute(text)

        if len(ep):
            ep[-1]['episode_done'] = True
        return ep



class DefaultTeacher(TouristTeacher):
    pass
