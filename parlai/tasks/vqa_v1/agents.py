# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build, buildImage
try:
    import torch
except Exception as e:
    raise ModuleNotFoundError('Need to install Pytorch: go to pytorch.org')
from torch.utils.data import Dataset
from parlai_external.agents.mlb.mlb import VqaDictionaryAgent

import json
import os


def _path(opt):
    build(opt)
    buildImage(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        ques_suffix = 'MultipleChoice_mscoco_train2014'
        annotation_suffix = 'mscoco_train2014'
        img_suffix = os.path.join('train2014', 'COCO_train2014_')
    elif dt == 'valid':
        ques_suffix = 'MultipleChoice_mscoco_val2014'
        annotation_suffix = 'mscoco_val2014'
        img_suffix = os.path.join('val2014', 'COCO_val2014_')
    elif dt == 'test':
        ques_suffix = 'MultipleChoice_mscoco_test2015'
        annotation_suffix = 'None'
        img_suffix = os.path.join('test2015', 'COCO_test2015_')
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'VQA-v1',
                             ques_suffix + '_questions.json')

    annotation_path = os.path.join(opt['datapath'], 'VQA-v1',
                                   annotation_suffix + '_annotations.json')

    image_path = os.path.join(opt['datapath'], 'COCO-IMG', img_suffix)

    return data_path, annotation_path, image_path


class VQADataset(Dataset):
    """A Pytorch Dataset utilizing streaming"""
    def __init__(self, opt):
        self.opt = opt
        self.datatype = self.opt.get('datatype')
        _, _, self.image_path = _path(opt)
        self.image_loader = ImageLoader(opt)
        data_path, annotation_path, self.image_path = _path(opt)
        self._setup_data(data_path, annotation_path)
        self.dict_agent = VqaDictionaryAgent(opt)

    def __getitem__(self, index):
        index %= self.num_episodes()
        qa = self.ques['questions'][index]
        im_path = self.image_path + '%012d.jpg' % (qa['image_id'])
        ep = {
            'text': qa['question'],
            'image': self.image_loader.load(im_path),
            'episode_done': True
        }
        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][index]
            labels = [ans['answer'] for ans in anno['answers']]
            ep['labels'] = [ans['answer'] for ans in anno['answers']]
            ep['valid'] = True
            if 'mc_label' in ep:
                if not ep['mc_label'][0] in self.dict_agent.ans2ind:
                    ep['valid'] = False
            ep = self.dict_agent.encode_question([ep], True)
            ep = self.dict_agent.encode_answer(ep)
            ep[0]['labels'] = labels
        else:
            ep = self.dict_agent.encode_question([ep], False)
        return (index, ep)

    def __len__(self):
        return int(self.num_episodes() * max(self.opt.get('num_epochs'), 1))

    def _load_lens(self):
        with open(self.length_datafile) as length:
            lengths = json.load(length)
            self.num_eps = lengths['num_eps']
            self.num_exs = lengths['num_exs']

    def _setup_data(self, data_path, annotation_path):
        with open(data_path) as data_file:
            self.ques = json.load(data_file)
        if not self.datatype.startswith('test'):
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)
        self.image_paths = set()
        for qa in self.ques['questions']:
            self.image_paths.add(self.image_path + '%012d.jpg' % (qa['image_id']))


    def num_episodes(self):
        return len(self.ques['questions'])

    def num_examples(self):
        return self.num_episodes()


class DefaultDataset(VQADataset):
    pass


class OeTeacher(FixedDialogTeacher):
    """
    VQA Open-Ended teacher, which loads the json vqa data and implements its
    own `act` method for interacting with student agent.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        data_path, annotation_path, self.image_path = _path(opt)
        self.datafile = data_path
        self.image_mode = opt.get('image_mode', 'none')

        if shared and 'ques' in shared:
            self.ques = shared['ques']
            if 'annotation' in shared:
                self.annotation = shared['annotation']
            self.image_loader = shared['image_loader']
        else:
            self._setup_data(data_path, annotation_path)
            self.image_loader = ImageLoader(opt)
        self.reset()

    def reset(self):
        super().reset()
        self.example = None
        # call this once to get the cache moving
        self.next_example()

    def num_examples(self):
        """Number of examples in VQA-v1."""
        return len(self.ques['questions'])

    def num_episodes(self):
        # same as number of examples since all episodes are of length one
        return self.num_examples()

    def submit_load_request(self, image_id):
        img_path = self.image_path + '%012d.jpg' % (image_id)
        self.data_loader.request_load(self.receive_data, self.image_loader.load, (img_path,))

    def get(self, episode_idx, entry_idx=0):
        # queue up the next one
        qa = self.ques['questions'][episode_idx]
        question = qa['question']

        action = {
            'text': question,
            'image_id': qa['image_id'],
            'episode_done': True
        }

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][episode_idx]
            action['labels'] = [ans['answer'] for ans in anno['answers']]

        return action

    def next_example(self):
        # save the currently queued example
        ready = None
        if self.example is not None:
            if self.image_mode != 'none':
                image = self.data_queue.get()
                self.example['image'] = image
            ready = (self.example, self.epochDone)
        # queue up the next example
        self.example, self.epochDone = super().next_example()
        image_id = self.example['image_id']
        if self.image_mode != 'none':
            self.submit_load_request(image_id)
        return ready

    def share(self):
        shared = super().share()
        shared['ques'] = self.ques
        if hasattr(self, 'annotation'):
            shared['annotation'] = self.annotation
        shared['image_loader'] = self.image_loader
        return shared

    def _setup_data(self, data_path, annotation_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.ques = json.load(data_file)

        if not self.datatype.startswith('test'):
            print('loading: ' + annotation_path)
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)


class McTeacher(OeTeacher):
    """
    VQA Multiple-Choice teacher, which inherits from OeTeacher but overrides
    the label and label_candidates fields with multiple choice data.
    """

    def get(self, episode_idx, entry_idx=0):
        action = super().get(episode_idx, entry_idx)
        qa = self.ques['questions'][episode_idx]
        multiple_choices = qa['multiple_choices']
        action['label_candidates'] = multiple_choices

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][episode_idx]
            action['labels'] = [anno['multiple_choice_answer']]

        return action


class AllTeacher(OeTeacher):
    """
    VQA Teacher, which inherits from OeTeacher and gives access to
    the multiple choices and the multiple choice answer.
    """

    def act(self):
        # parent class increments episode_idx after getting ex, so need to
        # cache the episode_idx first
        episode_idx = self.episode_idx
        action = super().act()

        qa = self.ques['questions'][episode_idx]
        multiple_choices = qa['multiple_choices']

        action['label_candidates'] = multiple_choices

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][episode_idx]
            self.mclabel = [anno['multiple_choice_answer']]

        if self.datatype.startswith('train'):
            action['mc_label'] = self.mclabel

        return action


class DefaultTeacher(McTeacher):
    # default to Multiple-Choice Teacher
    pass
