# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.fixed_data_teacher import FixedDataTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build, buildImage

import json
import os
from threading import Thread
import queue
import concurrent.futures


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


class MasterLoader(Thread):
    def __init__(self, opt):
        Thread.__init__(self, daemon=True)
        self.num_threads = opt.get('numthreads', 8)
        self.request_queue = queue.Queue()

    def __len__(self):
        return len(self.ques['questions'])

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            while True:
                teacher, load_fn, data_req = self.request_queue.get()
                future = executor.submit(load_fn, data_req)
                teacher.receive(future)


class OeTeacher(FixedDataTeacher):
    """
    VQA Open-Ended teacher, which loads the json vqa data and implements its
    own `act` method for interacting with student agent.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        data_path, annotation_path, self.image_path = _path(opt)
        self.image_mode = opt.get('image_mode', 'none')

        if shared and 'ques' in shared:
            self.ques = shared['ques']
            if 'annotation' in shared:
                self.annotation = shared['annotation']
            self.image_loader = shared['image_loader']
            self.master_loader = shared['master_loader']
        else:
            self._setup_data(data_path, annotation_path)
            self.image_loader = ImageLoader(opt)
            self.master_loader = MasterLoader(opt)
            self.master_loader.start()

        self.example_queue = queue.Queue()
        self.reset()

    def reset(self):
        super().reset()
        self.example = None
        self.example_queue = queue.Queue()
        # call this once to get the cache moving
        self.next_example()

    def __len__(self):
        """Number of examples in VQA-v1."""
        return len(self.ques['questions'])

    def num_episodes(self):
        # same as number of examples since all episodes are of length one
        return len(self)

    def submit_example_request(self, image_id):
        img_path = self.image_path + '%012d.jpg' % (image_id)
        self.master_loader.request_queue.put(
            (self, self.image_loader.load, img_path))

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
            answers = [ans['answer'] for ans in anno['answers']]
            self.lastY = answers
            action['labels'] = answers

        return action

    def next_example(self):
        # save the currently queued example
        ready = None
        if self.example is not None:
            if self.image_mode != 'none':
                image = self.example_queue.get()
                self.example['image'] = image
            ready = (self.example, self.epochDone)
        # queue up the next example
        self.example, self.epochDone = super().next_example()
        image_id = self.example.pop('image_id')
        if self.image_mode != 'none':
            self.submit_example_request(image_id)
        return ready

    def receive(self, future):
        data = future.result()
        self.example_queue.put(data)

    def share(self):
        shared = super().share()
        shared['ques'] = self.ques
        if hasattr(self, 'annotation'):
            shared['annotation'] = self.annotation
        shared['image_loader'] = self.image_loader
        shared['master_loader'] = self.master_loader
        return shared

    def _setup_data(self, data_path, annotation_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.ques = json.load(data_file)

        if self.datatype != 'test':
            print('loading: ' + annotation_path)
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)


class McTeacher(OeTeacher):
    """
    VQA Multiple-Choice teacher, which inherits from OeTeacher but overrides
    the label and label_candidates fields with multiple choice data.
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
            self.lastY = [anno['multiple_choice_answer']]

        if self.datatype.startswith('train'):
            action['labels'] = self.lastY

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
