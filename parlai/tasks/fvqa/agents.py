from parlai.core.agents import Teacher
from parlai.core.image_featurizers import ImageLoader
from parlai.core.metrics import Metrics
from .build import build

import json
import random
import os

def _path(opt):
    build(opt)

    questions_path = os.path.join(opt['datapath'], 'FVQA', 'new_dataset_release', 'all_qs_dict_release.json')
    trainset_path = os.path.join(opt['datapath'], 'FVQA', 'Name_Lists')
    image_path = os.path.join(opt['datapath'], 'FVQA', 'new_dataset_release', 'images', '')

    return questions_path, trainset_path, image_path


class FVQATeacher(Teacher):
    """FVQA Teacher, which loads the json VQA data and implements its own
    `act` method for interacting with student agent.
    """
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FVQA Task Arguments')
        agent.add_argument('--fvqa-trainset', default=0, choices=range(0, 5), type=int,
                           help="ID of train/test split.  0-4 inclusive.")

    def __init__(self, opt, shared=None):
        super().__init__(opt)

        dt = opt['datatype'].split(':')[0]
        if dt not in ('train', 'test'):
            raise RuntimeError('Not valid datatype.')

        if not hasattr(self, 'factmetrics'):
            if shared and shared.get('factmetrics'):
                self.factmetrics = shared['factmetrics']
            else:
                self.factmetrics = Metrics(opt)
            self.datatype = opt['datatype']
        questions_path, trainset_path, self.image_path = _path(opt)

        if shared and 'ques' in shared:
            self.ques = shared['ques']
        else:
            self._setup_data(questions_path, trainset_path, opt)
        self.len = len(self.ques)

        self.asked_question = False
        # for ordered data in batch mode (especially, for validation and
        # testing), each teacher in the batch gets a start index and a step
        # size so they all process disparate sets of the data
        self.step_size = opt.get('batchsize', 1)
        self.data_offset = opt.get('batchindex', 0)
        self.image_loader = ImageLoader(opt)

        self.reset()

    def __len__(self):
        return self.len

    def report(self):
        r = super().report()
        r['factmetrics'] = self.factmetrics.report()
        return r

    def reset(self):
        # Reset the dialog so that it is at the start of the epoch,
        # and all metrics are reset.
        super().reset()
        self.lastY = None
        self.episode_idx = self.data_offset - self.step_size

    def reset_metrics(self):
        super().reset_metrics()
        self.factmetrics.clear()

    def observe(self, observation):
        """Process observation for metrics."""
        if self.lastY is not None:
            if self.asked_question:
                self.metrics.update(observation, self.lastY[0])
            else:
                self.factmetrics.update(observation, self.lastY[1])
                self.lastY = None
        return observation

    def act(self):
        if self.asked_question:
            self.asked_question = False
            action = {'text': 'What fact supports this answer?', 'episode_done': True}
            if self.datatype.startswith('train'):
                action['labels'] = self.lastY[1]
            if self.datatype != 'train' and self.episode_idx + self.step_size >= len(self):
                self.epochDone = True
            return action

        if self.datatype == 'train':
            self.episode_idx = random.randrange(self.len)
        else:
            self.episode_idx = (self.episode_idx + self.step_size) % len(self)

        self.asked_question = True
        qa = self.ques[self.episode_idx]
        question = qa['question']
        img_path = self.image_path + qa['img_file']

        action = {
            'image': self.image_loader.load(img_path),
            'text': question,
            'episode_done': False
        }

        human_readable = qa['fact_surface'].replace('[', '').replace(']', '')
        self.lastY = [[qa['answer']], [human_readable]]

        if self.datatype.startswith('train'):
            action['labels'] = self.lastY[0]

        return action

    def share(self):
        shared = super().share()
        shared['factmetrics'] = self.factmetrics
        shared['ques'] = self.ques
        if hasattr(self, 'facts'):
            shared['facts'] = self.facts
        return shared

    def _setup_data(self, questions_path, trainset_path, opt):
        print('loading: ' + questions_path)
        with open(questions_path) as questions_file:
            questions = json.load(questions_file)
        train_test = opt['datatype'].split(':')[0]
        train_test_images = set()
        with open(os.path.join(trainset_path, '{}_list_{}.txt'.format(train_test, opt['fvqa_trainset']))) as imageset:
            for line in imageset:
                train_test_images.add(line.strip())
        self.ques = [questions[k] for k in sorted(questions.keys()) if questions[k]['img_file'] in train_test_images]

class DefaultTeacher(FVQATeacher):
    pass
