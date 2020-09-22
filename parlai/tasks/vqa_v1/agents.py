#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
import numpy as np
from collections import Counter

from parlai.core.agents import Agent
from parlai.utils.io import PathManager
from collections import defaultdict
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage as buildImage_2014
from parlai.tasks.coco_caption.build_2015 import buildImage as buildImage_2015


def _path(opt):
    build(opt)
    buildImage_2014(opt)
    buildImage_2015(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        ques_suffix = 'MultipleChoice_mscoco_train2014'
        annotation_suffix = 'mscoco_train2014'
        img_suffix = os.path.join('train2014', 'COCO_train2014_')
        img_version = '2014'
    elif dt == 'valid':
        ques_suffix = 'MultipleChoice_mscoco_val2014'
        annotation_suffix = 'mscoco_val2014'
        img_suffix = os.path.join('val2014', 'COCO_val2014_')
        img_version = '2014'
    elif dt == 'test':
        ques_suffix = 'MultipleChoice_mscoco_test2015'
        annotation_suffix = 'None'
        img_suffix = os.path.join('test2015', 'COCO_test2015_')
        img_version = '2015'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'VQA-v1', ques_suffix + '_questions.json')

    annotation_path = os.path.join(
        opt['datapath'], 'VQA-v1', annotation_suffix + '_annotations.json'
    )

    image_path = os.path.join(
        opt['datapath'], 'COCO-IMG-{}'.format(img_version), img_suffix
    )

    return data_path, annotation_path, image_path


def escape(s):
    """
    Replace potential special characters with escaped version.

    For example, newline => \\n and tab => \\t
    """
    return s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')


def unescape(s):
    """
    Revert escaped characters back to their special version.

    For example, \\n => newline and \\t => tab
    """
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')


class VqaDictionaryAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        dictionary = argparser.add_argument_group('Dictionary Arguments')
        dictionary.add_argument(
            '--dict-file',
            help='if set, the dictionary will automatically save to this path'
            + ' during shutdown',
        )
        dictionary.add_argument(
            '--dict-initpath',
            help='path to a saved dictionary to load tokens / counts from to '
            + 'seed the dictionary with initial tokens and/or frequencies',
        )
        dictionary.add_argument(
            '--dict-maxexs',
            default=300000,
            type=int,
            help='max number of examples to build dict on',
        )
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

        if opt.get('dict_file') and PathManager.exists(opt['dict_file']):
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
        """
        Builds dictionary from the list of provided tokens.

        Only adds words contained in self.embedding_words, if not None.
        """
        for token in tokens:
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

    def add_to_ans_dict(self, token):
        """
        Builds dictionary from the list of provided tokens.

        Only adds words contained in self.embedding_words, if not None.
        """
        self.ansfreq[token] += 1
        if token not in self.ans2ind:
            index = len(self.ans2ind)
            self.ans2ind[token] = index
            self.ind2ans[index] = token

    def tokenize_mcb(self, s):
        t_str = s.lower()
        for i in [
            r'\?',
            r'\!',
            r'\'',
            r'\"',
            r'\$',
            r'\:',
            r'\@',
            r'\(',
            r'\)',
            r'\,',
            r'\.',
            r'\;',
        ]:
            t_str = re.sub(i, '', t_str)
        for i in [r'\-', r'\/']:
            t_str = re.sub(i, ' ', t_str)
        q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
        q_list = list(filter(lambda x: len(x) > 0, q_list))
        return q_list

    def split_tokenize(self, s):
        return (
            s.lower()
            .replace('.', ' . ')
            .replace('. . .', '...')
            .replace(',', ' , ')
            .replace(';', ' ; ')
            .replace(':', ' : ')
            .replace('!', ' ! ')
            .replace('?', ' ? ')
            .split()
        )

    def act(self):
        """
        Add any words passed in the 'text' field of the observation to this dictionary.
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
        """
        Load pre-existing dictionary in 'token[<TAB>count]' format.

        Initialize counts from other dictionary, or 0 if they aren't included.
        """
        print('Dictionary: loading dictionary from {}'.format(filename))
        with PathManager.open(filename) as read:
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

        with PathManager.open(filename[:-5] + "_ans.dict") as read:
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
        """
        Save dictionary to file. Format is 'token<TAB>count' for every token in the
        dictionary, sorted by count with the most frequent words first.

        If ``append`` (default ``False``) is set to ``True``, appends instead
        of overwriting.

        If ``sort`` (default ``True``), then first sort the dictionary before
        saving.
        """
        cw = sorted([(count, w) for w, count in self.ansfreq.items()], reverse=True)
        final_exs = cw[: self.opt.get('nans', 2000)]
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

        with PathManager.open(filename, 'a' if append else 'w') as write:
            for i in range(len(self.ind2tok)):
                tok = self.ind2tok[i]
                cnt = self.freq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))

        with PathManager.open(
            filename[:-5] + "_ans.dict", 'a' if append else 'w'
        ) as write:
            for i in range(len(self.ind2ans)):
                tok = self.ind2ans[i]
                cnt = self.ansfreq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))

    def shutdown(self):
        """
        Save on shutdown if ``save_path`` is set.
        """
        if hasattr(self, 'save_path'):
            self.save(self.save_path)


class OeTeacher(FixedDialogTeacher):
    """
    VQA Open-Ended teacher, which loads the json vqa data and implements its own `act`
    method for interacting with student agent.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        data_path, annotation_path, self.image_path = _path(opt)
        self.datafile = data_path
        self.image_mode = opt.get('image_mode', 'no_image_model')

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
        self.imageEpochDone = False

    def num_examples(self):
        """
        Number of examples in VQA-v1.
        """
        return len(self.ques['questions'])

    def num_episodes(self):
        # same as number of examples since all episodes are of length one
        return self.num_examples()

    def submit_load_request(self, image_id):
        img_path = self.image_path + '%012d.jpg' % (image_id)
        self.data_loader.request_load(
            self.receive_data, self.image_loader.load, (img_path,)
        )

    def get(self, episode_idx, entry_idx=0):
        # queue up the next one
        qa = self.ques['questions'][episode_idx]
        question = qa['question']

        action = {'text': question, 'image_id': qa['image_id'], 'episode_done': True}

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][episode_idx]
            action['labels'] = [ans['answer'] for ans in anno['answers']]

        return action

    def next_example(self):
        """
        Returns the next example from this dataset after starting to queue up the next
        example.
        """
        ready = None
        # pull up the currently queued example
        if self.example is not None:
            if self.image_mode != 'no_image_model' and 'image_id' in self.example:
                # move the image we loaded in the background into the example
                image = self.data_queue.get()
                self.example['image'] = image
            ready = (self.example, self.imageEpochDone)
        # get the next base example: super().next_example() calls self.get()
        self.example, self.imageEpochDone = super().next_example()
        if self.image_mode != 'no_image_model' and 'image_id' in self.example:
            # load the next image in the background
            image_id = self.example['image_id']
            self.submit_load_request(image_id)
        # Try to return the previously cached example
        if ready is None:
            return self.next_example()
        else:
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
        with PathManager.open(data_path) as data_file:
            self.ques = json.load(data_file)

        if not self.datatype.startswith('test'):
            print('loading: ' + annotation_path)
            with PathManager.open(annotation_path) as data_file:
                self.annotation = json.load(data_file)


class McTeacher(OeTeacher):
    """
    VQA Multiple-Choice teacher, which inherits from OeTeacher but overrides the label
    and label_candidates fields with multiple choice data.
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
    VQA Teacher, which inherits from OeTeacher and gives access to the multiple choices
    and the multiple choice answer.
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
