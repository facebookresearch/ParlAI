#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
These agents contain a number of "unit test" corpora, or fake corpora that ensure models
can learn simple behavior easily. They are useful as unit tests for the basic models.

The corpora are all randomly, but deterministically generated
"""

from parlai.core.teachers import (
    FixedDialogTeacher,
    DialogTeacher,
    AbstractImageTeacher,
    Teacher,
    ChunkTeacher,
)
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
import time
from parlai.utils.io import PathManager

# default parameters
VOCAB_SIZE = 7
EXAMPLE_SIZE = 4
NUM_CANDIDATES = 10
NUM_TRAIN = 500
NUM_TEST = 100
INFINITE = 1e20


class CandidateBaseTeacher(Teacher, ABC):
    """
    Base Teacher.

    Contains some functions that are useful for all the subteachers.
    """

    def __init__(
        self,
        opt: Opt,
        shared: dict = None,
        vocab_size: int = VOCAB_SIZE,
        example_size: int = EXAMPLE_SIZE,
        num_candidates: int = NUM_CANDIDATES,
        num_train: int = NUM_TRAIN,
        num_test: int = NUM_TEST,
    ):
        """
        :param int vocab_size:
            size of the vocabulary
        :param int example_size:
            length of each example
        :param int num_candidates:
            number of label_candidates generated
        :param int num_train:
            size of the training set
        :param int num_test:
            size of the valid/test sets
        """
        self.opt = opt
        opt['datafile'] = opt['datatype'].split(':')[0]
        self.datafile = opt['datafile']

        self.vocab_size = vocab_size
        self.example_size = example_size
        self.num_candidates = num_candidates
        self.num_train = num_train
        self.num_test = num_test

        # set up the vocabulary
        self.words = list(map(str, range(self.vocab_size)))
        super().__init__(opt, shared)

    def build_corpus(self):
        """
        Build corpus; override for customization.
        """
        return [list(x) for x in itertools.permutations(self.words, self.example_size)]

    def num_episodes(self) -> int:
        if self.datafile == 'train':
            return self.num_train
        else:
            return self.num_test

    def num_examples(self) -> int:
        return self.num_episodes()

    def _setup_data(self, fold: str):
        # N words appearing in a random order
        self.rng = random.Random(42)
        full_corpus = self.build_corpus()
        self.rng.shuffle(full_corpus)

        it = iter(full_corpus)
        self.train = list(itertools.islice(it, self.num_train))
        self.val = list(itertools.islice(it, self.num_test))
        self.test = list(itertools.islice(it, self.num_test))

        # check we have enough data
        assert len(self.train) == self.num_train, len(self.train)
        assert len(self.val) == self.num_test, len(self.val)
        assert len(self.test) == self.num_test, len(self.test)

        # check every word appear in the training set
        assert len(set(itertools.chain(*self.train)) - set(self.words)) == 0

        # select which set we're using
        if fold == "train":
            self.corpus = self.train
        elif fold == "valid":
            self.corpus = self.val
        elif fold == "test":
            self.corpus = self.test

        # make sure the corpus is actually text strings
        self.corpus = [' '.join(x) for x in self.corpus]


class FixedDialogCandidateTeacher(CandidateBaseTeacher, FixedDialogTeacher):
    """
    Base Candidate Teacher.

    Useful if you'd like to test the FixedDialogTeacher
    """

    def __init__(self, *args, **kwargs):
        """
        Override to build candidates.
        """
        super().__init__(*args, **kwargs)
        opt = args[0]
        if 'shared' not in kwargs:
            self._setup_data(opt['datatype'].split(':')[0])
            self._build_candidates()
        else:
            shared = kwargs['shared']
            self.corpus = shared['corpus']
            self.cands = shared['cands']
        self.reset()

    def share(self):
        shared = super().share()
        shared['corpus'] = self.corpus
        shared['cands'] = self.cands
        return shared

    def _build_candidates(self):
        self.cands = []
        for i in range(len(self.corpus)):
            cands = []
            for j in range(NUM_CANDIDATES):
                offset = (i + j) % len(self.corpus)
                cands.append(self.corpus[offset])
            self.cands.append(cands)

    def get(self, episode_idx: int, entry_idx: int = 0):
        return {
            'text': self.corpus[episode_idx],
            'episode_done': True,
            'labels': [self.corpus[episode_idx]],
            'label_candidates': self.cands[episode_idx],
        }


class CandidateTeacher(CandidateBaseTeacher, DialogTeacher):
    """
    Candidate teacher produces several candidates, one of which is a repeat of the
    input.

    A good ranker should easily identify the correct response.
    """

    def setup_data(self, fold):
        super()._setup_data(fold)
        for i, text in enumerate(self.corpus):
            cands = []
            for j in range(NUM_CANDIDATES):
                offset = (i + j) % len(self.corpus)
                cands.append(self.corpus[offset])
            yield (text, [text], 0, cands), True


class OverfitTeacher(CandidateTeacher, DialogTeacher):
    @classmethod
    def add_cmdline_args(self, argparser):
        argparser.add_argument('--corpus-size', default=4, type=int)

    def __init__(self, opt, shared=None):
        self.corpussize = opt.get('corpus_size', 4)
        super().__init__(opt, shared)

    def setup_data(self, fold):
        super()._setup_data('train')
        for i, text in enumerate(self.corpus[: self.corpussize]):
            cands = []
            for j in range(NUM_CANDIDATES):
                offset = (i + j) % len(self.corpus)
                cands.append(self.corpus[offset])
            yield (text, [text], 0, cands), True

    def num_examples(self):
        return self.corpussize

    def num_episodes(self):
        return self.corpussize


class OverfitMultiturnTeacher(CandidateTeacher, DialogTeacher):
    @classmethod
    def add_cmdline_args(self, argparser):
        argparser.add_argument('--corpus-size', default=4, type=int)

    def __init__(self, opt, shared=None):
        self.corpussize = opt.get('corpus_size', 4)
        super().__init__(opt, shared)

    def setup_data(self, fold):
        super()._setup_data('train')
        for text in self.corpus[: self.corpussize]:
            words = text.split(' ')
            for j in range(1, len(words) + 1):
                real_text = ' '.join(words[:j])
                yield (real_text, text), True

    def num_examples(self):
        return self.corpussize * EXAMPLE_SIZE

    def num_episodes(self):
        return self.corpussize * EXAMPLE_SIZE


class VariableLengthTeacher(CandidateTeacher):
    def build_corpus(self):
        corpus = super().build_corpus()
        for i in range(len(corpus)):
            length = len(corpus[i]) - i % 3
            corpus[i] = corpus[i][:length]
        return corpus


class MultiturnCandidateTeacher(CandidateTeacher):
    """
    Splits inputs/targets by spaces into multiple turns.

    Good for testing models that use the dialog history.
    """

    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, a, r, cs), _e in raw:
            split_t = t.split(' ')
            split_a = a[0].split(' ')
            split_cs = [c.split(' ') for c in cs]
            for i in range(len(split_t)):
                yield (
                    (
                        split_t[i],
                        [' '.join(split_a[: i + 1])],
                        r,
                        [' '.join(c[: i + 1]) for c in split_cs],
                    ),
                    i == 0,
                )

    def num_examples(self):
        return self.example_size * self.num_episodes()


class MultiturnTeacher(MultiturnCandidateTeacher):
    """
    Simple alias.
    """

    pass


class NocandidateTeacher(CandidateTeacher):
    """
    Strips the candidates so the model can't see any options.

    Good for testing simple generative models.
    """

    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, a, _r, _c), e in raw:
            yield (t, a), e


class RepeatWordsTeacher(NocandidateTeacher):
    """
    Each input/output pair is a word repeated n times.

    Useful for testing beam-blocking.
    """

    def __init__(self, *args, **kwargs):
        # Set sizes so that we have appropriate number of examples (700)
        kwargs['vocab_size'] = 70
        kwargs['example_size'] = 11
        super().__init__(*args, **kwargs)

    def build_corpus(self):
        """
        Override to repeat words.
        """
        return [
            [x for _ in range(l)]
            for l in range(1, self.example_size)
            for x in self.words
        ]


class MultiturnNocandidateTeacher(MultiturnCandidateTeacher):
    """
    Strips the candidates so the model can't see any options.

    Good for testing simple generative models.
    """

    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, a, _r, _c), e in raw:
            yield (t, a), e


class ClassifierTeacher(CandidateTeacher):
    """
    Classifier Teacher.

    Good for testing simple classifier models.
    """

    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, _a, _r, _c), e in raw:
            letters = t.split(' ')
            # everything starts with 0 or 1
            letters[0] = str(int(int(t[0]) % 2))
            label = 'one' if letters[0] == '1' else 'zero'
            text = ' '.join(letters)
            yield (text, [label], 0, ['one', 'zero']), e


class ReverseTeacher(CandidateTeacher):
    """
    Reverse Teacher.

    Label is opposite of text; good for testing more complex generative models.
    """

    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, a, r, c), e in raw:
            label = a[0][::-1]
            yield (t, [label], r, c + [label]), e


class ImageTeacher(AbstractImageTeacher):
    """
    Teacher which provides images and captions.

    In __init__, setup some fake images + features
    """

    def __init__(self, opt, shared=None):
        self._setup_test_data(opt)
        super().__init__(opt, shared)

    def _setup_test_data(self, opt):
        datapath = os.path.join(opt['datapath'], 'ImageTeacher')
        imagepath = os.path.join(datapath, 'images')
        PathManager.mkdirs(imagepath)

        self.image_features_path = os.path.join(
            datapath, f'{opt["image_mode"]}_image_features'
        )

        # Create fake images and features
        imgs = [f'img_{i}' for i in range(10)]
        for i, img in enumerate(imgs):
            image = Image.new('RGB', (16, 16), color=i)
            with PathManager.open(os.path.join(imagepath, f'{img}.jpg'), 'wb') as fp:
                image.save(fp, 'JPEG')

        # write out fake data
        for dt in ['train', 'valid', 'test']:
            random.seed(42)
            data = [
                {'image_id': img, 'text': string.ascii_uppercase[i]}
                for i, img in enumerate(imgs)
            ]
            with PathManager.open(os.path.join(datapath, f'{dt}.json'), 'w') as f:
                json.dump(data, f)

    def get_image_features_path(self, task, image_model_name, dt):
        """
        Return path dummy image features.
        """
        return self.image_features_path

    def image_id_to_image_path(self, image_id):
        """
        Return path to image on disk.
        """
        return os.path.join(
            self.opt['datapath'], 'ImageTeacher/images', f'{image_id}.jpg'
        )


class RepeatTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = 'unused_path'
        task = opt.get('task', 'integration_tests:RepeatTeacher:50')
        try:
            self.data_length = int(task.split(':')[-1])
        except ValueError:
            self.data_length = 10
        super().__init__(opt, shared)

    def setup_data(self, unused_path):
        for i in range(self.data_length):
            yield ((str(i), [str(i)]), True)

    def num_examples(self):
        return self.data_length

    def num_episodes(self):
        return self.data_length


class ChunkyTeacher(ChunkTeacher):
    def _get_data_folder(self):
        return None

    def get_num_samples(self, opt) -> Tuple[int, int]:
        datatype = opt['datatype']
        if 'train' in datatype:
            return NUM_TRAIN, NUM_TRAIN
        elif 'valid' in datatype:
            return NUM_TEST, NUM_TEST
        elif 'test' in datatype:
            return NUM_TEST, NUM_TEST

    def get_fold_chunks(self, opt) -> List[int]:
        datatype = opt['datatype']
        if 'train' in datatype:
            return list(range(50))
        elif 'valid' in datatype:
            return list(range(50, 60))
        elif 'test' in datatype:
            return list(range(60, 70))

    def load_from_chunk(self, chunk_idx: int):
        output = []
        for i in range(10):
            text = " ".join([str(i)] + [str(chunk_idx)] * 5)
            resp = " ".join([str(i)])
            output.append((text, resp))
        return output

    def create_message(self, sample_item, entry_idx=0):
        text, label = sample_item
        return {'text': text, 'labels': [label], 'episode_done': True}


class InfiniteTrainTeacher(ChunkyTeacher):
    """
    Chunk teacher with an effectively infinite number of training examples.
    """

    def get_num_samples(self, opt) -> Tuple[int, int]:
        datatype = opt['datatype']
        if 'train' in datatype:
            return INFINITE, INFINITE
        elif 'valid' in datatype:
            return NUM_TEST, NUM_TEST
        elif 'test' in datatype:
            return NUM_TEST, NUM_TEST


class ChunkyUniqueSlowTeacher(ChunkyTeacher):
    """
    Unique examples that load slowly.
    """

    def load_from_chunk(self, chunk_idx: int):
        output = []
        for i in range(10):
            text = str(i + chunk_idx * 10)
            resp = str(i + chunk_idx * 10)
            output.append((text, resp))
            time.sleep(0.1)
        return output


class ShortFixedTeacher(FixedDialogCandidateTeacher):
    """
    Fixed Dialog Candidate teacher with only 10 training examples.
    """

    def __init__(self, opt: Opt, shared: dict = None):
        super().__init__(opt, shared, num_train=10, num_test=10)


class DefaultTeacher(CandidateTeacher):
    pass
