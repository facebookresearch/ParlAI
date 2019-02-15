#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
These agents contain a number of "unit test" corpora, or
fake corpora that ensure models can learn simple behavior easily.
They are useful as unit tests for the basic models.

The corpora are all randomly, but deterministically generated
"""

from parlai.core.teachers import DialogTeacher
import random
import itertools

# default parameters
VOCAB_SIZE = 7
EXAMPLE_SIZE = 4
NUM_CANDIDATES = 10
NUM_TRAIN = 500
NUM_TEST = 100


class CandidateTeacher(DialogTeacher):
    """
    Candidate teacher produces several candidates, one of which is a repeat
    of the input. A good ranker should easily identify the correct response.
    """
    def __init__(self, opt, shared=None,
                 vocab_size=VOCAB_SIZE,
                 example_size=EXAMPLE_SIZE,
                 num_candidates=NUM_CANDIDATES,
                 num_train=NUM_TRAIN,
                 num_test=NUM_TEST):
        """
        :param int vocab_size: size of the vocabulary
        :param int example_size: length of each example
        :param int num_candidates: number of label_candidates generated
        :param int num_train: size of the training set
        :param int num_test: size of the valid/test sets
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

    def num_episodes(self):
        if self.datafile == 'train':
            return self.num_train
        else:
            return self.num_test

    def num_examples(self):
        return self.num_episodes()

    def setup_data(self, fold):
        # N words appearing in a random order
        self.rng = random.Random(42)
        full_corpus = [
            list(x) for x in itertools.permutations(self.words, self.example_size)
        ]
        self.rng.shuffle(full_corpus)

        it = iter(full_corpus)
        self.train = list(itertools.islice(it, self.num_train))
        self.val = list(itertools.islice(it, self.num_test))
        self.test = list(itertools.islice(it, self.num_test))

        # check we have enough data
        assert (len(self.train) == self.num_train), len(self.train)
        assert (len(self.val) == self.num_test), len(self.val)
        assert (len(self.test) == self.num_test), len(self.test)

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

        for i, text in enumerate(self.corpus):
            cands = []
            for j in range(NUM_CANDIDATES):
                offset = (i + j) % len(self.corpus)
                cands.append(self.corpus[offset])
            yield (text, [text], 0, cands), True


class MultipassTeacher(CandidateTeacher):
    """
    Multiturn teacher, where each episode goes:

    call      response
    1         1
    2         1 2
    3         1 2 3
    4         1 2 3 4
    """
    def num_examples(self):
        return super().num_examples() * self.example_size

    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, a, _, cs), _ in raw:
            split_t = t.split(' ')
            ans = a[0]
            for i, bit in enumerate(split_t):
                label = ans[:2 * i + 1]
                cands = [c[:2 * i + 1] for c in cs]
                yield (bit, [label], 0, cands), i == 0


class MultiturnCandidateTeacher(CandidateTeacher):
    """Splits inputs/targets by spaces into multiple turns.

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
                    (split_t[i], [' '.join(split_a[:i + 1])], r,
                     [' '.join(c[:i + 1]) for c in split_cs]),
                    i == 0
                )


class NocandidateTeacher(CandidateTeacher):
    """
    Strips the candidates so the model can't see any options. Good for testing
    simple generative models.
    """
    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, a, _r, _c), e in raw:
            yield (t, a), e


class MultiturnNocandidateTeacher(MultiturnCandidateTeacher):
    """
    Strips the candidates so the model can't see any options. Good for testing
    simple generative models.
    """
    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, a, _r, _c), e in raw:
            yield (t, a), e


class DefaultTeacher(CandidateTeacher):
    pass
