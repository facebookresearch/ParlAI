#!/usr/bin/env

"""
These agents contain a number of "unit test" corpora, or
fake corpora that ensure models can learn simple behavior easily.
They are useful as unit tests for the basic models.

The corpora are all randomly, but deterministically generated
"""

from parlai.core.agents import Teacher
from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
import random
import itertools

VOCAB_SIZE = 10
EX_SIZE = 5
NUM_CANDIDATES = 20
NUM_TRAIN = 1000
NUM_TEST = 100


class UnitTestTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.opt = opt
        opt['datafile'] = opt['datatype'].split(':')[0]
        self.words = list(map(str, range(VOCAB_SIZE)))
        super().__init__(opt, shared)

    def setup_data(self, fold):
        # N words appearing in a random order
        self.rng = random.Random(42)
        full_corpus = [list(x) for x in itertools.permutations(self.words, EX_SIZE)]
        self.rng.shuffle(full_corpus)

        it = iter(full_corpus)
        self.train = list(itertools.islice(it, NUM_TRAIN))
        self.val = list(itertools.islice(it, NUM_TEST))
        self.test = list(itertools.islice(it, NUM_TEST))

        # check we have enough data
        assert (len(self.train) == NUM_TRAIN)
        assert (len(self.val) == NUM_TEST)
        assert (len(self.test) == NUM_TEST)

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

        for text in self.corpus:
            yield (text, [text]), True


class RepeatTeacher(UnitTestTeacher):
    pass


class CandidateTeacher(RepeatTeacher):
    pass


class DefaultTeacher(UnitTestTeacher):
    pass
