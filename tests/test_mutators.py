#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.core.message import Message

CONTEXT = "All your base\n" "Are belong to us."

EXAMPLE1 = {
    'text': "Hi, my name is Stephen",
    'labels': ["Hello Stephen!"],
    'episode_done': False,
}

EXAMPLE2 = {
    'text': "What is your name?",
    'labels': ["My name is Emily."],
    'episode_done': True,
}
EXAMPLE3 = {
    'text': "Hello, I'm Emily.",
    'labels': ["Hi Emily. Nice to meet you."],
    'episode_done': False,
}

EXAMPLE4 = {
    'text': "What are you called?",
    'labels': ["I am called Stephen."],
    'episode_done': True,
}


class TestIntegrations(unittest.TestCase):
    """
    Test that mutators work with teachers.
    """

    def _run_through(self, task, mutators):
        pp = ParlaiParser(True, False)
        opt = pp.parse_kwargs(task=task, mutators=mutators)
        teacher = create_task_agent_from_taskname(opt)[0]
        outputs = []
        for _ in range(5):
            outputs.append(teacher.act())
        return outputs

    def test_example(self):
        for example in self._run_through('integration_tests', 'word_reverse'):
            assert "".join(reversed(example['text'])) == example['labels'][0]

    def test_episode(self):
        examples = self._run_through('integration_tests:multiturn', 'episode_reverse')
        examples = examples[:4]  # hardcoded for this teacher
        total = []
        for example in examples:
            total.append(example['text'])
        assert example['labels'][0] == ' '.join(reversed(total))

    def test_many_episode(self):
        examples = self._run_through('integration_tests:multiturn', 'flatten')
        for example in examples:
            texts = example['text'].split('\n')
            labels = example['labels'][0].split(' ')
            for i, l in enumerate(labels):
                assert texts[2 * i] == l


class TestSpecificMutators(unittest.TestCase):
    def _setup_data(self):
        yield Message(EXAMPLE1)
        yield Message(EXAMPLE2)
        yield Message(EXAMPLE3)
        yield Message(EXAMPLE4)

    def _setup_data_with_context(self):
        yield Message(self._add_context(EXAMPLE1))
        yield Message(EXAMPLE2)
        yield Message(self._add_context(EXAMPLE3))
        yield Message(EXAMPLE4)

    def _add_context(self, message):
        return {k: v if k != 'text' else CONTEXT + '\n' + v for k, v in message.items()}

    def _apply_mutator(self, mutator_class):
        opt = Opt()
        mutator = mutator_class(opt)
        mutated = mutator(self._setup_data())
        return list(mutated)

    def _apply_context_mutator(self, mutator_class):
        opt = Opt()
        mutator = mutator_class(opt)
        mutated = mutator(self._setup_data_with_context())
        return list(mutated)

    def _text_eq(self, ex1, ex2):
        """
        Return if the text field is equal.
        """
        return ex1['text'] == ex2['text']

    def test_context_shuffle(self):
        from parlai.mutators.context_shuffle import ContextShuffleMutator

        ex1, ex2, ex3, ex4 = self._apply_context_mutator(ContextShuffleMutator)

        ex1_lines = ex1['text'].split('\n')
        assert len(ex1_lines) == 3
        assert sorted(ex1_lines) == sorted(CONTEXT.split("\n") + [EXAMPLE1['text']])
        ex3_lines = ex3['text'].split('\n')
        assert len(ex3_lines) == 3
        assert sorted(ex3_lines) == sorted(CONTEXT.split("\n") + [EXAMPLE3['text']])

    def test_episode_reverse(self):
        from parlai.mutators.episode_reverse import EpisodeReverseMutator

        ex1, ex2, ex3, ex4 = self._apply_mutator(EpisodeReverseMutator)

        assert ex1['text'] == EXAMPLE2['text']
        assert ex2['text'] == EXAMPLE1['text']
        assert ex3['text'] == EXAMPLE4['text']
        assert ex4['text'] == EXAMPLE3['text']
        assert ex1['text'] == EXAMPLE2['text']

    def test_episode_shuffle(self):
        from parlai.mutators.episode_shuffle import EpisodeShuffleMutator

        ex1, ex2, ex3, ex4 = self._apply_mutator(EpisodeShuffleMutator)

        # check episode done is always set correctly
        assert not ex1['episode_done']
        assert ex2['episode_done']
        assert not ex3['episode_done']
        assert ex4['episode_done']

        # check there was a mutation
        assert self._text_eq(ex1, EXAMPLE1) or self._text_eq(ex2, EXAMPLE1)
        assert self._text_eq(ex2, EXAMPLE2) or self._text_eq(ex1, EXAMPLE2)
        assert not self._text_eq(ex1, ex2)

        assert self._text_eq(ex3, EXAMPLE3) or self._text_eq(ex4, EXAMPLE3)
        assert self._text_eq(ex4, EXAMPLE4) or self._text_eq(ex3, EXAMPLE4)
        assert not self._text_eq(ex3, ex4)

    def test_flatten(self):
        from parlai.mutators.flatten import FlattenMutator

        ex1, ex2, ex3, ex4 = self._apply_mutator(FlattenMutator)

        # check episode done is always set correctly
        assert ex1['episode_done']
        assert ex2['episode_done']
        assert ex3['episode_done']
        assert ex4['episode_done']

        # check there was a mutation
        assert ex1['text'] == "\n".join(e['text'] for e in [EXAMPLE1])
        assert ex2['text'] == "\n".join(
            [EXAMPLE1['text'], EXAMPLE1['labels'][0], EXAMPLE2['text']]
        )
        assert ex3['text'] == "\n".join(e['text'] for e in [EXAMPLE3])
        assert ex4['text'] == "\n".join(
            [EXAMPLE3['text'], EXAMPLE3['labels'][0], EXAMPLE4['text']]
        )

    def test_last_turn(self):
        from parlai.mutators.last_turn import LastTurnMutator

        ex1, ex2, ex3, ex4 = self._apply_mutator(LastTurnMutator)

        # check episode done is always set correctly
        assert ex1['episode_done']
        assert ex2['episode_done']
        assert ex3['episode_done']
        assert ex4['episode_done']

        # check there was a mutation
        assert ex1['text'] == EXAMPLE1['text']
        assert ex2['text'] == EXAMPLE2['text']
        assert ex3['text'] == EXAMPLE3['text']
        assert ex4['text'] == EXAMPLE4['text']

    def test_word_reverse(self):
        from parlai.mutators.word_reverse import WordReverseMutator

        ex1, ex2, ex3, ex4 = self._apply_mutator(WordReverseMutator)

        # check episode done is always set correctly
        assert not ex1['episode_done']
        assert ex2['episode_done']
        assert not ex3['episode_done']
        assert ex4['episode_done']

        # assert correct texts
        assert ex1['text'] == "Stephen is name my Hi,"
        assert ex2['text'] == "name? your is What"
        assert ex3['text'] == "Emily. I'm Hello,"
        assert ex4['text'] == "called? you are What"

    def test_word_shuffle(self):
        from parlai.mutators.word_shuffle import WordShuffleMutator

        ex1, ex2, ex3, ex4 = self._apply_mutator(WordShuffleMutator)

        # check episode done is always set correctly
        assert not ex1['episode_done']
        assert ex2['episode_done']
        assert not ex3['episode_done']
        assert ex4['episode_done']

        # check there was a mutation
        assert ex1 != EXAMPLE1
        assert ex2 != EXAMPLE2
        assert ex3 != EXAMPLE3
        assert ex4 != EXAMPLE4

        # check words are the same in each setting
        assert set(ex1['text'].split()) == set(EXAMPLE1['text'].split())
        assert set(ex2['text'].split()) == set(EXAMPLE2['text'].split())
        assert set(ex3['text'].split()) == set(EXAMPLE3['text'].split())
        assert set(ex4['text'].split()) == set(EXAMPLE4['text'].split())

    def test_msc_ltm_mutator(self):
        from parlai.tasks.msc.mutators import LongTermMemoryMutator

        ex1, ex2, ex3, ex4 = self._apply_mutator(LongTermMemoryMutator)

        assert (
            ex1['labels']
            == ex2['labels']
            == ex3['labels']
            == ex4['labels']
            == ['personal_knowledge']
        )


class TestMutatorStickiness(unittest.TestCase):
    """
    Test that mutations DO NOT stick with episode.
    """

    def test_not_sticky(self):
        pp = ParlaiParser(True, False)
        opt = pp.parse_kwargs(
            task='integration_tests:multiturn',
            mutators='flatten',
            datatype='train:ordered',
        )
        teacher = create_task_agent_from_taskname(opt)[0]
        first_epoch = []
        second_epoch = []
        for _ in range(teacher.num_examples()):
            first_epoch.append(teacher.act())
        teacher.reset()
        for _ in range(teacher.num_examples()):
            second_epoch.append(teacher.act())

        assert all(f == s for f, s in zip(first_epoch, second_epoch))


class TestUniqueness(unittest.TestCase):
    """
    Test that mutators cannot have duplicate names.
    """

    def test_uniqueness(self):
        from parlai.core.mutators import register_mutator, Mutator

        @register_mutator("test_unique_mutator")
        class Mutator1(Mutator):
            pass

        # don't freak out if we accidentally register the exact same class twice
        register_mutator("test_unique_mutator")(Mutator1)

        # but do demand uniqueness
        with self.assertRaises(NameError):

            @register_mutator("test_unique_mutator")
            class Mutator2(Mutator):
                pass
