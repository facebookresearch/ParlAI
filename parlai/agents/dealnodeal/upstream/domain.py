# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A collection of the implemented negotiation domain.
"""

import re


def get_domain(name):
    """Creates domain by name."""
    if name == 'object_division':
        return ObjectDivisionDomain()
    raise()


class Domain(object):
    """Domain interface."""
    def selection_length(self):
        """The length of the selection output."""
        pass

    def input_length(self):
        """The length of the context/input."""
        pass

    def generate_choices(self, ctx):
        """Generates all the possible valid choices based on the given context.

        ctx: a list of strings that represents a context for the negotiation.
        """
        pass

    def parse_context(self, ctx):
        """Parses a given context.

        ctx: a list of strings that represents a context for the negotiation.
        """
        pass

    def score(self, context, choice):
        """Scores the dialogue.

        context: the input of the dialogue.
        choice: the generated choice by an agent.
        """
        pass

    def parse_choice(self, choice):
        """Parses the generated choice.

        choice: a list of strings like 'itemX=Y'
        """
        pass

    def parse_human_choice(self, inpt, choice):
        """Parses human choices. It has extra validation that parse_choice.

        inpt: the context of the dialogue.
        choice: the generated choice by a human
        """
        pass

    def score_choices(self, choices, ctxs):
        """Scores choices.

        choices: agents choices.
        ctxs: agents contexes.
        """
        pass


class ObjectDivisionDomain(Domain):
    """Instance of the object division domain."""
    def __init__(self):
        self.item_pattern = re.compile('^item([0-9])=([0-9\-])+$')

    def selection_length(self):
        return 6

    def input_length(self):
        return 3

    def generate_choices(self, inpt):
        cnts, _ = self.parse_context(inpt)

        def gen(cnts, idx=0, choice=[]):
            if idx >= len(cnts):
                left_choice = ['item%d=%d' % (i, c) for i, c in enumerate(choice)]
                right_choice = ['item%d=%d' % (i, n - c) for i, (n, c) in enumerate(zip(cnts, choice))]
                return [left_choice + right_choice]
            choices = []
            for c in range(cnts[idx] + 1):
                choice.append(c)
                choices += gen(cnts, idx + 1, choice)
                choice.pop()
            return choices
        choices = gen(cnts)
        choices.append(['<no_agreement>'] * self.selection_length())
        choices.append(['<disconnect>'] * self.selection_length())
        return choices

    def parse_context(self, ctx):
        cnts = [int(n) for n in ctx[0::2]]
        vals = [int(v) for v in ctx[1::2]]
        return cnts, vals

    def score(self, context, choice):
        assert len(choice) == (self.selection_length())
        choice = choice[0:len(choice) // 2]
        if choice[0] == '<no_agreement>':
            return 0
        _, vals = self.parse_context(context)
        score = 0
        for i, (c, v) in enumerate(zip(choice, vals)):
            idx, cnt = self.parse_choice(c)
            # Verify that the item idx is correct
            assert idx == i
            score += cnt * v
        return score

    def parse_choice(self, choice):
        match = self.item_pattern.match(choice)
        assert match is not None, 'choice %s' % choice
        # Returns item idx and it's count
        return (int(match.groups()[0]), int(match.groups()[1]))

    def parse_human_choice(self, inpt, output):
        cnts = self.parse_context(inpt)[0]
        choice = [int(x) for x in output.strip().split()]

        if len(choice) != len(cnts):
            raise
        for x, n in zip(choice, cnts):
            if x < 0 or x > n:
                raise
        return ['item%d=%d' % (i, x) for i, x in enumerate(choice)]

    def _to_int(self, x):
        try:
            return int(x)
        except:
            return 0

    def score_choices(self, choices, ctxs):
        assert len(choices) == len(ctxs)
        cnts = [int(x) for x in ctxs[0][0::2]]
        agree, scores = True, [0 for _ in range(len(ctxs))]
        for i, n in enumerate(cnts):
            for agent_id, (choice, ctx) in enumerate(zip(choices, ctxs)):
                taken = self._to_int(choice[i][-1])
                n -= taken
                scores[agent_id] += int(ctx[2 * i + 1]) * taken
            agree = agree and (n == 0)
        return agree, scores
