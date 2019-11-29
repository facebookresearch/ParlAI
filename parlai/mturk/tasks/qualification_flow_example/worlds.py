#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import parlai.mturk.core.mturk_utils as mturk_utils

import random


class QualificationFlowOnboardWorld(MTurkOnboardWorld):
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = (
            'This demo displays the functionality of using qualifications to '
            'filter the workers who are able to do your tasks. The first task '
            'you will get will check to see if you pass the bar that the task '
            'requires against a prepared test set. If you pass, the next task '
            'will be a real one rather than the test one.'
            '\n'
            'Send anything to get started.'
        )
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True


class QualificationFlowSoloWorld(MTurkTaskWorld):
    """
    World that asks a user 5 math questions, first from a test set if the user is
    entering for the first time, and then randomly for all subsequent times.

    Users who don't get enough correct in the test set are assigned a
    qualification that blocks them from completing more HITs during shutdown

    Demos functionality of filtering workers with just one running world.

    Similar results could be achieved by using two worlds where the first acts
    as just a filter and gives either a passing or failing qualification. The
    second would require the passing qualification. The first world could then
    be runnable using the --unique flag.
    """

    test_set = [
        ['What is 1+1?', '2'],
        ['What is 3+2?', '5'],
        ['What is 6+6?', '12'],
        ['What is 5-3?', '2'],
        ['What is 6*4?', '24'],
    ]

    collector_agent_id = 'System'

    def __init__(self, opt, mturk_agent, qualification_id, firstTime):
        self.mturk_agent = mturk_agent
        self.firstTime = firstTime
        if not firstTime:
            self.questions = self.generate_questions(5)
        else:
            self.questions = self.test_set
        self.episodeDone = False
        self.correct = 0
        self.curr_question = 0
        self.qualification_id = qualification_id
        self.opt = opt

    def generate_questions(self, num):
        questions = []
        for _ in range(num):
            num1 = random.randint(1, 20)
            num2 = random.randint(3, 16)
            questions.append(
                ['What is {} + {}?'.format(num1, num2), '{}'.format(num1 + num2)]
            )
        return questions

    def parley(self):
        if self.curr_question == len(self.questions):
            ad = {
                'episode_done': True,
                'id': self.__class__.collector_agent_id,
                'text': 'Thank you for your answers!',
            }
            self.mturk_agent.observe(validate(ad))
            self.episodeDone = True
        else:
            ad = {
                'episode_done': True,
                'id': self.__class__.collector_agent_id,
                'text': self.questions[self.curr_question][0],
            }
            self.mturk_agent.observe(validate(ad))
            answer = self.mturk_agent.act()
            if answer['text'] == self.questions[self.curr_question][1]:
                self.correct += 1
            self.curr_question += 1

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        """
        Here is where the filtering occurs.

        If a worker hasn't successfully answered all the questions correctly, they are
        given the qualification that marks that they should be blocked from this task.
        """
        if self.firstTime and self.correct != len(self.questions):
            mturk_utils.give_worker_qualification(
                self.mturk_agent.worker_id,
                self.qualification_id,
                is_sandbox=self.opt['is_sandbox'],
            )
        self.mturk_agent.shutdown()

    def review_work(self):
        pass
