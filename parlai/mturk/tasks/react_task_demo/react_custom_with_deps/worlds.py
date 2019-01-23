#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import threading


class AskerOnboardingWorld(MTurkOnboardWorld):
    """Example onboarding world. Sends a message from the world to the
    worker and then exits as complete after the worker uses the interface
    """
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = (
            "Welcome onboard! You'll be playing the role of the asker. Ask "
            "a question that can be answered with just a number. Send any "
            "message to continue."
        )
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True


class AnswererOnboardingWorld(MTurkOnboardWorld):
    """Example onboarding world. Sends a message from the world to the
    worker and then exits as complete after the worker uses the interface
    """
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = (
            "Welcome onboard! You'll be playing the role of the answerer. "
            "You'll be asked a question that should be answered with a number. "
            "Answer with something that makes sense. Enter any number to "
            "continue."
        )
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True


class EvaluatorOnboardingWorld(MTurkOnboardWorld):
    """Example onboarding world. Sends a message from the world to the
    worker and then exits as complete after the worker uses the interface
    """
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = (
            "Welcome onboard! You'll be playing the evaluator. You'll "
            "observe a series of three questions, and then you'll evaluate "
            "whether or not the exchange was accurate. Send an eval to begin."
        )
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True


class MultiRoleAgentWorld(MTurkTaskWorld):
    """
    World to demonstrate workers with assymetric roles. This task amounts
    to three rounds and then an evaluation step. It is purposefully created
    as a task to demo multiple views and has no other purpose.
    """

    collector_agent_id = 'Moderator'

    def __init__(self, opt, mturk_agents):
        self.mturk_agents = mturk_agents
        for agent in mturk_agents:
            if agent.demo_role == 'Asker':
                self.asker = agent
            elif agent.demo_role == 'Answerer':
                self.answerer = agent
            else:  # 'Evaluator'
                self.evaluator = agent
        self.episodeDone = False
        self.turns = 0
        self.questions = []
        self.answers = []
        self.accepted = None

    def parley(self):
        if self.turns == 0:
            # Instruction for evaluator
            ad = {
                'id': 'System',
                'text': "Please observe the chat for accuracy.",
            }
            self.evaluator.observe(ad)
        if self.turns < 3:
            # QA pairing
            ad = {
                'id': 'System',
                'text': "Please ask a question with a numeric answer.",
            }
            self.asker.observe(ad)
            question = self.asker.act()
            ad = {
                'id': 'System',
                'text': 'Please answer this question.',
            }
            self.answerer.observe(ad)
            self.answerer.observe(question)
            self.evaluator.observe(question)
            answer = self.answerer.act()
            self.evaluator.observe(answer)
            self.asker.observe(answer)
            self.questions.append(question)
            self.answers.append(answer)
            self.turns += 1
        else:
            # evaluate
            ad = {
                'id': 'System',
                'text': "Please provide your evaluation.",
            }
            self.evaluator.observe(ad)
            ad = {
                'id': 'System',
                'text': "Please wait for evaluation.",
            }
            self.answerer.observe(ad)
            self.asker.observe(ad)
            self.accepter = self.evaluator.act()
            self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        # Parallel shutdown of agents
        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent
        threads = []
        for agent in self.mturk_agents:
            t = threading.Thread(target=shutdown_agent, args=(agent,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def review_work(self):
        # Can review the work here to accept or reject it
        pass

    def get_custom_task_data(self):
        # brings important data together for the task, to later be used for
        # creating the dataset. If data requires pickling, put it in a field
        # called 'needs-pickle'.
        return {
            'questions': self.questions,
            'answers': self.answers,
            'evaluation': self.accepted,
        }
