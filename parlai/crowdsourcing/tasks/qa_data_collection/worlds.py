#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.crowdsourcing.utils.worlds import CrowdTaskWorld
from parlai.core.worlds import validate


class QADataCollectionWorld(CrowdTaskWorld):
    """
    World for recording a turker's question and answer given a context.

    Assumes the context is a random context from a given task, e.g. from SQuAD, CBT,
    etc.
    """

    collector_agent_id = 'QA Collector'

    def __init__(self, opt, agent):
        self.teacher = opt["teacher"]
        self.agent = agent
        self.agent.agent_id = "QA Agent"
        self.episodeDone = False
        self.context = None
        self.question = None
        self.answer = None
        self.opt = opt

    def parley(self):

        act = {'episode_done': False}
        act['id'] = self.__class__.collector_agent_id

        if not self.question:
            """
            First, the QA Collector agent provides the context and prompts the turker to
            ask a question regarding the context.
            """

            # Get context from dataloader
            passage = self.teacher.act()
            self.context = passage['text']
            act['passage'] = passage['text']

            # Add a prompt telling the turker what to do next
            act['text'] = 'Please provide a question given the passage.'
            self.agent.observe(validate(act))
            self.question = self.agent.act(timeout=self.opt["turn_timeout"])
            # Can log the turker's question here
            return

        if not self.answer:
            """
            Next, the QA Collector collects the turker's question, and then prompts the
            turker to provide the answer.
            """

            # A prompt telling the turker what to do next
            act['text'] = 'Thanks. And what is the answer to your question?'

            self.agent.observe(validate(act))
            self.answer = self.agent.act(timeout=self.opt["turn_timeout"])

            self.episodeDone = True

    def episode_done(self):
        return self.episodeDone


def make_world(opt, agents):
    return QADataCollectionWorld(opt, agents[0])


def get_world_params():
    return {"agent_count": 1}
