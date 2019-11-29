#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.worlds import World, validate
import importlib


def get_task(opt):
    module_name = 'parlai.tasks.squad.agents'
    class_name = 'DefaultTeacher'
    my_module = importlib.import_module(module_name)
    task_class = getattr(my_module, class_name)
    task_opt = opt.copy()
    task_opt['datatype'] = 'train'
    task_opt['datapath'] = opt['datapath']
    return task_class(task_opt)


class QADataCollectionTaskWorld(World):
    """
    World for recording a person's question and answer given a context.

    Assumes the context is a random context from a given task, e.g. from SQuAD, CBT,
    etc.
    """

    collector_agent_id = 'QA Collector'

    def __init__(self, opt, task, agent):
        self.task = task
        self.agent = agent
        self.episodeDone = False
        self.turn_index = -1

    @staticmethod
    def generate_world(opt, agents):
        task = get_task(opt)
        return QADataCollectionTaskWorld(opt, task, agents[0])

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'Agent'

    def parley(self):
        # Each turn starts from the QA Collector agent
        self.turn_index = (self.turn_index + 1) % 2
        ad = {'episode_done': False}
        ad['id'] = self.__class__.collector_agent_id

        if self.turn_index == 0:
            # At the first turn, the QA Collector agent provides the context
            # and prompts the person to ask a question regarding the context

            # Get context from SQuAD teacher agent
            qa = self.task.act()
            context = '\n'.join(qa['text'].split('\n')[:-1])

            # Wrap the context with a prompt telling the person what to do next
            ad['text'] = context + '\n\nPlease provide a question given this context.'

            self.agent.observe(validate(ad))
            self.question = self.agent.act()
            while self.question is None:
                self.question = self.agent.act()
            # Can log the person's question here

        if self.turn_index == 1:
            # At the second turn, the QA Collector collects the person's
            # question from the first turn, and then prompts the
            # person to provide the answer

            # A prompt telling the person what to do next
            ad['text'] = 'Thanks. And what is the answer to your question?'

            ad['episode_done'] = True  # end of episode

            self.agent.observe(validate(ad))
            self.answer = self.agent.act()
            while self.answer is None:
                self.answer = self.agent.act()
            # Can log the person's answer here

            self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        self.task.shutdown()
        self.agent.shutdown()

    def review_work(self):
        pass
