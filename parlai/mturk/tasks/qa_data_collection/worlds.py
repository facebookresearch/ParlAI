# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld

class QADataCollectionOnboardWorld(MTurkOnboardWorld):
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = 'Welcome onboard!'
        self.mturk_agent.observe(ad)
        response = self.mturk_agent.act()
        self.episodeDone = True

class QADataCollectionWorld(MTurkTaskWorld):
    """
    World for recording a turker's question and answer given a context.
    Assumes the context is a random context from a given task, e.g.
    from SQuAD, CBT, etc.
    """

    collector_agent_id = 'QA Collector'

    def __init__(self, opt, task, mturk_agent):
        self.task = task
        self.mturk_agent = mturk_agent
        self.episodeDone = False
        self.turn_index = -1

    def parley(self):
        self.turn_index = (self.turn_index + 1) % 2; # Each turn starts from the QA Collector agent
        ad = { 'episode_done': False }
        ad['id'] = self.__class__.collector_agent_id

        if self.turn_index == 0:
            # At the first turn, the QA Collector agent provides the context and
            # prompts the turker to ask a question regarding the context

            # Get context from SQuAD teacher agent
            qa = self.task.act()
            context = '\n'.join(qa['text'].split('\n')[:-1])

            # Wrap the context with a prompt telling the turker what to do next
            ad['text'] = (context +
                        '\n\nPlease provide a question given this context.')

            self.mturk_agent.observe(validate(ad))
            self.question = self.mturk_agent.act() # Can log the turker's question here

        if self.turn_index == 1:
            # At the second turn, the QA Collector collects the turker's question from the first turn,
            # and then prompts the turker to provide the answer

            # A prompt telling the turker what to do next
            ad['text'] = 'Thanks. And what is the answer to your question?'

            ad['episode_done'] = True  # end of episode

            self.mturk_agent.observe(validate(ad))
            self.answer = self.mturk_agent.act() # Can log the turker's answer here

            self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        self.task.shutdown()
        self.mturk_agent.shutdown()

    def review_work(self):
        pass
