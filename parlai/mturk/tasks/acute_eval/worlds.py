#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import threading


class EvalOnboardWorld(MTurkOnboardWorld):
    """Example onboarding world. Sends a message from the world to the
    worker and then exits as complete after the worker uses the interface
    """

    def parley(self):
        act = {}
        act['id'] = 'System'
        act['text'] = "Welcome!"
        act['task_data'] = {
            "conversations": [
                [
                    {
                        "speaker": "bot1",
                        "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
                    },
                    {
                        "speaker": "human",
                        "text": "Vivamus hendrerit vel lorem vel maximus. Nulla at eleifend urna. Ut vel orci tortor. Donec tempus efficitur sapien id consectetur. ",
                    },
                    {
                        "speaker": "bot1",
                        "text": "Pellentesque risus nisi, hendrerit vitae sem eu, interdum egestas lectus. Duis venenatis tellus non ante blandit mollis.",
                    },
                    {
                        "speaker": "human",
                        "text": "Etiam sodales nulla sed molestie sodales. ",
                    },
                    {
                        "speaker": "bot1",
                        "text": "Suspendisse sit amet pretium dui. In porttitor, massa tincidunt ultrices pharetra, diam erat ultrices eros, eu viverra leo turpis non turpis.",
                    },
                ],
                [
                    {
                        "speaker": "human",
                        "text": "In efficitur, velit placerat accumsan volutpat, mi ligula consectetur erat, et hendrerit est urna in est.",
                    },
                    {
                        "speaker": "bot2",
                        "text": "Duis scelerisque ligula id quam sollicitudin pharetra.",
                    },
                    {
                        "speaker": "human",
                        "text": "Etiam nisi elit, fermentum ac dolor ac, pretium placerat odio. Sed hendrerit tellus at turpis placerat, ornare commodo massa aliquet. Proin vel convallis nunc, a mollis ex. ",
                    },
                    {"speaker": "bot2", "text": "Nulla tristique massa sem."},
                ],
            ],
            "paired_eval": {
                "question_statement": "Which speaker is more interesting to you?",
                "speakers_to_eval": ["bot1", "bot2"],
            },
        }
        self.mturk_agent.observe(act)
        self.mturk_agent.act()
        self.episodeDone = True


class PairwiseEvalWorld(MTurkTaskWorld):
    ##TODO @margaretli description
    """
    """

    def __init__(self, opt, worker_agent, convid):
        self.opt = opt
        self.worker_agent = worker_agent
        self.agents = [self.worker_agent]
        self.episodeDone = False
        self.turns = 0
        self.qualities = []
        self.choices = []
        self.reasons = []
        self.convid = convid

    def parley(self):
        # TODO Parley
        act = {}
        act['id'] = 'System'
        act['text'] = "Welcome!"
        act['task_data'] = {
            "conversations": [
                [
                    {
                        "speaker": "bot1",
                        "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
                    },
                    {
                        "speaker": "human",
                        "text": "Vivamus hendrerit vel lorem vel maximus. Nulla at eleifend urna. Ut vel orci tortor. Donec tempus efficitur sapien id consectetur. ",
                    },
                    {
                        "speaker": "bot1",
                        "text": "Pellentesque risus nisi, hendrerit vitae sem eu, interdum egestas lectus. Duis venenatis tellus non ante blandit mollis.",
                    },
                    {
                        "speaker": "human",
                        "text": "Etiam sodales nulla sed molestie sodales. ",
                    },
                    {
                        "speaker": "bot1",
                        "text": "Suspendisse sit amet pretium dui. In porttitor, massa tincidunt ultrices pharetra, diam erat ultrices eros, eu viverra leo turpis non turpis.",
                    },
                ],
                [
                    {
                        "speaker": "human",
                        "text": "In efficitur, velit placerat accumsan volutpat, mi ligula consectetur erat, et hendrerit est urna in est.",
                    },
                    {
                        "speaker": "bot2",
                        "text": "Duis scelerisque ligula id quam sollicitudin pharetra.",
                    },
                    {
                        "speaker": "human",
                        "text": "Etiam nisi elit, fermentum ac dolor ac, pretium placerat odio. Sed hendrerit tellus at turpis placerat, ornare commodo massa aliquet. Proin vel convallis nunc, a mollis ex. ",
                    },
                    {"speaker": "bot2", "text": "Nulla tristique massa sem."},
                ],
            ],
            "paired_eval": {
                "question_statement": "Which speaker is more interesting to you?",
                "speakers_to_eval": ["bot1", "bot2"],
            },
        }
        self.worker_agent.observe(act)
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
        for agent in self.agents:
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
            # 'questions': self.questions,
            # 'answers': self.answers,
            # 'evaluation': self.accepted,
        }
