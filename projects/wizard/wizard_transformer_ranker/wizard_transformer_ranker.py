# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.transformer.transformer import TransformerRankerAgent
from parlai.core.torch_agent import TorchAgent

import numpy as np

class WizardTransformerRankerAgent(TransformerRankerAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(WizardTransformerRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Wizard Transformer Ranker Arguments')
        agent.add_argument(
            '--use-knowledge', type='bool', default=True,
            help='use knowledge field instead of personas'
        )
        agent.add_argument(
            '--knowledge-dropout', type=float, default=0.7,
            help='dropout some knowledge during training'
        )
        agent.add_argument(
            '--chosen-sentence', type='bool', default=False,
            help='instead of using all knowledge, use gold'
                 'label, i.e. the chosen sentence'
        )
        agent.add_argument(
            '--join-history-tok', type=str, default=' ',
            help='Join history lines with this token, defaults to newline'
        )
        agent.add_argument(
            '-lr', '--learningrate', type=float, default=0.0001,
            help='learning rate'
        )
        return agent

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        opt['candidates'] = 'batch'  # this needs to be made the default
        super().__init__(opt, shared)
        self.use_knowledge = opt.get('use_knowledge', False)
        if self.use_knowledge:
            self.opt['use_memories'] = True
        self.chosen_sentence = (opt.get('chosen_sentence', False) and
                                self.use_knowledge)
        self.knowledge_dropout = opt.get('knowledge_dropout', 0)
        # TODO: add knowledge dropout capability

    def vectorize_knowledge(self, observation):
        if not self.use_knowledge:
            observation['memory_vecs'] = []
            return observation
        if self.chosen_sentence:
            if observation.get('checked_sentence'):
                observation['memory_vecs'] = [
                    self._vectorize_text(
                        observation['checked_sentence'],
                        truncate=self.truncate
                    )
                ]
        else:
            if observation.get('knowledge'):
                observation['memory_vecs'] = []
                for line in observation['knowledge'].split('\n'):
                    keep = 1
                    if not observation.get('eval_labels'):
                        keep = np.random.binomial(1, 1 - self.knowledge_dropout)
                    if keep:
                        observation['memory_vecs'].append(
                            self._vectorize_text(
                                line,
                                truncate=self.truncate
                            )
                        )

        return observation

    def vectorize(self, obs, add_start=True, add_end=True, truncate=None,
                  split_lines=False):
        return TorchAgent.vectorize(self, obs, add_start, add_end, truncate,
                                    split_lines)

    def get_dialog_history(self, observation, reply=None,
                           add_person_tokens=False, add_p1_after_newln=False,
                           join_history_tok='\n'):
        return TorchAgent.get_dialog_history(self, observation, reply,
                                             add_person_tokens,
                                             add_p1_after_newln,
                                             join_history_tok)

    def observe(self, observation):
        """Process incoming message in preparation for producing a response.

        This includes remembering the past history of the conversation.
        """
        reply = self.last_reply(
            use_label=(self.opt.get('use_reply', 'label') == 'label'))
        self.observation = self.get_dialog_history(
            observation, reply=reply, add_person_tokens=self.add_person_tokens,
            add_p1_after_newln=self.opt.get('add_p1_after_newln', False),
            join_history_tok=self.opt.get('join_history_tok', ' '))
        self.observation = self.vectorize_knowledge(self.observation)
        return self.vectorize(self.observation, truncate=self.truncate)
