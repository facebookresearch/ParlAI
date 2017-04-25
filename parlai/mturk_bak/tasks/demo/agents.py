# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.core.agents import Agent
from .task_config import task_config

state_config = task_config['state_config']

class MTurkDemoAgent(Agent):

    def __init__(self, opt, shared=None):
        self.context = None
        self.response = None
        self.cur_state_id = 0
        self.id = task_config['bot_agent_id']

    def _check_precondition_and_change_state(self, new_message_agent_id):
        if self.cur_state_id+1 >= len(state_config): # if there is no next state, then return
            return
        if new_message_agent_id == state_config[self.cur_state_id+1]['precondition']:
            print("[State] " + state_config[self.cur_state_id]['state_name'] + " -> " + state_config[self.cur_state_id+1]['state_name'])
            print("")
            self.cur_state_id += 1

    def observe(self, obs):
        print('Bot '+str(self.id)+' received: ', obs)
        
        if self.cur_state_id == 0:  # initial_state
            context = obs['text']
            self.context = context
            self.response = None
        elif self.cur_state_id == 1:  # teacher_should_ask_question
            teacher_question = obs['text']
            # Get response from your bot
            self.response = '(This is my answer.)'
        elif self.cur_state_id == 2:  # student_should_answer_question
            self.response = None
        elif self.cur_state_id == 3:  # teacher_should_give_reward
            teacher_reward = obs['reward']
            self.response = None
        elif self.cur_state_id == 4:  # task_done
            self.response = None

        agent_id = obs['id']
        self._check_precondition_and_change_state(agent_id)

    def act(self):
        if self.response:
            print('Bot '+str(self.id)+' response: ', self.response)
        self._check_precondition_and_change_state(self.id)
        return self.response