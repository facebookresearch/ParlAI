# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.core.agents import Agent
from .task_config import task_config

state_config = task_config['state_config']


class MTurkSquadEvalAgent(Agent):
    """
    MTurk agent for recording context as well as question and answer that the MTurk teacher provides.
    """
    def __init__(self, opt, shared=None):
        self.context = None
        self.response = None
        self.action = None
        self.cur_state_name = 'initial_state'
        self.id = task_config['bot_agent_id']

    def _check_precondition_and_change_state(self, new_message_agent_id, new_message_agent_action):
        if self.cur_state_name == 'task_done':
            return
        next_states = state_config[self.cur_state_name]['next_states']
        for next_state_ind in range(len(next_states)):
            next_state = next_states[next_state_ind]
            precondition_agent_id = next_state['precondition_agent_id']
            precondition_agent_action = next_state['precondition_agent_action']
            next_state_name = next_state['state_name']
            if new_message_agent_id == precondition_agent_id:
                if precondition_agent_action == 'any' or \
                    new_message_agent_action == precondition_agent_action or \
                    (next_state_ind == len(next_states)-1 and precondition_agent_action == 'else'):
                    print("[State] " + self.cur_state_name + " -> " + next_state_name)
                    print("")
                    self.cur_state_name = next_state_name
                    break

    def observe(self, obs):
        print('Bot '+str(self.id)+' received: ', obs)
        
        if self.cur_state_name == 'initial_state':
            context = obs['text']
            self.context = context
            print("Context: " + self.context)
            self.response = None
        elif self.cur_state_name == 'teacher_should_ask_question':
            teacher_question = obs['text']
            # Run your model to get the response
            print("Teacher Question: " + teacher_question)
            self.response = '(This is my answer.)'
        elif self.cur_state_name == 'student_should_answer_question':
            self.response = None
        elif self.cur_state_name == 'teacher_should_give_textual_feedback':
            teacher_textual_feedback = obs['text']
            self.response = None
        elif self.cur_state_name == 'teacher_should_give_reward':
            teacher_reward = obs['reward']
            self.response = None
        elif self.cur_state_name == 'teacher_should_provide_correct_answer':
            teacher_answer = obs['text']
            self.response = None
        elif self.cur_state_name == 'task_done':
            self.response = None

        agent_id = obs['id']
        agent_action = None
        if 'action' in obs:
            agent_action = obs['action']
        self._check_precondition_and_change_state(agent_id, agent_action)

    def act(self):
        if self.response:
            print('Bot '+str(self.id)+' response: ', self.response)
        if self.action:
            print('Bot '+str(self.id)+' action: ', self.action)
        self._check_precondition_and_change_state(self.id, self.action)
        return self.response, self.action