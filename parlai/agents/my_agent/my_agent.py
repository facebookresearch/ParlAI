# filename: parlai/agents/my_agent/my_agent.py

from parlai.core.agents import Agent

class MyAgentAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        print('!!!!! My Agent INI!!!')
        if shared is None:
            # do any other special initializing here
            self.model = [1] #load_fairseq_magic_code() # load in your model here
        else:
            # put any other stuff you need to share across instantiations here
            self.model = shared['model']

        self.dialogue_history = ''

    def reset(self):
        super().reset()
        self.dialogue_history = ''
    
    def share(self):
        # put any other special reusing stuff in shared too
        shared = super().share()
        shared['model'] = self.model
        return shared

    def observe(self, observation):
        # your goal is to build up the string input to the model here
        self.dialogue_history += observation['text']
        return observation

    def act(self):
        # do all the actual work of converting self.dialogue history into
        # a tensor, forward it through the model, convert the output to a string
        return {
            'text': 'hisotry:' + str(self.dialogue_history)
        }
