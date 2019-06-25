from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent

class InteractiveWorld(DialogPartnerWorld):

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print("[ loading personas.. ]")
        self.load_personas()


    def load_personas(self):
        # Create ConvAI2 data so we can assign personas.
        convai2_opt = self.opt.copy()
        convai2_opt['task'] = 'convai2:both'
        convai2_agent = RepeatLabelAgent(convai2_opt)
        self.convai2_world = create_task(convai2_opt, convai2_agent)
        self.cnt = 0
        
    def get_new_personas(self):
        # Find a new episode
        while True:
            self.convai2_world.parley()
            msg = self.convai2_world.get_acts()[0]
            if msg['episode_done']:
                self.convai2_world.parley()
                msg = self.convai2_world.get_acts()[0]
                break
        txt = msg.get('text', '').split('\n')
        a1_persona = ""
        a2_persona = ""
        for t in txt:
            if t.startswith("partner's persona:"):
                a1_persona += t + '\n'
                #print(t.replace("partner's ", 'your '))
            if t.startswith('your persona:'):
                a2_persona += t + '\n'
        print(a1_persona)
        print("--")
        print(a2_persona)
        print("Enter [DONE] if you want a new partner at any time.")
        return a1_persona, a2_persona

    
    def parley(self):
        """Agent 0 goes first. Alternate between the two agents."""
        print("parley!")
        if self.cnt == 0:
            self.p1, self.p2 = self.get_new_personas()

        acts = self.acts
        agents = self.agents
        acts[0] = agents[0].act()
        if self.cnt == 0:
            # add the persona on to the first message
            acts[0]['text'] = self.p2 + acts[0].get('text', 'hi') 
            print("gave bot its persona!")
        agents[1].observe(validate(acts[0]))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()
        self.cnt += 1

        if self.episode_done():
            print("CHAT DONE ")
            print("\n... preparing new chat... \n")
            self.cnt = 0

        
