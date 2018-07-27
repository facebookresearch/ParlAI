from parlai.core.torch_agent import TorchAgent
from parlai.core.params import ParlaiParser

p = ParlaiParser()
TorchAgent.add_cmdline_args(p)
p.set_params(no_cuda=True, dict_file='/tmp/dict_convai2')
opt = p.parse_args(print_args=False)
agent = TorchAgent(opt)

obs = {'text': 'hello world what are you thinking about things today?', 'labels': ['woah there idk what you think you are doing but pull it together']}

import time

t = time.time()
for i in range(1500000):
	v1 = agent.vectorize(obs, truncate=5)
	#print(v1)
	#v2 = agent.vectorize(obs, truncate=None)
	#print(v2)
	#v3 = agent.vectorize(obs, truncate=100)
	#print(v3)
#import pdb; pdb.set_trace()
f = time.time()
print(round(f - t, 4))
