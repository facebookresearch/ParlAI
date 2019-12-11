# ParlAI Projects

Here we list projects undertaken in the ParlAI framework that are shared publicly, either in the form of papers, public tasks (with leaderboards) and/or shared model code.

This directory also contains subfolders for some of the projects which are housed in the ParlAI repo, others are maintained via external websites. Please also refer to ParlAI's [agents](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks), [tasks](https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents) and [model zoo](https://github.com/facebookresearch/ParlAI/tree/master/parlai/zoo) for what else is in ParlAI.

## Generative Models

- **Unlikelihood Training for Consistent Dialogue** [[project]](https://parl.ai/projects/dialogue_unlikelihood/).
  _Methods to reduce copies & repeats, correct vocab usage, and avoiding contradiction via unlikelihood training._
  
- **What makes a good conversation? How controllable attributes affect human judgments** [[website]](https://github.com/facebookresearch/ParlAI/tree/master/projects/controllable_dialogue) [[paper]](https://arxiv.org/abs/1902.08654).
  _Optimizing for multi-turn engaging conversations -- by controlling question-asking, specificity, response-relatedness and repetition._

- **Retrieve and Refine** [[paper]](https://arxiv.org/abs/1808.04776).
  _Models for improved chitchat ability by combining retrieval with generative refinement._

- **Importance of Search Strategy** [[paper]](https://arxiv.org/abs/1811.00907).
  _Analysis of the performance of search in generative models for chitchat tasks._
 

## Retrieval Models
- **Poly-Encoders** [[project]](https://parl.ai/projects/polyencoder/) [[paper]](https://arxiv.org/abs/1905.01969).
  _State-of-the-art Transformer architectures + pretraining for dialogue retrieval._


## Interactive Learning

- **Self-Feeding Chatbot** [[paper]](https://arxiv.org/abs/1901.05415)
  _How an agent can learn from dialogue after deployment by imitating and asking for feedback._  
  
- **Beat-The-Bot Live Game** [[project]](https://parl.ai/projects/beat_the_bot/)
  _A new data collection and model evaluation tool, a Messenger-based Chatbot game called Beat the Bot._  
  
  

## Chit-chat

- **_dodeca_ Dialogue** [[project]](https://parl.ai/projects/dodecadialogue/).
  _Set of 12 (existing) tasks for building an agent that can see and talk. We build a strong baseline system with SOTA on many tasks._

- **Dialogue Natural Language Inference** [[external website]](https://wellecks.github.io/dialogue_nli/).
  _Task and method for improving dialogue consistency._

- **Empathetic Dialogues** [[paper]](https://arxiv.org/abs/1811.00207) [[external website]](https://github.com/facebookresearch/EmpatheticDialogues) [[video]](https://ai.facebook.com/blog/making-conversation-models-more-empathetic/).
_Task & models for chitchat displaying empathy._

- **ConvAI2 Competition** [[external website]](http://convai.io/).
_Competition on dialogue chitchat based on the PersonaChat task._

- **Persona-Chat** [[project]](https://github.com/facebookresearch/ParlAI/tree/master/projects/personachat).
_Task & models for chitchat with a given persona._


## Well-Behaved

- **Dialogue Safety** [[project]](https://parl.ai/projects/dialogue_safety/) [[paper]](https://arxiv.org/abs/1908.06083).
  _Task and method for improving the detection of offensive language in the context of dialogue._

- **Mitigating Genderation Bias** [[project]](https://parl.ai/projects/genderation_bias/).
  _Analysis and methods for mitigating gender bias in dialogue generation, using LIGHT as a testbed._

## Knowledge Grounded

- **Wizard of Wikipedia** [[project]](http://parl.ai/projects/wizard_of_wikipedia/) [[paper]](https://openreview.net/forum?id=r1l73iRqKm).
  _Knowledge-grounded open domain chitchat task & models._

## Visually Grounded

- **Image Chat** [[paper]](https://klshuster.github.io/image_chat/) [[task]](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/image_chat).
  _Task for personality-based engaging dialogue on images._

- **Personality-Captions** [[project]](http://parl.ai/projects/personality_captions/) [[paper]](https://arxiv.org/abs/1810.10665).
  _Task for personality-based engaging comments on images._

## Environment Grounded

- **LIGHT** [[project]](http://parl.ai/projects/light/)
_A large-scale text adventure game research platform for agents that speak and act._

- **Mastering the Dungeon** [[project]](https://github.com/facebookresearch/ParlAI/tree/master/projects/mastering_the_dungeon).
_Task & models for training grounded agents in a text adventure game via MTurk._

- **Talk The Walk** [[paper]](https://arxiv.org/abs/1807.03367).
_Task & models for grounded dialogue for the task of navigating New York City streets._

## QA

- **HotPotQA** [[external website]](https://hotpotqa.github.io/).
_QA task with multi-hop reasoning. Task built with ParlAI Mturk._

- **CoQA** [[external website]](https://stanfordnlp.github.io/coqa/).
_QA task with a series of interconnected questions. Task built with ParlAI Mturk._

- **DrQA** [[parlai agent]](https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa) [[project]](https://github.com/facebookresearch/ParlAI/tree/master/projects/drqa) [[external website]](https://github.com/facebookresearch/DrQA) [[paper]](https://arxiv.org/abs/1704.00051).
_QA model for answering questions by retrieving and reading knowledge._
