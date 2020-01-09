core.torch_agent
===================================

Torch Agent implements much of the boilerplate necessary for creating
a neural dialogue agent, so you can focus on modeling. Torch Agent limits its
functionality to maintaining dialogue history, transforming text into vectors of
indices, and loading/saving models. The user is required to implement their own
logic in methods like `train_step` and `eval_step`.

Torch Ranker Agent and Torch Generator Agent have more specialized stub
methods, and provide many rich features and benefits. Torch Ranker Agent
assumes your model ranks possible responses from a set of possible candidates,
and provides options around negative sampling, candidate sampling, and
large-scale candidate prediction. Torch Generator Agent assumes your model
generates utterances auto-regressively, and provides generic implementations of
beam search.

Torch Agent
-----------------------------------
.. automodule:: parlai.core.torch_agent
  :members:


Torch Ranker Agent
-----------------------------------
.. automodule:: parlai.core.torch_ranker_agent
  :members:


Torch Generator Agent
-----------------------------------
.. automodule:: parlai.core.torch_generator_agent
  :members:
