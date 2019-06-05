Using Torch Generator Agent
===========================

**Author**: Stephen Roller

:py:class:`~parlai.core.torch_generator_agent.TorchGeneratorAgent` is an abstract
parent class that provides functionality for building autoregressive generative
models. Extending TorchGeneratorAgent requires your model conform to a strict
interface, but then provides you rich functionality like beam search and sampling.


Example Models
--------------

Two major models in ParlAI inheret from TorchGeneratorAgent: seq2seq and transformer.
You can try one of these with the example below:

.. code-block:: bash

   python examples/train_model -m transformer/generator -t convai2 -mf /tmp/testtransformer --beam-size 5 -bs 16


Creating a Model
----------------

In order to write a generative model, your agent should extend
:py:class:`~parlai.core.torch_generator_agent.TorchGeneratorAgent`. This parent
class implements ``train_step`` and ``eval_step``, so you only need to implement
your model. However, your model should implement the 
`~parlai.core.torch_generator_agent.TorchGeneratorModel` interface.

