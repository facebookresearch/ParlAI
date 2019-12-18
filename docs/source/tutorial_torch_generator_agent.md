# Using Torch Generator Agent

**Authors**: Stephen Roller, Eric Smith

`parlai.core.torch_generator_agent.TorchGeneratorAgent` is an abstract parent class that provides functionality for building autoregressive generative models. Extending TorchGeneratorAgent requires your model conform to a strict interface, but then provides you rich functionality like beam search and sampling.


## Example Models

Two major models in ParlAI inherit from TorchGeneratorAgent: seq2seq and transformer. You can try one of these with the example below:

```
python examples/train_model -m transformer/generator -t convai2 -mf /tmp/testtransformer --beam-size 5 -bs 16
```

## Creating a Model

In order to write a generative model, your agent should extend `parlai.core.torch_generator_agent.TorchGeneratorAgent`. This parent class implements `train_step` and `eval_step`, so you only need to implement your model and instantiate it through `build_model`. However, your model should implement the `parlai.core.torch_generator_agent.TorchGeneratorModel` interface.


## Tutorial

This tutorial will walk you through creating a simple generative model, found at `parlai.agents.example_tga.agents`, that consists of an LSTM-based encoder and decoder.

### Extending `TorchGeneratorAgent`

A minimal `TorchGeneratorAgent` only needs to implement `build_model()`, but if you want to specify any command-line arguments, you'll need to add `add_cmdline_args()` as well. This method first adds flags for the agent's superclass and then adds a `--hidden-size` flag for the hidden dimension of the LSTMs.

In `build_model()`, instantiate our example model (defined below) by passing in the agent's dict <<< where this comes from >>> and the hidden size. You'll also need to add lines to optionally copy pre-existing token embeddings into the model's embedding module.

Altogether, our example agent is defined as follows:

```
import parlai.core.torch_generator_agent as tga


class ExampleTgaAgent(tga.TorchGeneratorAgent):

    @classmethod
    def add_cmdline_args(cls, argparser):
        super(ExampleTgaAgent, cls).add_cmdline_args(argparser)
        group = argparser.add_argument_group('Example TGA Agent')
        group.add_argument(
            '-hid', '--hidden-size', type=int, default=1024, help='Hidden size.'
        )

    def build_model(self):
        model = ExampleModel(self.dict, self.opt['hidden_size'])
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.embeddings.weight, self.opt['embedding_type']
            )
        return model
```

### Extending `TorchGeneratorModel`

<<<>>>

### Writing the encoder

<<<>>>

### Writing the decoder

<<<>>>

### Training

Finally, call training on the agent:

```
python examples/train_model.py -m example_tga \
    -mf /tmp/example_model \
    -t convai2 -bs 32 -eps 2 --truncate 128
```

You should get a perplexity of 139 on the validation set (which is equal to the test set for the ConvAI2 dataset), and <<< what else? Give commentary >>>
