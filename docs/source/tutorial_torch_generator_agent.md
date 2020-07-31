# Using Torch Generator Agent

**Authors**: Eric Smith

`parlai.core.torch_generator_agent.TorchGeneratorAgent` is an abstract parent class that provides functionality for building autoregressive generative models. Extending `TorchGeneratorAgent` requires your model conform to a strict interface, but then provides you rich functionality like beam search and sampling.

## Example Models

Two major models in ParlAI inherit from `TorchGeneratorAgent`: seq2seq and transformer. You can try the transformer with the example below:

```bash
parlai train_model -m transformer/generator \
  -t convai2 -mf /tmp/testtransformer \
  --beam-size 5 -bs 16
```

## Creating a Model

In order to write a generative model, your agent should extend `TorchGeneratorAgent`. This parent class implements `train_step` and `eval_step`, so you only need to implement your model and instantiate it through `build_model`. `TorchGeneratorAgent` will take care of many common generator features, such as forced decoding, beam search, n-gram beam blocking, top-k and top-p/nucleus sampling, etc.

Additionally, your model should implement the `TorchGeneratorModel` interface: see the tutorial below for an example of this.

## Tutorial

This tutorial will walk you through creating a simple generative model, found at `parlai.agents.examples.seq2seq`, that consists of a 1-layer-LSTM encoder and decoder.

### Extending `TorchGeneratorAgent`

Creating a generative model in ParlAI consists of subclassing `TorchGeneratorAgent` and subclassing `TorchGeneratorModel`. A minimal subclass of `TorchGeneratorAgent` only needs to implement `build_model()`, but if you want to specify any command-line arguments, you'll need to add `add_cmdline_args()` as well. Our implementation below first adds flags for `TorchGeneratorAgent` and then adds a `--hidden-size` flag for the hidden dimension of the LSTMs of the encoder and decoder.

In `build_model()`, we instantiate our example model (defined below) by passing in the agent's dict (set by `TorchAgent`) and the hidden size. We add lines to optionally copy pre-existing token embeddings into the model's embedding table.

Altogether, our example agent is defined as follows:

```python
import parlai.core.torch_generator_agent as tga


class Seq2seqAgent(tga.TorchGeneratorAgent):

    @classmethod
    def add_cmdline_args(cls, argparser):
        super(Seq2seqAgent, cls).add_cmdline_args(argparser)
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

We now subclass `TorchGeneratorModel` to create `ExampleModel`. We initialize this by first calling `super().__init__()` and passing in dictionary tokens for padding, start, end, and UNKs; we then create an embedding lookup table with `nn.Embedding` and instantiate the encoder and decoder, described in the following sections.

```python
import torch.nn as nn
import torch.nn.functional as F

class ExampleModel(tga.TorchGeneratorModel):

    def __init__(self, dictionary, hidden_size=1024):
        super().__init__(
            padding_idx=dictionary[dictionary.null_token],
            start_idx=dictionary[dictionary.start_token],
            end_idx=dictionary[dictionary.end_token],
            unknown_idx=dictionary[dictionary.unk_token],
        )
        self.embeddings = nn.Embedding(len(dictionary), hidden_size)
        self.encoder = Encoder(self.embeddings, hidden_size)
        self.decoder = Decoder(self.embeddings, hidden_size)
```

We next define a function to project the output of the decoder back into the token space:

```python
    def output(self, decoder_output):
        return F.linear(decoder_output, self.embeddings.weight)
```

Lastly, we define two functions to reindex the latent states of the encoder and decoder. For the encoder, the indices that we pass in index the samples in the batch, and for the decoder, the indices index the candidates that we want to retain for the next step of decoding (for instance, in beam search). We reindex the encoder at the very beginning of beam search and when ranking candidates during eval, and we reindex the decoder after each step of decoding. Since our encoder and decoder both are based on LSTMs, these encoder/decoder states are the hidden and cell states:

```python
    def reorder_encoder_states(self, encoder_states, indices):
        h, c = encoder_states
        return h[:, indices, :], c[:, indices, :]

    def reorder_decoder_incremental_state(self, incr_state, indices):
        h, c = incr_state
        return h[:, indices, :], c[:, indices, :]
```

### Creating the encoder

The encoder is straightfoward: it contains an embedding layer and an LSTM, and a forward pass through the encoder consists of passing the sequences of input tokens through both of them sequentially. The final hidden state is returned.

```python
class Encoder(nn.Module):

    def __init__(self, embeddings, hidden_size):
        super().__init__()
        _vocab_size, esz = embeddings.weight.shape
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=esz, hidden_size=hidden_size, num_layers=1, batch_first=True
        )

    def forward(self, input_tokens):
        embedded = self.embeddings(input_tokens)
        _output, hidden = self.lstm(embedded)
        return hidden
```

### Creating the decoder

The decoder is initialized in the same way as the encoder, but now the forward pass reflects the fact that the input tokens need to be passed through the embedder and LSTM one token at a time rather than all at once. If this is the first pass through the decoder, we pass a tuple `encoder_state` to the LSTM that consists of the initial hidden and cell state, as taken from the output of the encoder. If this is a subsequent pass through the decoder, the LSTM will have given us the current values of the hidden and cell states, so we pass that back in to the LSTM, after potentially having reindexed the states with `ExampleModel().reorder_decoder_incremental_state()`.

```
class Decoder(nn.Module):

    def __init__(self, embeddings, hidden_size):
        super().__init__()
        _vocab_size, self.esz = embeddings.weight.shape
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=self.esz, hidden_size=hidden_size, num_layers=1, batch_first=True
        )

    def forward(self, input, encoder_state, incr_state=None):
        embedded = self.embeddings(input)
        if incr_state is None:
            state = encoder_state
        else:
            state = incr_state
        output, incr_state = self.lstm(embedded, state)
        return output, incr_state
```


### Training

The full code for the agent can be seen [here](https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/examples/seq2seq.py). To call training:

```bash
parlai train_model -m examples/seq2seq \
    -mf /tmp/example_model \
    -t convai2 -bs 32 -eps 2 --truncate 128
```

You should get a perplexity of around 140 and a token accuracy of around 28% on the ConvAI2 validation/test set.
