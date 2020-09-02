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

```python
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

### Using custom loss function

It is common in downstream tasks that people want to use a custom loss function on top of the original cross-entropy loss used to train the generative model. You can achieve this by overriding the `compute_loss` function. And a concrete example can be found [here](https://github.com/facebookresearch/ParlAI/blob/0c25349ac91abe2a5f232c63985aed14a37f5dcf/projects/wizard_of_wikipedia/generator/agents.py#L121).  Please note `compute_loss` changes its output type based on the `return_output` parameter. It either returns only the loss or returns the loss and whatever the model's `output.forward()` was. Be mindful of that. 

Let us walk though an example:

```python
    def compute_loss(self, batch, return_output=False):
        # first compute our regular forced decoding loss
        token_loss, model_output = super().compute_loss(batch, return_output=True)
        # compute our additional losses, you can replace this with your losses.
        notnull = batch.label_vec.ne(self.NULL_IDX)
        num_tokens = notnull.long().sum().item()

        encoder_states = model_output[2]
        ctx_know_attn = encoder_states[2]
        # knowledge_alpha is the loss weight between the decoding loss and custom losses
        if self.knowledge_alpha == 0.0:
            loss = token_loss
        else:
            # calculate additional losses, you can send your extra field
            # (like reward in RL tasks) in the batch
            # we will cover this later
            _, know_pred = ctx_know_attn.max(1)
            know_acc = (know_pred == batch.cs_ids).float().sum().item()
            know_chance = batch.ck_mask.sum(1).float().reciprocal().sum().item()
            # don't forget to log your losses into metrics
            self.metrics['know_chance'] += know_chance
            self.metrics['bsz'] += batch.text_vec.size(0)
            self.metrics['know_acc'] += know_acc
            know_loss = th.nn.functional.cross_entropy(
                ctx_know_attn, batch.cs_ids, reduction='mean'
            )
            self.metrics['know_loss'] += know_loss.item() * batch.text_vec.size(0)
            know_loss /= num_tokens
            # combine them
            loss = (
                1 - self.knowledge_alpha
            ) * token_loss + self.knowledge_alpha * know_loss
        if return_output:
            return (loss, model_output)
        else:
            return loss
```

### Adding more fields to the batch

Batch is a `namedtuple` containing data sent to an agent. This class is the input type of the train_step and eval_step functions. Agents can override the `batchify` function to return an extended namedtuple with additional fields if they would like. However, we recommend calling the parent function to set up these fields as a base. An example is:

```python
    def batchify(self, obs_batch):
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]

        checked_sentences = []
        for obs in reordered_observations:
            checked_sentence = '{} {} {}'.format(
                obs.get('title', ''), TOKEN_KNOWLEDGE, obs.get('checked_sentence', '')
            )
            checked_sentences.append(checked_sentence)

        batch['checked_sentence'] = checked_sentences
        return batch
```

### Adding tokens to dialogue history

In some cases, we want to add a fixed utterance to dialogue history. It can be a task token to mark a downstream fine-tuning task, or a special utterances for your own research purpose. You can simply override the `get_temp_history` function. Whatever specified there will be added to the end of your dialogue history. See [this](https://github.com/facebookresearch/ParlAI/blob/f10a7c5f9d681b6246df836c5998a2efccd928bb/parlai/core/torch_agent.py#L1670). An example of using it can be:

```python
    def get_temp_history(self, observation):
        return '\n' + self.opt['control']
```

This code adds your specified control string to the end of the dialogue history.
