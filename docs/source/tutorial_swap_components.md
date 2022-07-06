# Swapping Out Transformer Subcomponents

__Author__: Spencer Poff

Sometimes you find yourself wanting to experiment with an architecture that looks a lot like another, but with one component modified. If that component is buried deep within the model, this is not easily accomplished with subclassing without copying and pasting much of the original implementation.

To make this easier and avoid copypasta, we provide the `@swappable` decorator.

## Making a Module Swappable

Let's say you have an existing class, `TransformerLayer`, that uses a module that you'd like to modify, `TransformerFFN`. You can make that FFN swappable in two steps:

1. Decorate `TransformerLayer` with `@swappable`, passing in a name for the component you'd like to swap and its default class/constructor:
```python
@swappable(ffn=TransformerFFN)
class TransformerLayer(nn.Module):
    ...
```

2. At runtime, the class for ffn will be added to a property `swappables` of `TransformerLayer`. Replace your instantiation of `TransformerFFN` with a call to that constructor:

```python
self.feedforward = self.swappables.ffn(opt, ...)
```

That's it!

## Making the Swap

You can now replace `TransformerFFN` with whatever class or constructor you want before instantiating `TransformerLayer`:
```python
layer = TransformerLayer.with_components(ffn=NewCustomFFN)(opt, ...)
```

As long as `NewCustomFFN` has the same `__init__` and `forward` method signatures as `TransformerFFN`, everything should just work.

For examples, see:
- `parlai/agents/examples/transformer_variant.py`
- `projects/params_vs_compute/hash_ladder/hash_ladder.py`

## Composability

Since the swapping happens before instantiation, decorated components can be transparently composed. For example:
```python
model = TransformerGeneratorModel.with_components(
    encoder=TransformerEncoder.with_components(
        layer=TransformerEncoderLayer.with_components(
            self_attention=MultiHeadAttention,
            feedforward=TransformerFFN,
        )
    ),
    decoder=TransformerDecoder.with_components(
        layer=TransformerDecoderLayer.with_components(
            encoder_attention=MultiHeadAttention,
            self_attention=MultiHeadAttention,
            feedforward=TransformerFFN,
        )
    ),
)(opt=self.opt, dictionary=self.dict)
```

## Implementation

See `parlai/agents/transformer/modules/modular.py`