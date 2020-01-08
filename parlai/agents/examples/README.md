# Examples

Contains example agents and models. 


## Basic Examples

Train an example generator model, described in the tutorial in `docs/source/tutorial_torch_generator_agent.md`:

```
python examples/train_model.py -m examples/seq2seq \
    -mf /tmp/example_model \
    -t convai2 -bs 32 -eps 2 --truncate 128
```
