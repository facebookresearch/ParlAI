# Examples

Contains example agents and models. 


## Basic Examples

Train an example generator model, described in the tutorial in `docs/source/tutorial_torch_generator_agent.md`:

```
parlai train_model -m examples/seq2seq \
    -mf /tmp/example_model \
    -t convai2 -bs 32 -eps 2 --truncate 128
```

Train an example ranking model, described in the tutorial in `docs/source/tutorial_torch_ranker_agent.rst` :

```
python  examples/train_model.py -m examples/tra  -t convai2 -mf /tmp/example_model -bs 32 -eps 2 --truncate 128
```