# Example Seq2Seq

This is an example sequence to sequence agent featured in the ["Creating an Agent"](http://parl.ai/static/docs/tutorial_seq2seq.html) tutorial.
Please follow the tutorial for more information.

## Basic Examples
(These are also featured in the tutorial.)

Train a model with `num_threads == 1` and `batchsize == 32`.
```bash
python examples/train_model.py -t babi:task10k:1 --dict-file /tmp/dict_babi:task10k:1 -bs 32 -vtim 30 -vcut 0.95 -m example_seq2seq
```

Train a model with `num_threads == 40`, `batchsize == 1`, a lower learning rate, and without using CUDA.
```bash
python examples/train_model.py -t babi:task10k:1 --dict-file /tmp/dict_babi:task10k:1 -bs 1 -nt 40 -vtim 30 -vcut 0.95 -m example_seq2seq --no-cuda -lr 0.01
```
