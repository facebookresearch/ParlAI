# MemNN

This is a memory network agent that inherits from the TorchAgent class.
Read more about Memory Networks [here](https://arxiv.org/abs/1410.3916).


## Basic Examples

Train a memory network on the bAbi task.
```bash
parlai train_model -m memnn -t babi:task10k:1 -mf /tmp/memnn_babi.mdl
```
