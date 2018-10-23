# Language Model

The **language model** is an agent which trains an RNN on a language modeling task.
It is adapted from the language model featured in Pytorch's examples repo
[here](https://github.com/pytorch/examples/tree/master/word_language_model).


## Basic Examples

Train a language model with embedding size 128 and sequence length 30 on the Persona-Chat task.
```bash
python examples/train_model.py -m language_model -t personachat -esz 128 -sl 30 -mf /tmp/LM_personachat_test.mdl
```

After training, load and evaluate that model on the Persona-Chat test set.
```bash
python examples/eval_model.py -m language_model -t personachat -mf /tmp/LM_personachat_test.mdl -dt test
```
