This directory contains example training and interactive scripts for some of
the models used in the paper [Personalizing Dialogue Agents: I have a dog, do
you have pets too?](https://arxiv.org/pdf/1801.07243.pdf). Note that the
dataset for the ConvAI2 competition is larger than the Persona-Chat dataset, so
we will be computing new baselines for that competition soon.

## Examples
Interact with a pre-trained Key-Value Memory Net model trained on Persona-Chat
using persona 'self':
```bash
python projects/personachat/scripts/kvmemnn_interactive.py
```

Evaluate pre-trained profile memory model trained to optimize f1 metric on
Persona-Chat using persona 'self':
```bash
python projects/personachat/scripts/profilememory_eval_f1.py
```

Train a profile memory model on Persona-Chat using persona 'self':
```bash
bash projects/personachat/scripts/profilememory_train.sh
```
