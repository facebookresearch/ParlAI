This directory contains example training and interactive scripts for some of
the models used in the paper [Personalizing Dialogue Agents: I have a dog, do
you have pets too?](https://arxiv.org/pdf/1801.07243.pdf). If you are interested in
the Mechanical Turk tasks used to collect the Persona-Chat dataset used in this paper, you can view them
[here](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/personachat).

Note that the dataset for the ConvAI2 competition is larger than the Persona-Chat dataset, so
we are computing new baselines for that competition. See the baselines for that competition
[here](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2).

## Examples

Look at the PersonaChat data:
```bash
python examples/display_data.py --task personachat --datatype train
```

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
