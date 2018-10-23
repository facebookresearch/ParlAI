# Starspace

This agent contains a simple implementation of the *starspace* algorithm, slightly adapted for dialogue. To learn more about the starspace algorithm, see [this](https://arxiv.org/abs/1709.03856) paper.


## Basic Examples

Train a starspace model on the "sentence SQuAD" [task](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/squad/agents.py).
```bash
python examples/train_model.py --task squad:sentence --model starspace -lr 0.01 -esz 512 -k 10 -mf /tmp/starspacesquad
```
