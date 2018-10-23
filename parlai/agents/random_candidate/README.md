# Random Candidate

This is an agent which chooses a random label candidate, to be used for debugging or as an extremely simple baseline.

If label candidates are available for the task, it chooses one randomly as a response. It randomly sorts the candidates to return in the `text_candidates` field, used for calculating hits@k metrics. If label candidates are not available, it just repeats the label.

## Basic Examples

Evaluate on bAbi by choosing a random candidate.
```bash
python examples/eval_model.py -m random_candidate -t babi
```
