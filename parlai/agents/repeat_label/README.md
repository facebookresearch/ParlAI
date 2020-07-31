# Repeat Label

This is an agent which simply repeats the label for debugging purposes or as a baseline.

## Basic Examples

Evaluate on bAbi, returning the answer "I don't know" roughly 20% of the time.
```bash
parlai eval_model -m repeat_label -t babi --cant_answer_percent 0.2
```
