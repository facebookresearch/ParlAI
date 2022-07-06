# Unigram Agent

Baseline model which predicts the $n$ most common unigrams regardless of input.

## Basic Examples

Training the unigram model on convai2:

```bash
parlai train_model -m unigram -mf unigram.model -t convai2 -eps 1 --num-words 15
```
