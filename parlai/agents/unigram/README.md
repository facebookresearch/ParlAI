# Unigram Agent

Baseline model which predicts the $n$ most common unigrams regardless of input.

## Examples

    $ python -m parlai.scripts.train_model -m unigram -mf unigram.model -t convai2 -eps 1 --num-words 15
    ...
    [ running eval: valid ]
    [creating task(s): convai2]
    [loading fbdialog data:/private/home/roller/working/parlai/data/ConvAI2/valid_self_original.txt]
    valid:{'exs': 7801, 'accuracy': 0, 'f1': 0.2093, 'bleu': 1.857e-08}
    ...
