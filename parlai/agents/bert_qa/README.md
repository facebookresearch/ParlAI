BERT model for Question Answering (span extraction).
This module is composed of the BERT model with a linear layer on top of the sequence output that computes start_logits and end_logits.

## Example usage:

Fine-tune the bert-large-uncased model on SQuAD v1.1:
```bash
python examples/train_model.py -m bert_qa -t squad -mf /tmp/bertqa -bs 10 -tr 320 -lr 3e-5 --bert_model bert-large-uncased --do_lower_case -eps 2
```

