Transformer model for Question Answering (span extraction).

## Example usage:

Fine-tune the bert-large-uncased-whole-word-masking model on SQuAD v1.1:
```bash
python examples/train_model.py -m transformer_qa -t squad -mf /tmp/transformer_qa -bs 8 -tr 384 -lr 3e-5 --model_type bert --model_name_or_path bert-large-uncased-whole-word-masking --do_lower_case -eps 2
```

