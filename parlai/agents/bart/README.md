# BART

BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

See https://arxiv.org/abs/1910.13461.

The BART agent can be instantiated as simply `-m bart`, however it is recommended to specify `--init-model zoo:bart/bart_large/model` or `-mf zoo:bart/bart_large/model` to ensure correct dictionaries are saved.


## Basic Examples

### Train BART on convai2.
```bash
parlai train_model -m bart -mf /tmp/model_file -t convai2 -bs 24 --fp16 true -eps 1 -lr 1e-5 --optimizer adam
```

or

```bash
parlai train_model -m bart --init-model zoo:bart/bart_large/model -mf /tmp/model_file -t convai2 -bs 24 --fp16 true -eps 1 -lr 1e-5 --optimizer adam
```

### Interact with a BART Model fine-tuned in fairseq

```bash
parlai interactive -m bart --init-fairseq-model /path/to/fairseq/model
```
