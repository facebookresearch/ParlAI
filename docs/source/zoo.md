Model Zoo
=========

This is a list of pretrained ParlAI models. They are listed by task, or
else in a pretraining section (at the end) when meant to be used as
initialization for fine-tuning on a task.

```{include} zoo_list.inc
```

Pretrained Word Embeddings
--------------------------

Some models support using Pretrained Embeddings, via
[torchtext](https://github.com/pytorch/text). As of writing, this
includes:

-   `fasttext`: 300-dim Fasttext vectors based on mixed corpora ([Mikolov et al., 2018](https://fasttext.cc/docs/en/english-vectors.html))
-   `fasttext_cc`: 300-dim Fasttext vectors based on Common Crawl ([Mikolov et al., 2018](https://fasttext.cc/docs/en/english-vectors.html))
-   `glove`: 300-dim Pretrained GLoVe vectors based on 840B Common Crawl ([Pennington et al., 2014](https://nlp.stanford.edu/projects/glove/))

Example invocation:

```bash
parlai train_model -t convai2 -m seq2seq -emb fasttext_cc
```

Adding '-fixed' to the name e.g. 'twitter-fixed' means backprop will not
go through this (i.e. they will remain unchanged).

BERT
----

BERT is in the model zoo and is automatically used for initialization of
bert bi-, poly- and cross-encoder rankers.

Example invocation:

```bash
parlai train_model -t convai2 -m bert_ranker/bi_encoder_ranker
```
