..
  Copyright (c) Facebook, Inc. and its affiliates.
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.

Model Zoo
=========

This is a list of pretrained ParlAI models. Some are meant to be used as components in
larger systems, while others may be used by themselves.

Pretrained Embeddings
---------------------

Some models support using Pretrained Embeddings, via `torchtext
<https://github.com/pytorch/text>`_. As of writing, this includes:

* ``fasttext``: Fasttext vectors based on mixed corpora (`Mikolov et al., 2018 <https://fasttext.cc/docs/en/english-vectors.html>`_)
* ``fasttext_cc``: Fasttext vectors based on Common Crawl (`Mikolov et al., 2018 <https://fasttext.cc/docs/en/english-vectors.html>`_)
* ``glove``: Pretrained GLoVe vectors based on 840B Common Crawl (`Pennington et al., 2014 <https://nlp.stanford.edu/projects/glove/>`_)
* ``glove-twitter``: GLoVe vectors pretrained on 2B tweets (`Pennington et al., 2014 <https://nlp.stanford.edu/projects/glove/>`_)

Example invocation:

.. code-block:: shell

  python -m parlai.scripts.train_model -t convai2 -m seq2seq -emb fasttext_cc

.. include:: zoo_list.inc
