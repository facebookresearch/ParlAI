..
  Copyright (c) Facebook, Inc. and its affiliates.
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.

Using Torch Ranker Agent
========================
**Authors**: Emily Dinan

TorchRankerAgent is an abstract parent class for PyTorch models that rank
possible responses from a set of possible candidates. It inherits from
TorchAgent and contains boilerplate code for training and evaluating ranking
models.

Example Models
--------------

Several existing models in ParlAI inherit from TorchRankerAgent. Try some of
the examples below:

Train a Transformer Ranker model on ConvAI2:

.. code-block:: bash

    python examples/train_model.py -m transformer/ranker -t convai2 -mf /tmp/tr_convai2_test

Train a Memory Network model on Daily Dialog:

.. code-block:: bash

    python examples/train_model.py -m memnn -t dailydialog -mf /tmp/memnn_dd_test -bs 20 -cands batch -ecands batch

Train a BERT-based Bi-Encoder ranker model on Twitter:

.. code-block:: bash

    python examples/train_model.py -m bert_ranker/bi_encoder_ranker -t twitter -mf /tmp/bert_twitter_test -bs 10 -cands batch -ecands batch --data-parallel True


Creating a Model
----------------

In order to write a ranking model that inherits from TorchRankerAgent, you
must implement the following functions:

.. code-block:: python

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """This function takes in a Batch object as well as a Tensor of
        candidate vectors. It must return a list of scores corresponding to
        the likelihood that the candidate vector at that index is the
        proper response. If `cand_encs` is not None (when we cache the
        encoding of the candidate vectors), you may use these instead of
        calling self.model on `cand_vecs`.
        """
        pass

    def build_model(self):
        """This function is required to build the model and assign to the
        object `self.model`.
        """
        pass

Training a Model
----------------

Setting ``--candidates``
^^^^^^^^^^^^^^^^^^^^^^^^
This flag is used to determine which candidates to rank during training.
There are several options:

Possible sources of candidates:

1. **batch:**
  Use all labels in the batch as the candidate set (with all but the
  example's label being treated as negatives).
  Note: with this setting, the candidate set is identical for all
  examples in a batch. This option may be undesirable if it is possible
  for duplicate labels to occur in a batch, since the second instance of
  the correct label will be treated as a negative.
2. **batch-all-cands:**
  Use all inline candidates in the batch as candidate set.
  Note 1: This can result in a very large number of candidates.
  Note 2: In this case we will deduplicate candidates.
  Note 3: just like with 'batch' the candidate set is identical for all
  examples in a batch.
3. **inline:**
  If each example comes with a list of possible label candidates, use those.
  Each teacher act for the task should contain the field 'label_candidates'.
  Note: With this setting, each example will have its own candidate set.
4. **fixed:**
  Use a global candidates list, provided by the user.
  If self.fixed_candidates is not None, use a set of fixed candidates for
  all examples.
  Note: this setting is not recommended for training unless the
  universe of possible candidates is very small.
  To use this, add the path to your text file with the candidates to the
  flag ``--fixed-candidates-path`` or ``-fcp``.


Tracking ranking metrics
^^^^^^^^^^^^^^^^^^^^^^^^

During training, we omit some ranking metrics (like hits@k) for the sake of
speed. To get these ranking metrics, use the flag ``--train-predict  True``.


Evaluating a Model
------------------

Evaluating on a fixed candidate set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As during training, you must add the path to your text file with the
candidates to the flag ``--fixed-candidates-path`` or ``-fcp``.
For many models, it's convenient to cache the the encoding of the candidates
in the case that the encoding is independent of the context. In order to do
this and save to a file, set the flag ``--encode-candidate-vecs True``. In
order to do this, you must implement the function ``encode_candidates()``
which takes in a batch of padded candidates and outputs a batch of candidates
encoded with the model.


Evaluating on "vocab" candidates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the options above for evaluating a model, we also have the
option of evaluating "vocab" candidates. This is one global candidate list,
extracted from the vocabulary with the exception of ``self.NULL_IDX``.
