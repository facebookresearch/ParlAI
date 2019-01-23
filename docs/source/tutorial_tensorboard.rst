..
  Copyright (c) Facebook, Inc. and its affiliates.
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.

Using tensorboard for metric tracking
=====================================

ParlAI uses tensorboardX package which provides tensorflow-free api to write tensorboard event files.
One can install it using pip:
``pip install tensorboardX``

Default usage inside training loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TrainingLoop`` class from ``train_model.py`` script supports saving any metric available in ``train_report`` or ``valid_report``.

Provide the following arguments to track Perplexity, Loss and Accuracy (this is presented as arguments for parser.set_default() function):

.. code-block:: python

    tensorboard_log=True,
    tensorboard_tag='task,batchsize,hiddensize,embeddingsize,attention,numlayers,rnn_class,learningrate,dropout,gradient_clip',
    tensorboard_metrics='ppl,loss,accuracy',

``tensorboard_tag`` provides sequence of arguments which will be used together with corresponding values for the tensorboard event folder name.
In the example above, the folder name will look like this:
``May31_10-15_task-convai2:self_batchsize-64_hiddensize-1024_embeddingsize-300_attention-``
``general_numlayers-2_rnn_class-lstm_learningrate-3_dropout-0.1_gradient_clip-0.1``

All folders are stored in ``${PARLAI_DATA}/tensorboard``

In order to launch tensorboard with ParlAI logs, run:
``tensorboard —logdir ${PARLAI_DATA}/tensorboard --port 8866``. TB will be avaialable on port 8866.

Usage in other part your code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One can track any other values during the runtime with ``TensorboardLogger`` class:

.. code-block:: python

    from parlai.core.logs import TensorboardLogger

    # you need access to global parlai opt to create an instance
    if opt['tensorboard_log'] is True:
        self.writer = TensorboardLogger(opt)

    # then you can track any metric:
    self.writer.add_scalar('test', 100)
