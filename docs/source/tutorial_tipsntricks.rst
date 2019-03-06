..
  Copyright (c) Facebook, Inc. and its affiliates.
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.

Tips and Tricks
===================================

Here we list other miscellaneous useful tips of things you can do in ParlAI not listed elsewhere.



Multi-tasking with weighted tasks
#################################

If you want to train/eval/display with multiple tasks you can just use for example:

.. code-block:: bash

  python examples/display_data.py -t personachat,squad -dt train

However, this will sample episodes equally from the two tasks (personachat and squad).
To sample squad 10x more often you can do:

.. code-block:: bash

  python examples/display_data.py -t personachat,squad --multitask_weights 1,10 -dt train
