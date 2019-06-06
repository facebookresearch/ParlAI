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


Tasks with Parameters
#####################

Some tasks have their own flags. While these can be separately added on the command line, especially
when multi-tasking it is possible to group them with the task name itself.
If you are using the same task, but with two different sets of parameters this is the only way that
will work, otherwise the flags would be ambiguous and not associated with those tasks.
This can be done on the command line in the following way:

.. code-block:: bash

  python examples/display_data.py -t light_dialog:light_label_type=speech,light_dialog:light_label_type=emote -dt train

That is, by adding a colon ":" followed by the flag name, an equals sign, and the value.
You can add multiple flags, all separated by ":".
