Tips and Tricks
===================================

Here we list other miscellaneous useful tips of things you can do in ParlAI not listed elsewhere.


Command line tool
#################

ParlAI comes with a "super" command, that has all the other commands built in:

.. code-block:: bash

  $ parlai help
  ParlAI - Dialogue Research Platform
  usage: parlai [-h] COMMAND ...

  optional arguments:
    -h, --help               show this help message and exit

  Commands:

    display_data (dd)        Display data from a task
    display_model (dm)       Display model predictions.
    eval_model (em, eval)    Evaluate a model
    train_model (tm, train)  Train a model
    interactive (i)          Interactive chat with a model on the command line
    safe_interactive         Like interactive, but adds a safety filter
    self_chat                Generate self-chats of a model

This is often more convenient than running the scripts from the examples directory.

This command also supports autocompletion of commands and options in your bash
prompt.  You can enable this by running

.. code-block:: bash

   python -m pip install argcomplete

and then adding the following line to your .bashrc or equivalent:

.. code-block:: bash

  eval "$(register-python-argcomplete parlai)"



Multi-tasking with weighted tasks
#################################

If you want to train/eval/display with multiple tasks you can just use for example:

.. code-block:: bash

  parlai display_data -t personachat,squad -dt train

However, this will sample episodes equally from the two tasks (personachat and squad).
To sample squad 10x more often you can do:

.. code-block:: bash

  parlai display_data -t personachat,squad --multitask_weights 1,10 -dt train


Tasks with Parameters
#####################

Some tasks have their own flags. While these can be separately added on the command line, especially
when multi-tasking it is possible to group them with the task name itself.
If you are using the same task, but with two different sets of parameters this is the only way that
will work, otherwise the flags would be ambiguous and not associated with those tasks.
This can be done on the command line in the following way:

.. code-block:: bash

  parlai display_data -t light_dialog:light_label_type=speech,light_dialog:light_label_type=emote -dt train

That is, by adding a colon ":" followed by the flag name, an equals sign, and the value.
You can add multiple flags, all separated by ":".


Self-Chats
##########

Sometimes it is useful to generate models talking to themselves. You can do this with:

.. code-block:: python

   # Self-chatting Poly-Encoder model on ConvAI2
   python parlai/scripts/self_chat.py -mf zoo:pretrained_transformers/model_poly/model -t convai2:selfchat --inference topk -ne 10 --display-examples True -dt valid

The task set by '-t' (in the above case "convai2:selfchat") links to a parlAI world that handles the particular nature of interactions, see e.g. `here <https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/convai2/worlds.py#L98>`__ 
or `here <https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/wizard_of_wikipedia/worlds.py#L106>`__.
If the model does not need to run on a particular task you can also use:
  

.. code-block:: python

   # Self-chatting Poly-Encoder model on a generic task (so e.g., no ConvAI2 personas are input)
   python parlai/scripts/self_chat.py -mf zoo:pretrained_transformers/model_poly/model -t self_chat --inference topk -ne 10 --display-examples True -dt valid


Prettifying Display of Chats
############################

This handy script can prettify the display of json file of chats (sequences of parlai messages):

.. code-block:: python

   # Display conversation in HTML format.
   python parlai/scripts/convo_render.py projects/wizard_of_wikipedia/chat_example1.jsonl -o /tmp/chat.html 


Internal Agents, Tasks and More
###############################

You can create a private folder in ParlAI with your own custom agents and tasks,
create your own model zoo, and manage it all with a separate git repository.

For more detailed instructions and features, see the `README <http://github.com/facebookresearch/ParlAI/blob/master/example_parlai_internal>`_
