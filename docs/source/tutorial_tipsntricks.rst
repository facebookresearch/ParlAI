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
