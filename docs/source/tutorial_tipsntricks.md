Tips and Tricks
===============

Here we list other miscellaneous useful tips of things you can do in
ParlAI not listed elsewhere.

Command line tool
-----------------

ParlAI comes with a "super" command, that has all the other commands
built in:

```bash
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
```

This is often more convenient than running the scripts from the examples
directory.

This command also supports autocompletion of commands and options in
your bash prompt. You can enable this by running

```bash
python -m pip install argcomplete
```

and then adding the following line to your .bashrc or equivalent:

```bash
eval "$(register-python-argcomplete parlai)"
```

Multi-tasking with weighted tasks
---------------------------------

If you want to train/eval/display with multiple tasks you can just use
for example:

```bash
parlai display_data -t personachat,squad -dt train
```

However, this will sample episodes equally from the two tasks
(personachat and squad). To sample squad 10x more often you can do:

```bash
parlai display_data -t personachat,squad --multitask_weights 1,10 -dt train
```

Tasks with Parameters
---------------------

Some tasks have their own flags. While these can be separately added on
the command line, especially when multi-tasking it is possible to group
them with the task name itself. If you are using the same task, but with
two different sets of parameters this is the only way that will work,
otherwise the flags would be ambiguous and not associated with those
tasks. This can be done on the command line in the following way:

```bash
parlai display_data -t light_dialog:light_label_type=speech,light_dialog:light_label_type=emote -dt train
```

That is, by adding a colon ":" followed by the flag name, an equals
sign, and the value. You can add multiple flags, all separated by ":".

Self-Chats
----------

Sometimes it is useful to generate models talking to themselves. You can
do this with:

```bash
# Self-chatting Poly-Encoder model on ConvAI2
parlai self_chat -mf zoo:pretrained_transformers/model_poly/model -t convai2 --inference topk --num-self-chats 10 --display-examples True -dt valid
```
will generate 10 selfchats between 2 poly-encoder models of the exact same model opt.

The task set by '-t' (in the above case "convai2") links to a
parlAI world that handles the particular nature of interactions, see
e.g.
[here](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/convai2/worlds.py#L98)
or
[here](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/wizard_of_wikipedia/worlds.py#L106).


Flags to set up the SelfChat world:

- `--num-self-chats`: the number of self-chats to generate (1 by default).
- `--selfchat-max-turns`: the number of self-chat turns (6 by default), including context turn, seeded-utterance turns. Some self-chat world includes context information (such as persona; Wizard of Wikipedia(WoW) topics) in addition to the model utterances. 
- `--selfchat-task`: whether to create a self-chat version of the task. If True (by default), it creates a [SelfChatWorld](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/self_chat/worlds.py#L52) that allows for loading contexts (see [])and openers that seed the self-chat.
- `--outfile` specify file to save self chat logs.
- `--save-format` specify the format to save self-chat logs in. Use `conversations` for jsonl format, or `parlai` for text format (default: conversations).
- `--partner-model-file` allows self chat to be performed between two different models. If so, set this flag to one model and `-mf` and for the second one.
- `--partner-opt-file` (optional) use this to define an opt file containing args to override for `--partner_model_file`. 

#### Self-Chat World
If the self-chat needs to run on a particular task which provides additional context to the selfchat, e.g. persona, topics, one can specify it with `-t {task_name}` which would create selfchat version of the task world, see SelfChatWorld in each world module `parlai.tasks.{task_name}.worlds`. 

The [base SelfChatWorld](https://github.com/facebookresearch/ParlAI/blob/selfchat_feature/parlai/tasks/self_chat/worlds.py#L52) consists of:
- ` contexts `: context information such as persona, topics, sometimes initial utterances.
- ` _opener`: list of seeded messages from the task.
- ` parley()`: [parley]https://github.com/facebookresearch/ParlAI/blob/selfchat_feature/parlai/tasks/self_chat/worlds.py#L116 logic for two agents self-chat with seeded contexts and/or utterances.

For example, to seed each agent in the selfchat with persona string and conversation openers from Blended Skill Talk (BST) valid dataset, run
```bash
parlai self_chat -mf zoo:pretrained_transformers/model_poly/model -t blended_skill_talk --display-examples True -dt valid
```
and each agent observe its own persona, a shared WoW topic(if there is any in the randomly chosen episode), and the initial utterance from the same episode.

Flags for setting up the SelfChatWorld:
- `-t`: name of the selfchat_task.
- `--seed-messages-from-task`: whether to seed the self-chat with first utterances from the task dataset with specified datatype (`train:evalmode` by default).


:::{warning} WARNING
To initialize a list of openers to seed the selfchat, the default method of [init_openers](https://github.com/facebookresearch/ParlAI/blob/dbae75bcbe9fd15691d2d724c5107d7489cac000/parlai/tasks/self_chat/worlds.py#L76-L82) skim through each episode of the task dataset and extract the first dialogue turn, which might itself contain context information, such as persona, in addition to the first dialogue messages. When using the `--seed-messages-from-task`, override the [load_openers](https://github.com/facebookresearch/ParlAI/blob/dbae75bcbe9fd15691d2d724c5107d7489cac000/parlai/tasks/self_chat/worlds.py#L17) in seed with the desired
:::

Additional flags for setting up the Customized SelfChatWorld, e.g. blended_skill_talk, convai2, etc. 

For example, 2 additional flags for Blended Skill Talk SelfChat World:
- ```--include-personas```: if True(by default), it will prepend the persona strings to the context each agent observes before the selfchat begins. 
- ```--include-initial-utterances```: if True (by default), it will prepend the initial utterances to the context each agent observes before the selfchat begins.
For example, the self chats evaluated in the [BlenderBot](https://parl.ai/projects/recipes/) paper were generated by
```bash
parlai self_chat -mf zoo:blender/blender_90M/model -t blended_skill_talk -dt valid --num-self-chats 200
```
that output 200 selfchats seeded with persona, WoW topic if any and initial utterances from BST. Notice here we concatenate first utterances with context data in `contexts`.

If the model does not need to run on a particular task you can also use:

```bash
# Self-chatting Poly-Encoder model on a generic task (so e.g., no ConvAI2 personas are input)
parlai self_chat -mf zoo:pretrained_transformers/model_poly/model --inference topk --num-self-chats 10 --display-examples True
```


Prettifying Display of Chats
----------------------------

This handy script can prettify the display of json file of chats
(sequences of parlai messages):

```bash
# Display conversation in HTML format.
python parlai/scripts/convo_render.py -i projects/wizard_of_wikipedia/chat_example1.jsonl -o /tmp/chat.html 
```

Additional flags can be used for convo-render.

- `--num-examples` the number of conversations to render from the json file (10 by default).

Internal Agents, Tasks and More
-------------------------------

You can create a private folder in ParlAI with your own custom agents
and tasks, create your own model zoo, and manage it all with a separate
git repository.

For more detailed instructions and features, see the
[README](http://github.com/facebookresearch/ParlAI/blob/master/example_parlai_internal)
