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
This will generate 10 self-chats between 2 poly-encoder models with persona context data from convai2.

Flags to generate and store the self-chat:

- `--num-self-chats` specify the number of self-chats to generate (1 by default).
- `--selfchat-max-turns` specify the number of self-chat turns (6 by default), including context turn, seeded-utterance turns. Some self-chat world includes context information (such as persona; Wizard of Wikipedia(WoW) topics) in addition to the model utterances. 
- `--selfchat-task` specify whether to create a self-chat version of the task. If True (by default), it allows for loading contexts and openers that seed the self-chat.
- `--outfile` specify file to save self-chat logs.
- `--save-format` specify the format to save self-chat logs in. Use `conversations` for jsonl format, or `parlai` for text format (`conversations` by default).
- `--partner-model-file` allows self-chat to be performed between two different models. If so, set this flag to one model and `-mf` for the second one.
- `--partner-opt-file` use this to define an opt file containing args to override for `--partner_model_file`.


Self-Chat World

If the self-chat needs additional context to start with, e.g. persona, topics, one can specify it with `-t <task_name>` (in the above case "convai2") which links to a ParlAI world in the task world module `parlai.tasks.{task_name}.worlds` that handles the particular nature of interactions, e.g.
[here](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/convai2/worlds.py#L98)
or
[here](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/wizard_of_wikipedia/worlds.py#L106).

The base [SelfChatWorld](https://github.com/facebookresearch/ParlAI/blob/selfchat_feature/parlai/tasks/self_chat/worlds.py#L52) consists of:
- ` contexts ` specify [context](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/self_chat/worlds.py#L135) information such as persona, topics, sometimes initial utterances.
- ` _opener` consists of [seeded messages](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/self_chat/worlds.py#L146) from the task.
- [` parley()`](https://github.com/facebookresearch/ParlAI/blob/selfchat_feature/parlai/tasks/self_chat/worlds.py#L116) handles the logic of two agents interacting with each other with additional seeded contexts and/or utterances.


Flags for setting up the SelfChatWorld:
- `-t`: name of the self-chat task.
- `--seed-messages-from-task`: whether to seed the self-chat with first utterances from the task dataset with specified datatype (`train:evalmode` by default).


:::{warning} WARNING
To initialize a list of openers to seed the self-chat, the default method of [init_openers](https://github.com/facebookresearch/ParlAI/blob/dbae75bcbe9fd15691d2d724c5107d7489cac000/parlai/tasks/self_chat/worlds.py#L76-L82) goes through each episode of the task dataset and extract the first dialogue turn, which might itself contain context information, such as persona, in addition to the first dialogue messages.
:::

Additional flags for setting up the task-specific SelfChatWorld, e.g. for Blended Skill Talk (BST) self-chat:
- ```--include-personas```: if `True` (by default), it will prepend the persona strings to the context each agent observes before the self-chat begins.
- ```--include-initial-utterances```: if `True` (by default), it will prepend the initial utterances to the context each agent observes before the self-chat begins.
For example, the self-chats evaluated in the [BlenderBot](https://parl.ai/projects/recipes/) paper were generated by
```bash
parlai self_chat -mf zoo:blender/blender_90M/model -t blended_skill_talk -dt valid --num-self-chats 200
```
which output 200 self-chats where each agent observe its own persona, a shared WoW topic if any and initial utterances from a BST episode.

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

Some additional flags that can be used for convo-render:

- `--num-examples` the number of conversations to render from the json file (10 by default).

Internal Agents, Tasks and More
-------------------------------

You can create a private folder in ParlAI with your own custom agents
and tasks, create your own model zoo, and manage it all with a separate
git repository.

For more detailed instructions and features, see the
[README](http://github.com/facebookresearch/ParlAI/blob/master/example_parlai_internal)

Contributing
------------

ParlAI is maintained by a small team, so we rely heavily on community contributions. We welcome pull requests with open wings!

### Overview
- Make your changes
- Open a pull request
- Iterate with reviewers, fix CI, add tests
- Celebrate!

### Creating a Pull Request

**Selecting reviewers**

There’s no master gatekeeper for ParlAI, so don’t worry about needing to add any particular person. Just take a look at the blame for the file you’re modifying and try to find someone who has made a significant and recent change to that code. Also, add people who might be affected by your change.

Some members of the team regularly go through the PR queue and triage, so if there’s a better person to review they’ll reassign to them.

**Work in progress**

If you are not ready for the pull request to be reviewed, keep it as a draft or tag it with “[WIP]” and we’ll ignore it temporarily. If you are working on a complex change, it’s good to start things off as WIP, because you will need to spend time looking at CI results to see if things worked out or not.

### Reviewing and Testing

**Iterating**

We’ll try our best to respond as quickly as possible, and will try to minimize the number of review roundtrips. Make sure to investigate lint errors and CI failures, as these should all be passing before merging. Keep iterating with us until it’s accepted!

**Testing**

See the [documentation on writing tests](http://parl.ai/docs/tutorial_tests.html)

**Merging**

Once a pull request is accepted, CI is passing, and there are no unresolved comments there is nothing else you need to do; we will merge the PR for you. 

### Celebrate
```bash
parlai party
```
