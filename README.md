# ParlAI

ask @jase, @ahm, or @willfeng if you need help with anything.

This framework provides the base for a unified format for dialog research.

## Observations

All agents use the same format for passing data, an "observation" table with the following fields:

    text: a string containing any text provided from this agent (e.g. a question)
    labels: an iterable of strings containing the label for this example
    reward: a string containing any reward being passed from this agent to others
    candidates: an iterable of strings containing candidates for responding to this example
    done: specifies whether this example is the last in an episode (for episode-less data, can always set this to True)
    image: format TBD, what is the most useful? a vector? a lambda to retrieve the image?
    audio: format TBD, what is the most useful? a vector? a lambda to retrieve the audio?

Each of these fields are technically optional, depending on your dataset, though the 'text' field will most likely be used in nearly all exchanges.

For a fixed supervised learning dataset like bAbI, a typical exchange might be as follows:

```python
Teacher: {
    'text': 'Sam went to the kitchen\nPat gave Sam the milk\nWhere is the milk?',
    'labels': ['kitchen'],
    'candidates': ['hallway', 'kitchen', 'bathroom'],
    'done': False
}
Student: {
    'text': 'hallway'
}
Teacher: {
    'text': 'Sam went to the hallway\nPat went to the bathroom\nWhere is the milk?',
    'labels': ['hallway'],
    'reward': '0',
    'candidates': ['hallway', 'kitchen', 'bathroom'],
    'done': True
}
Student: {
    'text': 'hallway'
}
Teacher: {
    'reward': '1',
    ... # starts next episode
}
...
```

## Code

The code is set up into several main directories:

    agents: contains agents which can interact with the different tasks
    core: contains the primary code for the framework
    examples: contains a few basic examples of different loops (building dictionary, train/valid, displaying data)
    tasks: contains code for the different tasks available from within ParlAI

Each directory is described in more detail below, ordered by dependencies.

### Core

The core library contains the following files:

    agents.py: this file contains a few basic agents which can be extended by your own model
        Agent: base class for all other agents, implements the act() method which receives an observation table and returns a table in response
        Teacher: child of Agent, also implements the report method for returning metrics. Tasks implement the Teacher class
        MultiTaskTeacher: creates a set of teachers based on a "task string" passed to the Teacher, creating multiple teachers within it and alternating between them
        create_task_teacher: instantiate a teacher from a given task string (e.g. 'babi:task:1' or 'squad')
    build_data.py: basic utilities for setting up data for tasks. you can override if your filesystem needs different functionality.
    data.py: contains some default classes for fixed text datasets
        TextData: sets up observation tables with 'text', 'labels', 'reward', and/or 'candidates' fields
        HogwildTextData: does the same thing as TextData, but stores underlying data in a shared-memory array
    dialog.py: contains default classes for doing dialog with basic data
        DialogTeacher: default parent class which automatically select the regular or hogwild dialog teacher based on whether multithreading (multiprocessing) is desired
        \_RegularDialogTeacher: generic parent teacher which sets up data.TextData and produces observations from that data
        \_HogwildDialogTeacher: generic multiprocess parent teacher which sets up a shared data.HogwildTextData and produces observations from that data (with shared metrics)
    dict.py: contains code for building general NLP-style dictionaries from observations
        DictionaryAgent: agent which tracks the index and frequency of words in a dictionary, and can parse a sentence into indices into its dictionary or back
    fbdialog.py: contains a teacher class which implements a function setup_data which parses data in the FB Dialog data format
    params.py: uses argparse to interpret command line arguments for ParlAI
    thread_utils: utility classes/functions for use in Hogwild multithreading (multiprocessing)
        SharedTable: provides a lock-protected, shared-memory, dictionary-like interface for keeping track of metrics
    worlds.py: contains a set of basic worlds for tasks to take place inside
        World: base class for all other worlds, implements `parley`, `shutdown`, `__enter__`, and `__exit__`
        DialogPartnerWorld: default world for turn-based two-agent communication
        MultiAgentDialogWorld: round-robin turn-based agent communication for two or more agents
        HogwildWorld: default world for setting up a separate world for every thread when using multiple threads (processes)


### Agents

The agents directory contains agents that have been approved into the ParlAI framework for shared use.
Currently availabe within this directory:

    memnn: code for a memory network agent for the ParlAI tasks
    remote_agent: basic class for any agent connecting over ZMQ (memnn_luatorch_cpu uses this)
    repeat_label: basic class for merely repeating all data sent to it (e.g. for piping to a file, debugging)


### Examples

This directory contains a few particular examples of basic loops.

    build_dict.py: build a dictionary from a particular task provided on the command-line using core.dict.DictionaryAgent
    display_data.py: uses agent.repeat_label to display data from a particular task provided on the command-line
    memnn_luatorch_cpu: shows a few examples of training a memory network on a few datasets

### Tasks

See the tasks directory for the current list of available tasks.
Each task folder contains a teachers.py file which contains default or special teacher classes used by core.create_task_teacher to instantiate these classes from command-line arguments (if desired).
When applicable, these tasks also contain a build.py file for setting up these classes (downloading data, etc).

To add your own task:
- (optional) implement build.py to download any needed data
- implement teachers.py, with at least a DefaultTeacher (extending Teacher or one of its children)
    - if your data is in FB Dialog format, subclass FbDialogTeacher
    - if not...
        - if your data is text-based, you can use extend DialogTeacher and thus core.data.TextData, in which case you just need to write your own setup_data function which provides an iterable over the data according to the format described in core.data
        - if your data uses other fields, write your own act() method which provides observations from your task each time it's called
