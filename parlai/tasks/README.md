The set of tasks in ParlAI.

Each directory contains a task or a set of related tasks.

The list of tasks can also be found in this file:
https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/task_list.py

or in the documentation, here:
http://www.parl.ai/docs/tasks.html


Each task folder contains:
- **build.py** file for setting up data for the task (downloading data, etc, only done the first time requested, and not downloaded if the task is not used).
- **agents.py** file which contains default or special teacher classes used by core.create_task to instantiate these classes from command-line arguments (if desired).
- **worlds.py** file can optionally be added for tasks that need to define new/complex environments.

To add your own task, see the [tutorial](http://www.parl.ai/docs/tutorial_task.html).
