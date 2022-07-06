# QA data collection task

Mephisto task to have crowdsource workers read a passage and then write a question-and-answer pair about that passage. For instance, if the passage is as follows:

> At the time, there were many varying opinions about Christian doctrine, and no centralized way of enforcing orthodoxy. Constantine called all the Christian bishops throughout the Roman Empire to a meeting, and some 318 bishops (very few from the Western Empire) attended the First Council of Nicaea. *(passage continues)*

a crowdsource worker could supply the question "Who called the bishops to the First Council of Nicaea?" and the answer "Constantine called the bishops".

Some useful parameters to set:
- `mephisto.blueprint.task_description_file`: HTML describing the data collection task, shown on the left-hand pane of the chat window
- `mephisto.blueprint.world_file`: the path to the Python module containing the class definition for the chat World, used for setting the logic for each turn of the task, when to end the task, actions upon shutdown, etc. (The onboarding World, if it exists, will be defined in this module as well.) Modify this value if you would like to write your own World class without having to create a new Blueprint class.
- `mephisto.teacher.task`: the ParlAI dataset to pull passages from. Defaults to SQuAD
- `mephisto.teacher.datatype`: the fold of the dataset in `mephisto.teacher.task` to pull passages from. Defaults to the training set

**NOTE**: See [parlai/crowdsourcing/README.md](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/README.md) for general tips on running `parlai.crowdsourcing` tasks, such as how to specify your own YAML file of configuration settings, how to run tasks live, how to set parameters on the command line, etc.
