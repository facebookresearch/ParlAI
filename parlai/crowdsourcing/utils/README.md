# Crowdsourcing utilities

## Overview
- `acceptability.py`: Used to ensure that a worker's messages throughout a conversation meet certain criteria (not too short, not all caps, not a lot of repetition, safety, etc.). More details about the acceptability checker can be found in the `acceptability.AcceptabilityChecker` section below.
- `analysis.py`: Abstract base classes for compiling the results of crowdsourcing runs. See `analysis.py` section below.
- `frontend.py`: Method for compiling the frontend code of crowdsourcing tasks.
- `mturk.py`: Code for soft-blocking MTurk crowdsourcing workers (preventing them from working on this specific task), as well as a Hydra flag to pass in lists of workers to soft-block.
- `tests.py`: Abstract base classes for testing different categories of crowdsourcing tasks.
- `worlds.py`: Abstract base classes for onboarding and chat worlds.

## `acceptability.AcceptabilityChecker`

### How to add a new check
- Add the code for this in `.check_messages()`, inside a `if 'check_name' in violation_types:` condition
- Add the name of the check to `self.ALL_VIOLATION_TYPES`; otherwise, this check will not be recognized, and an error will be raised if the check is specified when calling `.check_messages()`!
- To use the check: add the name of the check to the `violation_types` arg when calling `.check_messages()`

## `analysis.py`

Contains abstract classes that provide the basic functionalities for compiling data from a Mephsito task.
Mephisto provides two interfaces for retrieving its crowdsourced data; `MephistoDB` and `DataBrowser`.
Using `AbstractResultsCompiler` you do not need to directly interact with these two modules---it provides an abstraction on top of these two.
This class has methods such as `get_task_data` and `get_task_units` which handles interacting with Mephisto abstractions.
For compiling your dataset from your crowdsourced Mephisto task, you need to extend this class and implement the following methods:

* `compile_results` that returns a python *dictionary* (key-value pairs) or a pandas *dataframe*. We assume that, each unit of the crowdsourcing task (for example, annotation or conversation) has a unique id.
In the json format, this id is the key for the entry that keeps dialogue data for that conversation.
If the format is a dataframe, the convention is to have each row of the dataframe keep the data for a single utterance (interaction). Hence, the conversation id needs to be stored in a column for distinguishing the data from different dialogues.

* (optional) `is_unit_acceptable` helps with simple filtering and data clean up. It receives the data from a unit of work and returns a boolean. We discard this unit if it returns `False`.

### Example
Imagine you have a Mephisto task that the output of each unit of its work looks like this:
```.python
{
    'ID': 1234,
    'favorite_flower': 'rose',
    'favorite_season', 'winter',
    'status': 'accepted'
}
```

Let's say this task is called `flowers_crowdsourcing` and want to discard every participant with `status` being "accepted".
Here is how you can have your data compiled and saved:

```.python
from parlai.crowdsourcing.utils.analysis import AbstractResultsCompiler

class MyResultsCompiler(AbstractResultsCompiler):
    def is_unit_acceptable(self, unit_data):
        return unit_data['status'] ==  'accepted'

    def compile_results(self):
        data = dict()
        for work_unit in self.get_task_data():
            unit_id = work_unit.pop('ID')
            data[unit_id] = work_unit
        return data


if __name__ == '__main__':
    opt = {
        'task_name': 'flowers_crowdsourcing',
        'results_format': 'json',
        'output_folder': '/path/dataset/',
    }
    wizard_data_compiler = MyResultsCompiler(opt)
    wizard_data_compiler.compile_and_save_results()
```
