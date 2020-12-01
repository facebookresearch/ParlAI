# Turn annotations crowdsourcing task

This task will collect conversations between a human and a model. After each response by the model, the human will be prompted to annotate the model's response by selecting checkboxes that represent customizable attributes of that response.

## Launching

Call `run.py` to run this task with the default parameters, as set by `conf/example.yaml`.

Some parameters that you can adjust include where to save data, lists of workers to soft-block, the maximum response time, etc.

## Passing in task config files

The following flags can be passed in to specify filepaths for overriding the text shown to the workers and the settings of the annotation categories. If they are not specified, the defaults in the `task_config/` folder will be used.
- `mephisto.blueprint.annotations_config_path`: JSON file configuring annotation categories
- `mephisto.blueprint.left_pane_text_path`: HTML to show on the left-hand pane of the chat window
- `mephisto.blueprint.onboard_task_data_path`: JSON specifying parameters for testing workers during onboarding
- `mephisto.blueprint.task_description_file`: HTML to show on the initial task-description page shown to the worker

## Onboarding

In `worlds.py`, modify `TurnAnnotationsOnboardWorld.check_onboarding_answers()` to change the worker selection criteria.
