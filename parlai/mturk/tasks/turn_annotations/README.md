# Turn Annotations MTurk Task

This task will collect conversations between a human and a model. After each response by the model, the human will be prompted to annotate the model's response by selecting  checkboxes that represent customizable attributes of that response.

**WARNING**: This code uses the legacy version of the ParlAI MTurk integration, which will eventually be deprecated in favor of the [Mephisto](https://github.com/facebookresearch/Mephisto) platform.

## Launching

This script can be run either on the command line or via a Python launch script:
- **Command line**: see the following minimal sample command for launching 5 sandbox HITs:
```
python parlai/mturk/tasks/turn_annotations/run.py \
--block-qualification ${BLOCK_QUALIFICATION_NAME} \
--base-model-folder ${BASE_MODEL_FOLDER} \
--num-conversations 5 \
--sandbox \  # Replace with "--live" for live mode
--reward 3 \  # Only necessary in live mode
--conversations-needed ${MODEL_NAME}:5
# A model file should be present at ${BASE_MODEL_FOLDER}/${MODEL_NAME}/model
```
- **Python launch script**: see `launch.py` for an example script.

Some additional parameters that you can set include where to save data, lists of workers to soft-block, the maximum response time, etc.

## Passing in task config files

The following flags can be passed in to specify filepaths for overriding the text shown to the workers and the settings of the annotation categories. If they are not specified, the defaults in the `task_config/` folder will be used.
- `--annotations-config-path`: JSON file configuring annotation categories
- `--hit-config-path`: JSON of language that MTurk uses to describe the HIT to workers
- `--left-pane-text-path`: HTML to show on the left-hand pane of the chat window
- `--onboard-task-data-path`: JSON specifying parameters for testing workers during onboarding
- `--task-description-path`: HTML to show on the initial task-description page shown to the worker

Note that, when launching with a Python launch script, all of the above information can be passed into the task directly, instead of passing in a path to a file containing that information. For instance, in the launch script, the contents of the HTML file specified by `--task-description-path` can instead be passed in as a string by setting `opt['task_description']`.

## Onboarding

In `worlds.py`, modify `TurnAnnotationsOnboardWorld.check_onboarding_answers()` to change the worker selection criteria.
