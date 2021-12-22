# Turn Annotations Static Task
This task renders conversations from a file and asks for turn-by-turn annotations of them. This is in contrast to the crowdsourcing task `parlai.crowdsourcing.tasks.model_chat`, which collects turn-by-turn annotations from a live human/model conversation.

Two variants of the blueprint are supported:
- `TurnAnnotationStaticBlueprint`
    - The base static turn-annotations task
    - Called with `python parlai/crowdsourcing/tasks/turn_annotations_static/run.py`
        - (the task runs with the default parameters set in `hydra_configs/conf/example.yaml`)
- `TurnAnnotationStaticInFlightQABlueprint`
    - Includes the ability to add an additional in-flight (i.e. mid-HIT) quality assurance check
    - Called with `python parlai/crowdsourcing/tasks/turn_annotations_static/run.py conf=example_in_flight_qa`
        - (the task runs with the parameters set in `hydra_configs/conf/example_in_flight.yaml`)

For both variants of the blueprint, it is required to pass in your own file of conversations with `mephisto.blueprint.data_jsonl=${PATH_TO_CONVERSATIONS}`.

See `turn_annotations_blueprint.py` for various parameters of this task, including passing in custom annotation bucket definitions using the `annotations_config_path` YAML flag, being able to group multiple conversations into one HIT using the `subtasks_per_unit` flag, passing in onboarding data with answers, and being able to ask only for the final utterance as an annotation.

The validation of the response field is handled by `validateFreetextResponse` function in `task_components.jsx` and checks for a minimum number of characters, words, and vowels specified by function parameters. To change this, modify the values passed in to the function call or override the function to set your own validation requirements.

**NOTE**: See [parlai/crowdsourcing/README.md](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/README.md) for general tips on running `parlai.crowdsourcing` tasks, such as how to specify your own YAML file of configuration settings, how to run tasks live, how to set parameters on the command line, etc.

## Analysis

Run `analysis/compile_results.py` to compile and save statistics about collected static turn annotations. The `TurnAnnotationsStaticResultsCompiler` in that script uses dummy annotation buckets by default; set `--problem-buckets` in order to define your own. Set `--results-folders` to a comma-separated list of the folders that the data was saved to (likely of the format `"/basefolder/mephisto/data/runs/NO_PROJECT/123"`)
