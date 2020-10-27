# Turn Annotations Static Task
This task renders conversations from a file and asks for turn-by-turn annotations of them. This is in contrast to the crowdsourcing task `parlai.mturk.tasks.turn_annotations`, which collects turn-by-turn annotations from a live human/model conversation.

Two variants of the blueprint are supported:
- `TurnAnnotationStaticBlueprint`
    - The base static turn-annotations task
    - Called with `python parlai/crowdsourcing/tasks/turn_annotations_static/run.py`
- `TurnAnnotationStaticInFlightQABlueprint`
    - Includes the ability to add an additional in-flight (i.e. mid-HIT) quality assurance check
    - Called with `python parlai/crowdsourcing/tasks/turn_annotations_static/run_in_flight_qa.py`
    
For both variants of the blueprint, it is required to pass in your own file of conversations with `mephisto.blueprint.data_jsonl=${PATH_TO_CONVERSATIONS}`.

See `turn_annotations_blueprint.py` for various parameters of this task, including passing in custom annotation bucket definitions using the `annotation_buckets` YAML flag, being able to group multiple conversations into one HIT using the `subtasks_per_unit` flag, passing in onboarding data with answers, and being able to ask only for the final utterance as an annotation.
