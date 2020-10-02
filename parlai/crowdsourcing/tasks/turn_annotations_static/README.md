# Turn Annotations Static Task
This task renders conversations from a file and asks for turn-by-turn annotations of them. This is in contrast to the crowdsourcing task `parlai.mturk.tasks.turn_annotations`, which collects turn-by-turn annotations from a live human/model conversation.

See `turn_annotations_blueprint.py` for various parameters of this task, including passing in custom annotation bucket definitions using the `annotation_buckets` YAML flag, being able to group multiple conversations into one HIT using the `subtasks_per_unit` flag, passing in onboarding data with answers, and being able to ask only for the final utterance as an annotation.

Two variants of the blueprint are supported, `TurnAnnotationStaticBlueprint` and `TurnAnnotationStaticInFlightQABlueprint`. The latter variant includes the ability to add an additional in-flight (i.e. mid-HIT) quality assurance check.
