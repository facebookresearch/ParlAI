**NOTE: this code is incompatible with the latest version of Mephisto, beginning with the changes made in [PR #246](https://github.com/facebookresearch/Mephisto/pull/246). This task will be upgraded shortly to regain compatibility with the latest version; in the meantime, please use the last version release of Mephisto prior to that PR, [v0.1](https://github.com/facebookresearch/Mephisto/releases/tag/v0.1).**

# Turn Annotations Static Task
This task renders conversations from a file and asks for turn by turn annotations of them. 
This is in contrast to the crowdsourcing task turn_annotations which collects turn by turn annotations from a live human-model conversation.

Look at turn_annotations_blueprint.py for various parameters including passing in custom annotation bucket definitions using the --annotation-buckets flag, being able to group multiple conversations into one HIT using the --subtasks-per-unit flag, passing in onboarding data with answers, and being able to ask only for the final utterance as an annotation.

Two variants of the Blueprint is supported: TurnAnnotationStaticBlueprint and TurnAnnotationStaticInFlightQABlueprint. The latter includes the ability to add an inflight additional Quality Assurance task.
