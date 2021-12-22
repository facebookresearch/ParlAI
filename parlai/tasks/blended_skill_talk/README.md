Task: Blended Skill Talk
===========================
Description: A dataset of 7k conversations explicitly designed to exhibit multiple conversation modes: displaying personality, having empathy, and demonstrating knowledge.
=========================== 

LICENSE: This dataset has been released under the CC-BY-4.0 License. Please
refer to the LICENSE_DOCUMENTATION file in this repository for more
information.

## BlendedSkillTalkTeacher
Conversation is between a "free" Amazon Mechanical Turk worker who can speak freely and a "guided" worker who is given 3 suggestions of what to say for each turn. This dataset presents the half of the conversation in which the label is the guided worker's utterance. Examples include the following special fields:
- `[context_dataset]`: If this string is `'empathetic_dialogues'`, a string representing a situation from the EmpatheticDialogues dataset was added to the context. If `'wizard_of_wikipedia'`, a WoW topic string was added to the context. If `'convai2'`, no additional context was added
- `[free_message]`: The free worker's utterance on the previous turn
- `[convai2]`, `[empathetic_dialogues]`, and `[wizard_of_wikipedia]`: The suggestions given to the guided worker by the three models that were trained on one single task each
- `[guided_chosen_suggestion]`: If not blank, specifies the dataset (convai2, empathetic_dialogues, or wizard_of_wikipedia) that the suggestion chosen by the guided worker came from

## ConvAI2PersonaTopicifierTeacher, EDPersonaTopicifierTeacher, WoWPersonaTopicifierTeacher
Versions of teachers for the ConvAI2, EmpatheticDialogues, and Wizard of Wikipedia datasets in which ConvAI2-style persona strings and Wizard-of-Wikipedia-style topic strings have been added to the start of all contexts.

## Files
The following files are downloaded when calling the BST dataset (for instance, with `parlai display_data -t blended_skill_talk`):
- `train.json`, `valid.json`, and `test.json`: Raw BST datafiles, each consisting of a list of dicts. Each dict is a conversation that includes the following fields, in addition to those listed under **BlendedSkillTalkTeacher** above:
  - `[personas]`: The strings giving the personas of the guided and free workers
  - `[free_turker_utterance]` and `[guided_turker_utterance]`: Two utterances that start off the conversation, displayed before the AMT workers continue the conversation from there. One utterance is given for each worker
  - `[additional_context]`: An additional string added to the context for certain values of `[context_dataset]`. If `[context_dataset]` is `'empathetic_dialogues'`, this is a string representing a situation from the EmpatheticDialogues dataset. If `[context_dataset]` is `'wizard_of_wikipedia'`, this is a topic string from the Wizard of Wikipedia dataset
  - `[dialog]`: Lines of dialogue given by the free worker (represented by a `0`) and the guided worker (`1`)
- `persona_list.txt`: A list of possible personas, used when adding personas to context strings by the `WoWPersonaTopicifierTeacher` and `EDPersonaTopicifierTeacher`
- `topic_to_persona_list.txt`: A list of WoW topics and the persona strings that they correspond to, used by the PersonaTopicifierTeachers to ensure that WoW topics are relevant given persona strings and vice versa
- `ed_persona_topicifier__train__both_sides.json`, `ed_persona_topicifier__train__experiencer_only.json`, `ed_persona_topicifier__valid__experiencer_only.json`, and `ed_persona_topicifier__test__experiencer_only.json`: cached files of EmpatheticDialogues conversations with WoW topics and persona strings added to contexts, used by `EDPersonaTopicifierTeacher`. EmpatheticDialogues conversations consist of a Speaker describing a situation and a Listener responding empathetically: the choice of file used depends on whether we are considering conversation turns in which the previous utterance comes from either the Speaker or the Listener (`both_sides`) or from the Speaker (i.e. experiencer) only (`experiencer_only`)
- `safe_personas.txt`: A list of safe personas, used with BlendedSkillTalk in interactive mode with `--safe-personas-only True`
- `human_annotations.json`: The set of ~700 per-turn human annotations of conversations in the BlendedSkillTalk validation set. Conversation turns were annotated to indicate whether they exhibited the modes of knowledge, empathy, personal situations, and/or personal background.

Tags: #BlendedSkillTalk, #All, #ChitChat
