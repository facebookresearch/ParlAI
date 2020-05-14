Task: Blended Skill Talk
===========================
Description: A dataset of 7k conversations explicitly designed to exhibit multiple conversation modes: displaying personality, having empathy, and demonstrating knowledge.
=========================== 
Dataset has been released under the CC BY-NC license.

## BlendedSkillTalkTeacher
Conversation is between a "free" Amazon Mechanical Turk worker who can speak freely and a "guided" worker who is given 3 suggestions of what to say for each turn. This dataset presents the half of the conversation in which the label is the guided worker's utterance. Examples include the following special fields:
- `[context_dataset]`: If this string is `'empathetic_dialogues'`, a string representing a situation from the EmpatheticDialogues dataset was added to the context. If `'wizard_of_wikipedia'`, a WoW topic string was added to the context. If `'convai2'`, no additional context was added
- `[free_message]`: The free worker's utterance on the previous turn
- `[convai2]`, `[empathetic_dialogues]`, and `[wizard_of_wikipedia]`: The suggestions given to the guided worker by the three models that were trained on one single task each
- `[guided_chosen_suggestion]`: If not blank, specifies the dataset (convai2, empathetic_dialogues, or wizard_of_wikipedia) that the suggestion chosen by the guided worker came from

## ConvAI2PersonaTopicifierTeacher, EDPersonaTopicifierTeacher, WoWPersonaTopicifierTeacher
Versions of teachers for the ConvAI2, EmpatheticDialogues, and Wizard of Wikipedia datasets in which ConvAI2-style persona strings and Wizard-of-Wikipedia-style topic strings have been added to the start of all contexts.

Tags: #BlendedSkillTalk, #All, #ChitChat
