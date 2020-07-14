Task: Style-Controlled Generation
===========================
Description: Dialogue datasets (BlendedSkillTalk, ConvAI2, EmpatheticDialogues, and Wizard of Wikipedia) labeled with personalities taken from the Image-Chat dataset.
=========================== 
Data has been released under the CC BY-NC license.

## Teachers
The four teachers in `agents.py` allow for iteration over different dialogue datasets in which each example has been annotated with a personality from the Image-Chat dataset. The personality, found in the `personality` field, was labeled using a classifier trained to classify the text string in the `label` field given that text and the final utterance in the `text` field. The teachers are as follows:
- `LabeledBlendedSkillTalkTeacher`: labeled version of `blended_skill_talk:BlendedSkillTalk`
- `LabeledConvAI2PersonaTopicifierTeacher`: labeled version of `blended_skill_talk:ConvAI2PersonaTopicifier`
- `LabeledEDPersonaTopicifierTeacher`: labeled version of `blended_skill_talk:EDPersonaTopicifier`
- `LabeledWoWPersonaTopicifierTeacher`: labeled version of `blended_skill_talk:WoWPersonaTopicifier`

## Files
This code downloads the following folders/files into the ParlAI data folder:
- `labeled_datasets/`: folder containing files of datasets labeled with Image-Chat personalities (see the "Teachers" section above). Files are grouped into folders according to dataset, and each file comprises the examples for one datatype of one dataset.
- `personality_list.txt`: list of all personalities in the Image-Chat train set. Used by the classifier that labeled the datasets with Image-Chat personalities.

Tags: #StyleGen, #All, #ChitChat
