Task: Wizard_of_Wikipedia
==========================
Description: A dataset with conversations directly grounded with knowledge
retrieved from Wikipedia. Contains 201k utterances from 22k dialogues spanning
over 1300 diverse topics, split into train, test, and valid sets. The test and
valid sets are split into two sets each: one with overlapping topics with the
train set, and one with unseen topics.See https://arxiv.org/abs/1811.01241 for
more information.

LICENSE: This dataset has been released under the CC-BY-4.0 License. Please
refer to the LICENSE_DOCUMENTATION file in this repository for more
information.

Tags: #Wizard_of_Wikipedia, #All, #ChitChat

Notes: To access the different valid/test splits (unseen/seen), specify the corresponding split (`random_split` for seen, `topic_split` for unseen) after the last colon in the task. E.g. `wizard_of_wikipedia:WizardDialogKnowledgeTeacher:random_split`

