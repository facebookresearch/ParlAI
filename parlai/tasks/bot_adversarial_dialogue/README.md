# Task: Bot Adversarial Dialogue Dataset

## Description
Dialogue datasets labeled with offensiveness from Bot Adversarial Dialogue task

[Project](parl.ai/projects/recipes/safety_recipes)
[Arxiv Paper](https://arxiv.org/abs/2010.07079)

## Teachers
The `BotAdversarialDialogueTeacher` in `agents.py` allows for iteration over adversarial dialogue datasets in which each example has been annotated for offensiveness. The `label` field represents the offensiveness of the final utterance in  `text` field given the dialogue context included in the `text` field as well.
The `HumanSafetyEvaluationTeacher` in `agents.py` display adversarial dialogue truncation for human safety evaluation task where the final utterance in `text` field within each episode is evaluated by crowdsourced workers for offensiveness. The exact turn indices of each dialogue truncation shown to the crowdsourcing workers is indicated by the field `human_eval_turn_range`.

## Files
This code downloads the following folders/files into the ParlAI data folder:
- `dialogue_datasets/`: folder containing files of adversarial datasets labeled for offensiveness.
- `human_eval/`: folder containing files of adversarial dialogue truncations used for human evaluation.

Tags: #Safety, #All, #ChitChat
