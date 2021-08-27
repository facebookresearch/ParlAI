# Hi, my name is Martha: Using names to measure and mitigate bias in generative dialogue models

Eric Michael Smith, Adina Williams

## Abstract

{{{TODO: add final abstract once posted to arXiv}}}

## Paper

[Link]() [[[TODO: add link once posted to arXiv]]]

## Code

- `projects.dialogue_bias.agents:NoBiasStyleGenAgent`: Agent that appends a `"no_bias"` string to the context of every example in order to perform controllable generation.

## Models

We release several models in which [BlenderBot3B](https://parl.ai/projects/recipes/) have been tuned to reduce bias along the axes of gender and/or race/ethnicity: see [here](https://github.com/facebookresearch/ParlAI/tree/master/parlai/zoo/dialogue_bias) for descriptions of these models.

We cannot guarantee that these reduced-bias models are free of all remaining bias: improving safety and reducing bias in models is an ongoing research effort whose metrics and aims have not yet been precisely pinned down by the wider AI community. See [this post](https://emdinan1.medium.com/a-recap-of-the-first-workshop-on-safety-for-conversational-ai-98201d257530) for one such discussion in November 2020 of how best to measure and solve the problem of unsafe dialogue agents.

To interact with the model in which gender has been reduced via name scrambling:

```
python parlai/scripts/safe_interactive.py \
-mf zoo:dialogue_bias/gender__name_scrambling/model \
-t blended_skill_talk
```

## Citation

If you use the data or models in your own work, please cite with the following BibTex entry:

{{{TODO: add this once posted to arXiv}}}
