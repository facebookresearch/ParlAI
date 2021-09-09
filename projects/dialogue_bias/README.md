# Hi, my name is Martha: Using names to measure and mitigate bias in generative dialogue models

Eric Michael Smith, Adina Williams

## Abstract

All AI models are susceptible to learning biases in data that they are trained on. For generative dialogue models, being trained on real human conversations containing unbalanced gender and race/ethnicity references can lead to models that display learned biases, which we define here broadly as any measurable differences in the distributions of words or semantic content of conversations based on demographic groups. We measure the strength of such biases by producing artificial conversations between two copies of a dialogue model, conditioning one conversational partner to state a name commonly associated with a certain gender and/or race/ethnicity. We find that larger capacity models tend to exhibit more gender bias and greater stereotyping of occupations by gender. We show that several methods of tuning these dialogue models, specifically name scrambling, controlled generation, and unlikelihood training, are effective in reducing bias in conversation, including on a downstream conversational task. Name scrambling is also effective in lowering differences in token usage across conversations where partners have names associated with different genders or races/ethnicities.

## Paper

[Link](https://arxiv.org/abs/2109.03300)

## Code

- `projects.dialogue_bias.agents:NoBiasStyleGenAgent`: Agent that appends a `"no_bias"` string to the context of every example in order to perform controllable generation.

## Models

We release several models in which [BlenderBot3B](https://parl.ai/projects/recipes/) has been tuned to reduce bias along the axes of gender and/or race/ethnicity: see [here](https://github.com/facebookresearch/ParlAI/tree/master/parlai/zoo/dialogue_bias) for descriptions of these models.

We cannot guarantee that these reduced-bias models are free of all remaining bias: improving safety and reducing bias in models is an ongoing research effort whose metrics and aims have not yet been precisely pinned down by the wider AI community. See [this post](https://emdinan1.medium.com/a-recap-of-the-first-workshop-on-safety-for-conversational-ai-98201d257530) for one such discussion in November 2020 of how best to measure and solve the problem of unsafe dialogue agents.

To interact with the model in which gender has been reduced via name scrambling:

```
parlai safe_interactive \
-mf zoo:dialogue_bias/gender__name_scrambling/model \
-t blended_skill_talk \
--beam-block-full-context True
```

## Citation

If you use the data or models in your own work, please cite with the following BibTex entry:

```
@misc{smith2021hi,
      title={Hi, my name is Martha: Using names to measure and mitigate bias in generative dialogue models}, 
      author={Eric Michael Smith and Adina Williams},
      year={2021},
      eprint={2109.03300},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
