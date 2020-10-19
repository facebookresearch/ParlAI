## Queens are Powerful too: Mitigating Gender Bias in Dialogue Generation

Emily Dinan, Angela Fan, Adina Williams, Jack Urbanek, Douwe Kiela, Jason Weston

## Abstract

Models often easily learn biases present in the training data, and their predictions directly reflect this bias.
We analyze the presence of gender bias in dialogue and examine the subsequent effect on generative chitchat
dialogue models. Based on this analysis, we propose a combination of three techniques to mitigate bias:
counterfactual data augmentation, targeted data collection, and conditional training. We focus on the multi-player
text-based fantasy adventure dataset LIGHT (Urbanek et al., 2019) as a testbed for our work.
LIGHT contains gender imbalance between male and female characters with around _1.6x_ as many male characters,
likely because it is entirely collected by crowdworkers and reflects common biases that exist in fantasy or
medieval settings.
We show that (i) our proposed techniques mitigate gender bias by balancing the genderedness of generated
dialogue utterances;  and (ii) they work particularly well in combination. Further,
we show through various metrics---such as quantity of gendered words, a dialogue safety classifier,
and human evaluation---that our models generate less gendered, but still engaging chitchat responses.

## Paper

[Link](https://arxiv.org/abs/1911.03842) to appear at EMNLP 2020.

## Data

The data for training models for this project can be found at:
```bash
parlai display_data -t light_genderation_bias
```

By default, all mitigation methods are turned on at once. Use the flags `--add-conditional` (Bias Ctrl training), `--add-new-data` (Positive Bias data), and `--add-counterfactual` (Counterfactual Data Augmentation) to control which mitigation methods to use.

## Models

TBD.