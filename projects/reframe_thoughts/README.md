# Training Models to Generate, Recognize, and Reframe Unhelpful Thoughts

Mounica Maddela, Megan Ung, Jing Xu, Andrea Madotto, Heather Foran, Y-Lan Boureau

## Abstract

Many cognitive approaches to well-being, such as recognizing and reframing unhelpful thoughts, have received considerable empirical support over the past decades, yet still lack truly widespread adoption in self-help format. A barrier to that adoption is a lack of adequately specific and diverse dedicated practice material. This work examines whether current language models can be leveraged to  both produce a virtually unlimited quantity of practice material illustrating standard unhelpful thought patterns matching specific given contexts, and generate suitable positive reframing proposals. We propose \textsc{PatternReframe}, a novel dataset of about 10k examples of thoughts containing unhelpful thought patterns conditioned on a given persona, accompanied by about 27k positive reframes. By using this dataset to train and/or evaluate current models, we show that existing models can already be powerful tools to help generate an abundance of tailored practice material and hypotheses, with no or minimal additional model training required.

## Paper



## Data

```
parlai display_data -t reframe_thoughts
```
