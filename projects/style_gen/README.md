# Style-controlled generation

Agent for training and evaluating generative models conditioned on a style token, for instance, the "personality" string attached to each example of the Image-Chat dataset.

- `--model projects.style_gen.style_gen:StyleGenAgent`: Subclass of `TransformerGeneratorAgent` for which style tokens can be appended to the history for every training/evaluation example. Found in `style_gen.py`.

## Basic examples

Evaluating a style-controlled generation model on the BlendedSkillTalk dataset, labeled with personalities from Image-Chat:
```
parlai eval_model \
--model-file zoo:style_gen/c75_labeled_dialogue_generator/model \
--model projects.style_gen.style_gen:StyleGenAgent \
--skip-generation True \
--task style_gen:LabeledBlendedSkillTalk \
--datatype test \
--use-style-frac 1.00
```
