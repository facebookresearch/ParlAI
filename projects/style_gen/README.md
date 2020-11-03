# Style-controlled generation

Agents for training and evaluating generative models conditioned on a style token, for instance, the "personality" string attached to each example of the Image-Chat dataset.

- `--model projects.style_gen.style_gen:StyleGenAgent`: Subclass of `TransformerGeneratorAgent` for which style tokens can be appended to the history for every training/evaluation example. Found in `style_gen.py`.
- `--model projects.style_gen.classifier:ClassifierAgent`: Subclass of `TransformerGeneratorAgent` that adds a classifier head on top of the base generator. Can be used for fine-tuning a classifier using the weights of a pretrained generator model. Classifier labels can be specified either in the `'labels'` field of the observation or in a separate `'personality'` field, and all encoder and decoder weights can optionally be frozen when training the classifier head. Found in `classifier.py`.

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

Evaluating the Image-Chat-trained classifier on the LabeledBlendedSkillTalk dataset, which it itself labeled with a style for each utterance:
```
parlai eval_model \
--model-file zoo:style_gen/prev_curr_classifier/model \
--model projects.style_gen.classifier:ClassifierAgent \
--classes-from-file image_chat_personalities_file \
--task style_gen:PrevCurrUttStyle \
--wrapper-task style_gen:LabeledBlendedSkillTalk
```
