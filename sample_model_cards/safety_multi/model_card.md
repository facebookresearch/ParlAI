# BERT Classifier Multi-turn Dialogue Safety Model



Classifier trained on the multi-turn adversarial safety task in addition to both the single-turn standard and adversarial safety tasks and Wikipedia Toxic Comments.
- Developed by Facebook AI Research using [ParlAI](https://parl.ai/) 
-  Model started training on September 03, 2015. 
- Type of model: Bert Classifier 

### Quick Usage


```
parlai eval_model -t dialogue_safety:multiturn -dt test -mf zoo:dialogue_safety/multi_turn/model --split-lines True -bs 40
```

### Sample Input And Output

```
[text]: Explanation
Why the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27

[labels]: __ok__
---
[identity_hate]: 0.0
[threat]: 0.0
[is_sensitive]: 0.0
[toxic]: 0.0
[sensitive]: 0.0
[data_type]: train
[label_candidates]: __notok__, __ok__
[insult]: 0.0
[severe_toxic]: 0.0
[obscene]: 0.0
```

## Intended Use

> :warning: This model is intended for the use of....	:warning:

## Limitations

> :warning: This model has has these limitations: ...	:warning:

## Privacy

> :warning: This model has the following privacy concerns....	:warning:

## Datasets Used

This model was trained on the datasets below (use the `parlai display_data` commands to show data). Visit the [task (dataset) list](https://parl.ai/docs/tasks.html) for more details about the datasets.


- [dialogue safety (wikiToxicComments)](https://parl.ai/docs/tasks.html#dialogue-safety-(wikitoxiccomments))
- [dialogue safety: adversarial: round-only=False: round=1](https://parl.ai/docs/tasks.html#dialogue-safety:-adversarial:-round-only=false:-round=1)
- [dialogue safety (multiturn)](https://parl.ai/docs/tasks.html#dialogue-safety-(multiturn))

In addition, we have also included some basic stats about the training datasets in the table below:

|Dataset | avg utterance length | unique tokens | utterances | Display Dataset Command
:---: | :---: | :---: | :---: | :---:
dialogue safety: adversarial: round-only=False: round=1 | 11.335 | 6446 | 8000 | `parlai dd -t dialogue_safety:adversarial:round-only=False:round=1`
dialogue safety (multiturn) | 51.891 | 10853 | 24000 | `parlai dd -t dialogue_safety:multiturn`
dialogue safety (wikiToxicComments) | 87.685 | 206110 | 127656 | `parlai dd -t dialogue_safety:wikiToxicComments`

Note: The display dataset commands were auto generated, so please visit [here](https://parl.ai/docs/cli_usage.html#display-data) for more details.


## Evaluation Results

For evalution, we used the same training datasets; check the [Datasets Used](#datasets-used) section for more information


We used the metric `class notok f1`, the f1 scores for the class notok as the validation metric. Recall that `class___notok___f1` is unigram F1 overlap, under a standardized (model-independent) tokenizer.

|  | All | dialogue safety (wikiToxicComments) | dialogue safety: adversarial: round-only=False: round=1 | dialogue safety (multiturn)
:---: | :---: | :---: | :---: | :---:
`class notok f1` | 78.87% | 81.24% | 75.86% | 67.41%

<!-- ## Extra Analysis/Quantitative Analysis -->

## Related Paper(s)

[Build it Break it Fix it for Dialogue Safety: Robustness from Adversarial Human Attack](https://parl.ai/projects/dialogue_safety/)

## Hyperparameters

- `lr_scheduler`: ` fixed `
- `batchsize`: ` 40 `
- `learningrate`: ` 5e-05 `
- `model`: ` bert_classifier `
- `validation_patience`: ` 15 `
- `validation_metric`: ` class___notok___f1 `
- `multitask_weights`: ` [0.5, 0.1, 0.1, 0.3] `
- `max_train_steps`: ` Not specified `
- `num_epochs`: ` -1 `
<details> 
 <summary> model / neural net info </summary>
 <br>

- `round`: ` 3 `
- `threshold`: ` 0.5 `
</details>
<details> 
 <summary> embedding info </summary>
 <br>

- `embedding_type`: ` random `
- `embedding_projection`: ` random `
</details>
<details> 
 <summary> validation and logging info </summary>
 <br>

- `validation_metric_mode`: ` max `
- `validation_max_exs`: ` 10000 `
- `validation_cutoff`: ` 1.0 `
- `validation_every_n_secs`: ` 60.0 `
- `save_after_valid`: ` True `
- `validation_every_n_epochs`: ` -1 `
</details>
<details> 
 <summary> dictionary info/pre-processing </summary>
 <br>

- `dict_unktoken`: ` __unk__ `
- `dict_starttoken`: ` __start__ `
- `dict_tokenizer`: ` re `
- `dict_nulltoken`: ` __null__ `
- `dict_textfields`: ` text,labels `
- `dict_language`: ` english `
- `dict_class`: ` parlai.agents.bert_ranker.bert_dictionary:BertDictionaryAgent `
- `dict_endtoken`: ` __end__ `
- `dict_build_first`: ` True `
- `dict_max_ngram_size`: ` -1 `
- `dict_maxtokens`: ` -1 `
</details>
<details> 
 <summary> other dataset-related info </summary>
 <br>

- `fix_contractions`: ` True `
- `truncate`: ` 300 `
- `split_lines`: ` True `
- `task`: ` dialogue_safety:multiturn `
- `evaltask`: ` internal:safety:multiturnConvAI2 `
</details>
<details> 
 <summary> more batch and learning rate info </summary>
 <br>

- `lr_scheduler_patience`: ` 3 `
- `batch_sort_cache_type`: ` pop `
- `batch_sort_field`: ` text `
- `batchindex`: ` 39 `
- `batch_length_range`: ` 5 `
- `lr_scheduler_decay`: ` 0.9 `
</details>
<details> 
 <summary> training info </summary>
 <br>

- `numthreads`: ` 1 `
- `shuffle`: ` True `
- `numworkers`: ` 4 `
- `metrics`: ` default `
- `gpu`: ` -1 `
- `data_parallel`: ` True `
- `optimizer`: ` sgd `
- `gradient_clip`: ` 0.1 `
- `adam_eps`: ` 1e-08 `
- `nesterov`: ` True `
- `nus`: ` [0.7] `
- `betas`: ` [0.9, 0.999] `
- `warmup_updates`: ` 2000 `
- `warmup_rate`: ` 0.0001 `
- `update_freq`: ` 1 `
- `max_train_time`: ` -1 `
</details>
<details> 
 <summary> pytorch info </summary>
 <br>

- `pytorch_context_length`: ` -1 `
- `pytorch_include_labels`: ` True `
</details>
<details> 
 <summary> miscellaneous </summary>
 <br>

- `image_size`: ` 256 `
- `save_every_n_secs`: ` 60.0 `
- `get_all_metrics`: ` True `
- `sep_last_utt`: ` True `
- `image_cropsize`: ` 224 `
- `type_optimization`: ` all_encoder_layers `
- `use_reply`: ` label `
- `image_mode`: ` raw `
- `datatype`: ` train `
- `add_cls_token`: ` True `
</details>

## Feedback

We would love any feedback about the model (or the model card script)! Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/ParlAI/issues) :blush:


[back-to-top](#bert-classifier-multi-turn-dialogue-safety-model)
