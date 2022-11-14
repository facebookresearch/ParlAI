# The CRINGE Loss: Learning what language *not* to model

Leonard Adolphs, Tianyu Gao, Jing Xu, Kurt Shuster, Sainbayar Sukhbaatar, Jason Weston


## Abstract
Standard language model training employs gold human documents or human-human interaction data, and 
treats all training data as positive examples. 
Growing evidence shows that even with very large amounts of positive training data, issues remain
that can be alleviated with relatively small amounts of negative data -- examples of what the model should not do.
In this work, we propose a novel procedure to train with such data called the CRINGE loss
(ContRastive Iterative Negative GEneration).
  We show the effectiveness of this approach across three different experiments on the tasks of safe generation,
  contradiction avoidance, and open-domain dialogue. Our models outperform multiple strong baselines and are
  conceptually simple, easy to train and implement.

## Paper Link


* [The CRINGE Loss: Learning what language not to model](https://arxiv.org/abs/2211.05826).
Leonard Adolphs, Tianyu Gao, Jing Xu, Kurt Shuster, Sainbayar Sukhbaatar, Jason Weston

## Train a CRINGE (single iter.) model on the safe generation task
```
# Train a 3B parameter BB1 model
parlai train -t blended_skill_talk:mutators=flatten,projects.director.tasks.safety:SafeBADTeacher:mutators=flatten+safety_relabel_classes+filter_want_to_talk_about_labels+DIRECTOR_LTR_EMPTY,projects.director.tasks.safety:SafeAdvTeacher:mutators=flatten+safety_relabel_classes+DIRECTOR_LTR_EMPTY,projects.director.tasks.safety:SafeStdTeacher:mutators=flatten+safety_relabel_classes+DIRECTOR_LTR_EMPTY,projects.director.tasks.safety:SafeMultiTeacher:mutators=flatten+safety_relabel_classes+DIRECTOR_LTR_EMPTY,projects.director.tasks.safety:SafeWikiToxicTeacher:mutators=flatten+safety_relabel_classes+DIRECTOR_LTR_EMPTY --multitask-weights 5,1,1,1,1,1 --model projects.cringe.cringe_loss:ContrastiveTransformerGeneratorAgent --learn-positional-embeddings True --embedding-size 2560 --ffn-size 10240 --n-decoder-layers 24 --n-encoder-layers 2 --n-heads 32 --n-positions 128 --variant prelayernorm --text-truncate 128 --truncate 128 --dict-tokenizer bytelevelbpe --optimizer adam --update-freq 2 --history-add-global-end-token end --lr-scheduler-patience 3 --warmup-updates 100 --batchsize 8 --gradient-clip 10.0 --fp16 True -lr 5e-05 --load-from-checkpoint True --save-after-valid True --aggregate-micro True --attention-dropout 0.1 --dropout 0.1 --label-truncate 512 --relu-dropout 0.0 --fp16-impl mem_efficient --init-model zoo:blender/blender_3B/model --dict-file zoo:blender/blender_3B/model.dict --model-file .models/cringe/safe_bb1/model --model-parallel true

```


## Evaluate the CRINGE (single iter.) model on the safe generation task

### Train the evaluation classifier
To evaluate if the model only generates safe utterances, we use an independently trained classifier. Here, we use the training
script from the [DIRECTOR](https://parl.ai/projects/director/): 
```
 parlai train --task projects.director.tasks.safety:SafeBADTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeAdvTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeStdTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeMultiTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeWikiToxicTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeBADTeacher:mutators=flatten+safety_relabel_classes+neg_only,projects.director.tasks.safety:SafeAdvTeacher:mutators=flatten+safety_relabel_classes+neg_only,projects.director.tasks.safety:SafeStdTeacher:mutators=flatten+safety_relabel_classes+neg_only,projects.director.tasks.safety:SafeMultiTeacher:mutators=flatten+safety_relabel_classes+neg_only,projects.director.tasks.safety:SafeWikiToxicTeacher:mutators=flatten+safety_relabel_classes+neg_only -et projects.director.tasks.safety:SafeBADTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeAdvTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeStdTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeMultiTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeWikiToxicTeacher:mutators=flatten+safety_relabel_classes+pos_only,projects.director.tasks.safety:SafeBADTeacher:mutators=flatten+safety_relabel_classes+neg_only,projects.director.tasks.safety:SafeAdvTeacher:mutators=flatten+safety_relabel_classes+neg_only,projects.director.tasks.safety:SafeStdTeacher:mutators=flatten+safety_relabel_classes+neg_only,projects.director.tasks.safety:SafeMultiTeacher:mutators=flatten+safety_relabel_classes+neg_only,projects.director.tasks.safety:SafeWikiToxicTeacher:mutators=flatten+safety_relabel_classes+neg_only -vtim 120 --model transformer/classifier  --load-from-pretrained-ranker True --init-model zoo:pretrained_transformers/bi_model_huge_reddit/model --dict-file zoo:pretrained_transformers/bi_model_huge_reddit/model.dict --history-size 20 --label-truncate 72 --text-truncate 360 --dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 --variant xlm --reduction-type mean --share-encoders False --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 --embedding-size 768 --activation gelu  --embeddings-scale False --n-segments 2 --learn-embeddings True --share-word-embeddings False --dict-endtoken __start__  -vp 30 -stim 60 --lr-scheduler fixed --lr-scheduler-patience 3 --lr-scheduler-decay 0.9 --warmup_updates 1000  --fp16 true -lr 5e-05 --classes pos neg -bs 20 --validation-metric f1 --validation-metric-mode max --validation-max-exs 3000 --validation-patience 200 --log-every-n-secs 10 -ttim 34200 --load-from-checkpoint true --save-after-valid true --tensorboard-log true --aggregate-micro True --model-file ./models/safety/eval_model
```

### Evaluate the model checkpoint
```
parlai em --batchsize 8 --log-every-n-secs 30 --fp16 True --metrics all --inference beam --beam-size 10 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3 --beam-block-full-context True --skip-generation False --task projects.director.tasks.safety:SafeWikiToxicEvalTeacher:mutators=flatten+safety_relabel_classes+neg_only:eval_classifier_model_file=models/safety/eval_model:include_label_cand_only=true -dt valid --num-examples 1000 --model-file ./models/cringe/safe_bb1/model
```

## Iterative Training

### Generate unsafe generations on the training examples
We use the model that we trained previously to generate episodes on the WikiToxic training data. We log all the results as WikiToxic_world_logs.jsonl.
```
parlai em --batchsize 16 --log-every-n-secs 30 --fp16 True --metrics all --inference beam --beam-size 10 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3 --beam-block-full-context True --skip-generation False --task projects.director.tasks.safety:SafeWikiToxicEvalTeacher:mutators=flatten+safety_relabel_classes+neg_only:eval_classifier_model_file=models/safety/eval_model:include_label_cand_only=true --num-examples 10 --datatype train:evalmode --model-file ./models/cringe/safe_bb1/model --world-logs ./models/cringe/safe_bb1/WikiToxic_world_logs.jsonl
```

### Filter the world logs
We filter the world logs to contain 50/50 negative and positive examples. The previously trained classifier determines the label.
```
python projects/cringe/safety_filter_world_logs.py --world-logs-file ./models/cringe/safe_bb1/WikiToxic_world_logs.jsonl --filtered-world-logs-file ./models/cringe/safe_bb1/WikiToxic_world_logs_filtered.jsonl
```

### Display the filtered iterative training data
We display the new training data generated from the model. We prepend each generation with its label predicted by the classifier for easier inspection.
```
parlai dd -t projects.cringe.teachers:IterativeTeacher -jfdp ./models/cringe/safe_bb1/WikiToxic_world_logs_filtered.jsonl --prepend-classifier-label true
```

### Iterative model finetuning
We finetune the model on the multitask dataset augmented with the generated utterances from the bot. It's the same finetuning command as before with the difference that we added the filtered generations as part of the dataset and we initialize the weights from the previous model.
```
parlai train -t blended_skill_talk:mutators=flatten,projects.director.tasks.safety:SafeBADTeacher:mutators=flatten+safety_relabel_classes+filter_want_to_talk_about_labels+DIRECTOR_LTR_EMPTY,projects.director.tasks.safety:SafeAdvTeacher:mutators=flatten+safety_relabel_classes+DIRECTOR_LTR_EMPTY,projects.director.tasks.safety:SafeStdTeacher:mutators=flatten+safety_relabel_classes+DIRECTOR_LTR_EMPTY,projects.director.tasks.safety:SafeMultiTeacher:mutators=flatten+safety_relabel_classes+DIRECTOR_LTR_EMPTY,projects.director.tasks.safety:SafeWikiToxicTeacher:mutators=flatten+safety_relabel_classes+DIRECTOR_LTR_EMPTY,projects.cringe.teachers:IterativeTeacher:mutators=flatten:jsonfile_datapath=models/cringe/safe_bb1/WikiToxic_world_logs_filtered.jsonl --multitask-weights 5,1,1,1,1,1,1 --model projects.cringe.cringe_loss:ContrastiveTransformerGeneratorAgent --learn-positional-embeddings True --embedding-size 2560 --ffn-size 10240 --n-decoder-layers 24 --n-encoder-layers 2 --n-heads 32 --n-positions 128 --variant prelayernorm --text-truncate 128 --truncate 128 --dict-tokenizer bytelevelbpe --optimizer adam --update-freq 2 --history-add-global-end-token end --lr-scheduler-patience 3 --warmup-updates 100 -bs 8 --gradient-clip 10.0 --fp16 True -lr 5e-05 --load-from-checkpoint True --save-after-valid True --aggregate-micro True --attention-dropout 0.1 --dropout 0.1 --label-truncate 512 --relu-dropout 0.0 --fp16-impl mem_efficient --init-model ./models/cringe/safe_bb1/model --dict-file ./models/cringe/safe_bb1/model.dict --model-file .models/cringe/safe_bb1_iterative/model --model-parallel true
```
