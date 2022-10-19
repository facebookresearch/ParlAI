# Reason first, then respond: Modular Generation for Knowledge-infused Dialogue
Leonard Adolphs, Kurt Shuster, Jack Urbanek, Arthur Szlam, Jason Weston

<b>Paper Link</b>: [https://arxiv.org/abs/2111.05204](https://arxiv.org/abs/2111.05204)

## Abstract
Large language models can produce fluent dialogue but often hallucinate factual inaccuracies. While retrieval-augmented models help alleviate this issue, they still face a difficult challenge of both reasoning to provide correct knowledge and generating conversation simultaneously. In this work, we propose a modular model, Knowledge to Response (K2R), for incorporating knowledge into conversational agents, which breaks down this problem into two easier steps. K2R first generates a knowledge sequence, given a dialogue context, as an intermediate step. After this "reasoning step", the model then attends to its own generated knowledge sequence, as well as the dialogue context, to produce a final response. In detailed experiments, we find that such a model hallucinates less in knowledge-grounded dialogue tasks,  and has advantages in terms
of interpretability and modularity.
In particular, it can be used to fuse QA and dialogue systems together to enable dialogue agents to give knowledgeable answers, or QA models to give conversational responses in a zero-shot setting.


## Train a shared K2R model on WoW
```
parlai train \
    -t projects.k2r.wow.task.agents:WizardOfWikipediaGeneratorTeacher:mutators=flatten+wow_checked_sentence_as_label,projects.k2r.wow.task.agents:WizardOfWikipediaGeneratorTeacher:mutators=flatten+wow_add_checked_sentence_to_input \
    --multitask_weights 1,1 --activation gelu --attention-dropout 0.0 --batchsize 16 --dropout 0.1 --fp16 True --gradient-clip 0.1 --label-truncate 128 \
    --text-truncate 512 --log-every-n-secs 30 --lr-scheduler reduceonplateau --lr-scheduler-patience 1 --max-train-time 169344.0 --model-parallel True \
    --model rag -o arch/bart_large --init-model zoo:bart/bart_large/model --dict-file zoo:bart/bart_large/model.dict --warmup-updates 0 \
    --multitask-weights stochastic --relu-dropout 0.0 --save-after-valid True --skip-generation True -lr 1e-05 -vmm min -veps 0.25 -vme 1000 \
    -vmt ppl -vp 5 --n-docs 5 -tblog True --indexer-type compressed --compressed-indexer-nprobe 128 \
    --model-file  ./models/wow/k2r_shared
```

## Evaluate the model on WoW
```
parlai em \
    -t projects.k2r.wow.task.agents:WizardOfWikipediaGeneratorTeacher:random_split \
    -m projects.k2r.stacked_agent.task.agents:StackedKnowledgeDialogueAgent \
    --knowledge-response-model-path ./models/wow/k2r_shared \
    --dialogue-response-model-path ./models/wow/k2r_shared \
    --dialogue-response-no-knowledge-model-path None \
    --dialogue-response-rag-wiki-model-path None \
    --mutators flatten -dt valid --krm-fp16 False --krm-model-parallel False --drm-model-parallel False --krm-beam-min-length 15 \
    --krm-beam-size 3 --krm-indexer-type compressed --krm-compressed-indexer-nprobe 128 --krm-n-docs 5 --drm-beam-size 3 --drm-beam-min-length 20 --batchsize 2 --log-every-n-secs 30 --metrics all
```

## Do interactive generations with the model
```
python projects/k2r/stacked_agent/scripts/stacked_agent_eval.py \
    --task wizard_of_wikipedia:Generator -dt test -bs 1 -n 100 \
    --interactive true --mutators flatten --random-order false --verbose true \
    --drm-beam-context-block-ngram 3 --beam-disregard-knowledge-for-context-blocking false \
    --knowledge-response-model-path ./models/wow/k2r_shared \
    --dialogue-response-model-path ./models/wow/k2r_shared
```

## LightQA data
Our goal with LightQA is to have a task that requires a model to answer questions *about the previous context*. For example, in LIGHT, a player might ask another character where to find a certain key to complete their quest. Here, we would want a model, acting as the character, to answer appropriately if the knowledge is in the context description. With this goal in mind, we design a dataset in the following way: First, we take a LightWild episode and use an abstractive summarization model, trained on CNN/Daily Mail and the SAMSum Corpus, to generate a summary. Then we identify all noun chunks, entities, and proper nouns and use them as possible answer candidates. For each answer candidate, we use a T5 question generation model, trained on SQuAD, to generate a possible question given the summary as context. As the last step, we filter the generated questions with a QA model, trained on SQuAD, by checking that it would generate the used answer candidate with access to the summary and question. An episode of our dataset consists of the original LightWild episode (up to a certain turn) and the generated question as the last utterance. Hence, our labels in this dataset are not the usual dialogue responses but short answers.
```
# Display the data.
parlai dd -t projects.k2r.lightqa.task.agents -dt valid
```

