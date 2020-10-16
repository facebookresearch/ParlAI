# Model card: Multi-Modal BlenderBot

Model card for the MMB DegenPos model described in [Multi-Modal Open-Domain Dialogue](https://arxiv.org/abs/2010.01082).

## Model details
This model was trained to create a dialogue agent that can converse engagingly about the content of an image as well as about general chitchat. It is a 2.7-billion-parameter Transformer sequence-to-sequence model based on the [BlenderBot](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/) open-domain chatbot, trained on the image tasks [COCO Captions](https://cocodataset.org/) and [Image-Chat](https://parl.ai/projects/image_chat/), and using an image encoder from [Faster R-CNN](https://arxiv.org/abs/1506.01497?context=cs).

## Training
### Domain-adaptive pre-training
To recreate a MMB DegenPos model, the base pre-trained BlenderBot3B model must first be further pre-trained on an image dataset, COCO Captions:
```
parlai tm \
-t coco_caption \
--include-rest-val True \
--include-image-token False \
--activation gelu \
--attention-dropout 0.0 \
--batchsize 128 \
--dropout 0.1 \
--fp16 True \
--gradient-clip 0.1 \
--label-truncate 128 \
--log-every-n-secs 30 \
--lr-scheduler reduceonplateau \
--max-train-time 169344.0 \
--model-parallel True \
--model image_seq2seq \
--init-model zoo:blender/reddit_3B/model \
--dict-file zoo:blender/reddit_3B/model.dict \
--embedding-size 2560 \
--ffn-size 10240 \
--n-decoder-layers 24 \
--n-encoder-layers 2 \
--n-heads 32 \
--n-positions 128 \
--variant prelayernorm \
--text-truncate 128 \
--truncate 128 \
--dict-tokenizer bytelevelbpe \
--fp16-impl mem_efficient \
--optimizer adam \
--update-freq 2 \
--history-add-global-end-token end \
--delimiter '  ' \
--lr-scheduler-patience 3 \
--warmup-updates 100 \
--multitask-weights 1,1 \
--relu-dropout 0.0 \
--save-after-valid True \
--skip-generation True \
-lr 7e-06 \
-vtim 1800 \
-vmm min \
-vmt ppl \
-vp 10 \
-vme 24000 \
--image-fusion-type early \
--n-segments 2 \
--n-image-channels 100 \
--model-file ${DOMAIN_PRETRAINED_MODEL_PATH}
```

### Fine-tuning
After the round of domain-adaptive pre-training above, the model must be fine-tuned on the dialogue datasets and on the Image-Chat dataset, using a degendering teacher:
```
parlai tm \
-t genderation_bias:controllable_task:blended_skill_talk,genderation_bias:controllable_task:convai2:normalized,genderation_bias:controllable_task:empathetic_dialogues,genderation_bias:controllable_task:wizard_of_wikipedia,genderation_bias:controllable_task:image_chat:Generation \
--prepend-personality True \
--image-mode faster_r_cnn_152_32x8d \
--include-image-token False \
--category-frac 0.75 \
--activation gelu \
--attention-dropout 0.0 \
--batchsize 128 \
--dropout 0.1 \
--fp16 True \
--gradient-clip 0.1 \
--label-truncate 128 \
--log-every-n-secs 30 \
--lr-scheduler reduceonplateau \
--max-train-time 169344.0 \
--model-parallel True \
--model image_seq2seq \
--dict-file zoo:blender/reddit_3B/model.dict \
--embedding-size 2560 \
--ffn-size 10240 \
--n-decoder-layers 24 \
--n-encoder-layers 2 \
--n-heads 32 \
--n-positions 128 \
--variant prelayernorm \
--text-truncate 128 \
--truncate 128 \
--dict-tokenizer bytelevelbpe \
--fp16-impl mem_efficient \
--optimizer adam \
--update-freq 2 \
--history-add-global-end-token end \
--delimiter $'\n' \
--lr-scheduler-patience 3 \
--warmup-updates 100 \
--multitask-weights stochastic \
--relu-dropout 0.0 \
--save-after-valid True \
--skip-generation True \
-lr 1e-06 \
-veps 0.25 \
-vmm min \
-vmt ppl \
-vp 10 \
--init-model ${DOMAIN_PRETRAINED_MODEL_PATH} \
--image-fusion-type early \
--n-segments 2 \
--n-image-channels 100 \
--model-file ${FINETUNED_MODEL_PATH}
```

## Inference
After fine-tuning, run the following command to interact with your model:
```
python parlai/scripts/safe_interactive.py \
-t blended_skill_talk \
-mf ${FINETUNED_MODEL_PATH} \
--model projects.multimodal_blenderbot.agents:BiasAgent \
--delimiter $'\n' \
--beam-block-ngram 3 \
--beam-context-block-ngram 3 \
--beam-min-length 20 \
--beam-size 10 \
--inference beam \
--model-parallel False
```

## Safety
We have undertaken several steps to make this bot safer. The original BlenderBot model was released with a `safe_interactive.py` script using a state-of-the-art safety classifier system on user and bot messages, as discussed [here](https://parl.ai/projects/recipes/), and we do so here as well. In *this* model, we additionally incorporate several strategies not found in previous systems:

1. The bot was trained to distinguish between utterances containing gendered words and utterances without gendered words, and during inference, the bot will be set to respond with utterances without gendered words, to reduce potential gender bias. 
   - Compared to a bot not trained to be sensitive to gendered words, this bot reduces the frequency of utterances with male words by a factor of 9 and with female words by a factor of 4.
2. When responding to an image, the bot was trained to distinguish between utterances categorized as “positive” or “neutral” in tone and utterances categorized as “negative”. During inference, the bot is able to respond to an image with only “positive” or “neutral” utterances given the appropriate context string.
    - We find that, in general, “positive” or “neutral” utterances are much less likely to be classified as containing offensive language than “negative” utterances.

Even with these steps, however, we have not addressed all possible safety concerns with this model, and we do not make any kind of guarantee that the model will not produce unsafe or offensive responses. (For instance, even with these measures, 10% of responses given examples from the Image-Chat validation set are still flagged as offensive by the safety classifier presented in [the “Build it break it fix it for dialogue safety” paper of Dinan et al., 2019](https://arxiv.org/abs/1908.06083). However, in this case we have not incorporated the safety features of the `safe_interactive.py` script described above, where the safety classifier mitigates a number of these concerns -- but not all.) If you choose to train or use this model, you do so entirely at your own risk.
