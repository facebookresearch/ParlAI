# Inference Guided Dialogue 

### Data Configurations

There are three main options for training: 

- `infqa_response`: First generate the question given the dialogue history, then generate an answer to it, then generate a response given all the context (dialogue history + question + answer).
- `inq_aresponse`: First generate ta answer to a given dialogue history and inference question, then generate a response given all the context. 
- `response` (default): Generate only the final response given the dialogue history + question + answer. 

Pass these key values as command line arguments via `-gt` or `--generation_target`. 
- i.e. `parlai dd -t inference_guided_dialogue -gt infqa_response` 


### Fine-tuning BlenderBot with this data

Sample command for BlenderBot 90M ([Reference](https://parl.ai/projects/recipes/))
```
parlai train_model \
    -t inference_guided_dialogue \
    -m transformer/generator \
    --init-model zoo:tutorial_transformer_generator/model \
    --dict-file zoo:tutorial_transformer_generator/model.dict \
    --embedding-size 512 \
    --n-layers 8 \
    --ffn-size 2048 \
    --dropout 0.1 \
    --n-heads 16 \
    --learn-positional-embeddings True \
    --n-positions 512 \
    --variant xlm \
    --activation gelu \
    --fp16 True \
    --text-truncate 512 --label-truncate 128 \
    --dict-tokenizer bpe --dict-lower True \
    -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 \
    -veps 0.25 \
    --betas 0.9,0.999 \
    --update-freq 1 \
    --attention-dropout 0.0 \
    --relu-dropout 0.0 \
    -vp 15 \
    -stim 60 \
    -vme 20000 \
    -bs 16 \
    -vmt ppl \
    -vmm min \
    --save-after-valid True \
    --model-file /tmp/test_train_90M # directory to save to 
```

For full set of arguments, refer to the ParlAI docs: https://www.parl.ai/docs/cli_usage.html#train-model

There are more command line arguments depending on the base model. For instance, BlenderBot is based on BART: https://www.parl.ai/docs/agent_refs/bart.html#bart 


# TODOs

Make train/val/test split for training. 