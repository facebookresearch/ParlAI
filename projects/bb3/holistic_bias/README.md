# Commands for running HolisticBias tests on BlenderBot 3

## Setup

```
# Setting up the ResponsibleNLP repo
git clone git@github.com:facebookresearch/ResponsibleNLP.git
cd ResponsibleNLP
python setup.py develop
```

## Measuring perplexities and computing bias metrics

```
### BB3-3B
# (Tested with 8 32-GB GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python projects/bb3/holistic_bias/scripts/eval_3b_model.py \
-mf zoo:bb3/bb3_3B/model \
--mutators skip_retrieval_mutator \
--task holistic_bias \
--world-logs ${LOG_JSONL_PATH} \
--batchsize 64 \
--use-blenderbot-context False # BB3 didn't always see personas during training

### OPT-175B (0-shot)
python projects/bb3/holistic_bias/scripts/eval_175b_model.py \
-m projects.bb3.agents.opt_api_agent:BB3OPTAgent \
--server ${SERVER} \
--log-every-n-secs 30 \
--batchsize 30 \
--skip-generation true \
--module vrm \
--metrics all \
--include-prompt True \
--num-shots 0 \
--include-substitution-tokens False \
--add-speaker-prefixes False \
--max-prompt-len 2000 \
--world-logs ${LOG_JSONL_PATH} \
--use-blenderbot-context False

### OPT-175B (few-shot)
python projects/bb3/holistic_bias/scripts/eval_175b_model.py \
-m projects.bb3.agents.opt_api_agent:BB3OPTAgent \
--server ${SERVER} \
--log-every-n-secs 30 \
--batchsize 30 \
--skip-generation true \
--module vrm \
--metrics all \
--include-prompt True \
--num-shots -1 \
--include-substitution-tokens False \
--add-speaker-prefixes False \
--max-prompt-len 2000 \
--world-logs ${LOG_JSONL_PATH} \
--use-blenderbot-context False

### BB3-175B
python projects/bb3/holistic_bias/scripts/eval_175b_model.py \
-m projects.bb3.agents.opt_api_agent:BB3OPTAgent \
--server ${SERVER} \
--log-every-n-secs 30 \
--batchsize 30 \
--skip-generation true \
--module vrm \
--metrics all \
--include-prompt False \
--num-shots 0 \
--include-substitution-tokens False \
--add-speaker-prefixes False \
--max-prompt-len 2000 \
--world-logs ${LOG_JSONL_PATH} \
--use-blenderbot-context False
```

## Re-computing bias metrics, given existing perplexity measurements

```
# (Run in the ResponsibleNLP repo root folder)
python holistic_bias/src/bias_measurements.py \
--world-logs ${LOG_JSONL_PATH}
```
