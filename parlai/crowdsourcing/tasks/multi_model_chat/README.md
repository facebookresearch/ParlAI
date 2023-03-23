# Multi-party model chat
An extension of the model chat that runs between the crowdsourcer and mutliple model-controlled agents.
This task uses the main model_chat task from ParlAI crowdsourcing. 
```.sh
python run.py \
--config-path /private/home/komeili/dev/ParlAI/parlai_internal/crowdsourcing/projects/multilight/model_chat/hydra_configs
conf=prod_fixed \
task_dir=/private/home/komeili/dev/ParlAI/parlai_internal/crowdsourcing/projects/multilight/model_chat
```