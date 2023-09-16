# Multi-party model chat
An extension of the model chat that runs between the human crowdworker and mutliple model-controlled agents.
```.sh
# assuming you are in multi_model_chat directory
python -m parlai.crowdsourcing.tasks.model_chat.run \
--config-path "$(eval pwd)/hydra_configs" conf=multiparty task_dir="$(eval pwd)"
```

# Modules structure
Most of the components are inherited from the regular model chat and have the same functionalities.

The main exta piece here is the `agents.py` module which is in charge of creating custom agents for controlling the conversation flow and utterance responses.

The `ContextGenerator` class, which is part of the worlds, generates location descriptions and personas. There is a minimal implementation of it here with only 2 hard-coded settings. The users must re-implement that in practice.
