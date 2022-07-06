# ParlAI chat task

Chat task using two ParlAI-style agents. Code can be extended to change the number of human agents or to add bot agents. See the Mephisto [README](https://github.com/facebookresearch/Mephisto/blob/main/examples/parlai_chat_task_demo/README.md) for more details, and see the [`model_chat/`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/model_chat) folder for an example of a chat task between a human and a model.

Since this task is imported from the Mephisto repo, `run.py` will look in `${MEPHISTO_REPO}/examples/parlai_chat_task_demo/conf/` for YAML configuration files by default. See [here](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks#specifying-your-own-yaml-file) for how to specify a custom folder for configuration files.

**NOTE**: See [parlai/crowdsourcing/README.md](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/README.md) for general tips on running `parlai.crowdsourcing` tasks, such as how to specify your own YAML file of configuration settings, how to run tasks live, how to set parameters on the command line, etc.
