# Wizard of Internet

This is the crowdsourcing task from the Internet-Augmented Dialogue Generation paper ([link](https://arxiv.org/abs/2107.07566)).
It uses [Mephisto](https://github.com/facebookresearch/Mephisto) platform to collect dialogue data using human workers on Amazon Mechanical Turk.

## How to use
Having setup your ParlAI and Mephisto environment properly (make sure you can run Mephisto demos), you should be able to run this task easily. Most of the configurations for running task are in `conf/dev.yaml` file. Note the files needed in the `data` directory:
*sample_personas.txt* and *sample_locations.txt* are needed to create the curated personas.

You need to have a functional search server running, and sets its address in `search_server` in the `conf/dev.yaml` file. You may set the server up to search internet or any knowledge source of your choosing.
This server responds to the search requests sent by the worker who takes *wizard* role during this task:
It receieves a json with two keys: `q` and `n`, which are a string that is the search query, and an integer that is the number of pages to return, respectively.
It sends its response also as a json under a key named `response` which has a list of documents retrieved for the received search query. Each document is a mapping (dictionary) of *string->string* with at least 3 fields: `url`, `title`, and `content` (see [SearchEngineRetriever](https://github.com/facebookresearch/ParlAI/blob/70ee4a2c63008774fc9e66a8392847554920a14d/parlai/agents/rag/retrieve_api.py#L73) for more info on how this task interacts with the search server).

## Creating the dataset

Having collected data from crowdsourcing task, you may use `compile_resullts.py` to create your dataset, as a json file.
For example, if you called your task `wizard-of-internet` (you set this name in the config file that you ran with your task from `hydra_config`),
the following code creates your dataset as a json file in the directory specified by `--output-folder` flag:

```.python
python compile_results.py --task-name wizard-of-internet --output-folder=/dataset/wizard-internet
```
