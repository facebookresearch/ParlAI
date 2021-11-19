# Task Oriented Dialogue (TOD): Agents, Worlds, Scripts, etc

### _Teaching Models new APIs: Domain-Agnostic Simulators for Task Oriented Dialogue_

Moya Chen, Paul A. Crook, Stephen Roller

## Abstract 

We demonstrate that large language models are able to simulate Task Oriented Dialogues in novel domains, provided only with an API implementation and a list of goals. We show these simulations can formulate online, automatic metrics that correlate well with human evaluations. Furthermore, by checking for whether the User's goals are met, we can use simulation to repeatedly generate training data and improve the quality of simulations themselves. With no human intervention or domain-specific training data, our simulations bootstrap end-to-end models which achieve a 37% error reduction in previously unseen domains. By including as few as 32 domain-specific conversations, bootstrapped models can match the performance of a fully-supervised model with 10× more data. To our knowledge, this is the first time simulations have been shown to be effective at bootstrapping models without explicitly requiring any domain-specific training data, rule-engineering, or humans-in-the-loop.

## Paper

[Link to arXiv](https://arxiv.org/abs/2110.06905)

# Explanation of content in project

This directory contains code for executing conversations for task-oriented dialogue (ex. setting an alarm, asking for the time) in a structured format. We introduce this structured format then go into the operational details for our setup: dataset generation + model training, simulation script usage, then give an overview of scripts in this folder. We then go into details of the specific datasets that we use as well as how to download and interact with our pre-trained models. 

As a terminology note, while the paper uses "Assistant" throughout, the same speaker is generally referred to as the "System" throughout code and documentation. 

## Conversation structure

In task oriented dialogue, we have a user (with some goal) that requests some form of action out of an assistant system. This assistant system normally has some external knowledge-base with to which it can interact with via APIs. 

To model this, we begin each episode with a grounding stage where:
1. an api schema agent gives a description string of the API to an api call and api response agents
2. a goal agent gives a target goal string to a user utterance agent to start the conversation

During the 'parlay' or normal conversational phase, we have four agents that speak in looped turns:
1. User utt agent
2. System API call agent
3. API response agent
4. System utt agent

In analogy to more traditional TOD-setups, one can think of the api call agent as dialogue state tracking and the system utt agent as natural language generation. Since many TOD-systems these days combine both dialogue state tracking and natural language generation into one model, we assume that the api call and system agents are the same.

To prevent leakage of information between agents during the parlay phase, each agent only observes only its own output and that of the agent which speaks immediately before. 

## Dataset setup + Model Training

See `parlai/core/tod/tod_agents.py` for information on how to build agents and teachers for a specific dataset.  

Of the agents described in the conversation, only the User and System need to be trained with generative models. These can be trained as normal ParlAI models (ie.`parlai train_model -t <insert task> -mf <model file path> -m <model type>`) using System- and UserSimulator- Teachers created via the documentation in the `tod_agents.py` file mentioned above.

## Simulation Script Usage
Use `python parlai/scripts/tod_world_script.py` or `parlai tod_world_script` (or the corresponding `distributed_` prefixed versions) to generate model-model chats. Arguments to the script are listed in file. Note that it is oftentimes preferable to use the `python ..` rather than `parlai ..` form of this command, especially if one has model or agent specific flags, due to argument order parsing. 

As a quick example, we provide

`parlai tod_world_script -o projects/tod_simulator/tod_world_configs/google_sgd_simulation_dump_data.json`

as an example of printing the validation data from Google SGD Out of Domain through the simulation script. 

Additionally, use this to specify a conversation where all of the agents take human input from the command line: 

```
parlai tod_world_script --system-model parlai.agents.local_human.local_human:LocalHumanAgent --user-model parlai.agents.local_human.local_human:LocalHumanAgent --api-resp-model parlai.agents.local_human.local_human:LocalHumanAgent --api-schema-grounding-model parlai.agents.local_human.local_human:LocalHumanAgent --goal-grounding-model parlai.agents.local_human.local_human:LocalHumanAgent
```

(which is the same as `parlai tod_world_script -o projects/tod_simulator/tod_world_configs/all_human.json`, included for convenience)

Defaults are provided for the grounding agents but must be specified for the rest. Pretrained model locations can also be specified for the user and system with `--user-model-file` and `--system-model-file` arguments respectively. Since the system agent + api call agent are assumed to be the same, we only specify the 5 distinct agents, rather than 6.

Further documentation of the simulation world and simulation world metrics are described in `parlai/core/tod/tod_world.py` and `parlai/core/tod/world_metrics.py`, respectively. 

## Scripts in `script` directory of this folder

**cleanup\_conversation.py**
As a convenience, we also add a script for parsing the output conversation of the TOD Script into a format slightly more ameniable to ACUTE-Eval. While the raw output of the TOD Script could be used as well, the provided cleanup script does things like remove API utterances + Goals. 

**do\_get\_passing\_only\_on\_dir.py**
Uses `get_passing_only.py` internaly to run on a directory

**get\_al\_samples\_for\_gsgd.py**
Gets active learning samples out of Google SGD's OutDomainSystemTeacher train set based on worst-performing API calls as extracted from `get_passing_only.py`. 

**get\_api\_data.py**
For models trained with `tod_distributed_uber_script.py` that have `--api-jga-record` set to `True`, this will automatically pull per-api Google SGD Out-of-Domain JGA and simulation success statistics.

**get\_interdistinct\_on\_conversations.py**
Deprecated script to calculate interdistinct metrics for simulation conversations. (Included for completeness.)

**get\_passing\_only.py**
Given a conversation generated from `tod_world_script`, outputs statistics about performance of different APIs. 

**get\_quick\_eval\_stats.py**
For models trained with `tod_distributed_uber_script.py`, this quickly grabs evaluation and model-model simulation data into a comma-separated format.

**tod\_distributed\_uber\_multiwoz\_script.py**
Version of `tod_distributed_uber_script.py` but with MultiWoz v2.2 as the primary task rather than Google SGD Out-of-Domain. (Included for completeness.)

**tod\_distributed\_uber\_script.py**
Multi-step train, evaluation, and data generation script used in Simulations paper. Uses Google SGD Out-of-Domain as primary dataset; note "STANDALONE\_API\_FILE\_PATH" that needs to be set in file. Makes use of `do_get_passing_only_on_dir.py` and `get_al_samples_for_gsgd.py`; use `get_passing_only.py` and `get_api_data.py` after the fact for analysis. 

Note that this script is intended to be run in a SLURM environment matching that of the Simulations paper authors. It is unknown how the script performs in other settings but is included as a reference.

## Tasks used in the paper 

See the appendix of [the paper](https://arxiv.org/abs/2110.06905) (or the description of the task in ParlAI Task List) for explanations of these datasets. Below, we include the dataset name, the command to run the `SystemTeacher` relevant for each of the datasets, and any other notable details. Other agents and teachers for the dataset are specified in the relevant task `agent.py` files. 

### Pretraining Tasks

* Google SGD In-Domain 
    * `parlai dd -t google_sgd_simulation_splits:InDomainSystemTeacher`
* MetalWoz
    * `parlai dd -t metalwoz:SystemTeacher`
* MSR_E2E
    * `parlai dd -t msr_e2e:SystemTeacher`
    * Note that due to the lack of annotations in this dataset, this System Teacher *only* includes utterance turns
* Multidogo
    * `parlai dd -t multidogo:SystemTeacher`
* MultiWoz 
    * We use a fb-internal pre-processing of MultiWoz, based on MultiWoz v2.1 and do not open source it at this time.
* Taskmaster
    * `parlai dd -t taskmaster:SystemTeacher`
* Taskmaster2
    * `parlai dd -t taskmaster2:SystemTeacher`
* Taskmaster3 (TicketTalk)
    * `parlai dd -t taskmaster3:SystemTeacher`

### Experimentation Tasks

* Google SGD Out-of-Domain
   * `parlai dd -t google_sgd_simulation_splits:OutDomainSystemTeacher`
* MultiWoz (not currently included in paper) 
   * `parlai dd -t multiwoz_v22:SystemTeacher`
   * This is a preprocessing of the dataset based on MultiWoz v2.2. Though utterances are the same as used for pre-training, API Call and API Response structures aer different. 

See "scripts in project directory" for scripts associated with training, evaluation, and data generation. 

## Pretrained models

We release Schema-Aware and Schema-Agnostic version of our intermediate task-pretraining. One can see the outputs of these models by running

```
parlai dd -t google_sgd_simulation_splits:OutDomainSystemTeacher -mf zoo:tod/tod_base_yes_api/model  --skip-generation false --api-schemas true
```

for the Schema-Aware version of the model and

```
parlai dd -t google_sgd_simulation_splits:OutDomainSystemTeacher -mf zoo:tod/tod_base_no_api/model  --skip-generation false --api-schemas false 
```

for the Schema-Agnostic version. 

Note the path names of the model files; they are `zoo:tod/tod_base_{yes,no}_api/mode` where "yes" corresponds to Schema-Aware and "no" corresponding to Schema-Agnostic. Care must be taken to specify `--api-schemas` correctly since task-setting flags are parsed from teacher-specific flags and not from model files. 

These models are both based on a BART-large (400 million paramater) base model. Hyperparameters for training can be found in the paper; tasks are listed in "Pretraining Tasks" above. 
