## BlenderBot 3: A 175B-parameter, publicly available chatbot that improves its skills & safety over time

<p align="center"><img width="30%" src="anim.gif" /></p>

- BlenderBot 3 (BB3) is a 175B-parameter, publicly available chatbot released with model weights, code, datasets, and model cards. We’ve deployed it in a live interactive conversational [AI demo](https://blenderbot.ai/).

- BB3 searches the internet to chat about nearly any topic, and is designed to learn how to improve its skills and safety through natural conversations and feedback from people "in the wild."

- [Initial experiments](https://parl.ai/projects/fits) show that as people interact with the model, the more it can learn, particularly using our new [Director](https://parl.ai/projects/director/) architecture.

- Learning from people "in the wild" is not straightforward. We have developed [new techniques](https://parl.ai/projects/trollhunting) that enable learning from helpful teachers while avoiding learning from people who are trying to trick the model into unhelpful or toxic responses.

-  We are committed to sharing participating organic conversational data collected from the live demo as well as model snapshots in the future. The goal is to help the community build ever-improving AI systems that can interact with people in safer and more helpful ways.

- [Read more about it here.](https://ai.facebook.com/blog/blenderbot-3-a-175b-parameter-publicly-available-chatbot-that-improves-its-skills-and-safety-over-time/)

## <a id="papers">Papers</a>

**BB3 main technical report**:

* [BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage](https://github.com/facebookresearch/ParlAI/blob/main/projects/bb3/BB3_main_tech_report.pdf).
Kurt Shuster†, Jing Xu†, Mojtaba Komeili†, Da Ju†, Eric Michael Smith, Stephen Roller, Megan Ung, Moya Chen, Kushal Arora+, Joshua Lane, Morteza Behrooz, William Ngan, Spencer Poff, Naman Goyal, Arthur Szlam, Y-Lan Boureau, Melanie Kambadur, Jason Weston.

We are also concurrently releasing **two companion papers** describing key innovations:

* [Learning New Skills after Deployment: _Improving open-domain internet-driven dialogue with human feedback_](https://parl.ai/projects/fits).
Jing Xu, Megan Ung, Mojtaba Komeili, Kushal Arora, Y-Lan Boureau, Jason Weston.

* [Learning from data in the mixed adversarial non-adversarial case:
_Finding the helpers and ignoring the trolls_](https://parl.ai/projects/trollhunting). Da Ju, Jing Xu, Y-Lan Boureau, Jason Weston.


Finally, BB3 is dependent on other recent work we have published, in particular [SaFeRDialogues: Taking Feedback Gracefully after Conversational Safety Failures](https://parl.ai/projects/saferdialogues/), [DIRECTOR: Generator-Classifiers For Supervised Language Modeling](https://parl.ai/projects/director/) and [SeeKeR: An Open source Search-Augmented Language Model](https://parl.ai/projects/seeker/).
BB3 also builds on all our previous work, including [BB1](https://parl.ai/projects/recipes/) and [BB2](https://parl.ai/projects/blenderbot2/) and related papers. See our team's projects and publications [here](https://parl.ai/projects/).

## Logbook

We are also releasing a BB3 Logbook documenting the development of our system, available [here](https://github.com/facebookresearch/ParlAI/blob/main/projects/bb3/bb3_logbook.pdf).

## <a id="models">Models</a>

We are releasing three model sizes:  3B, 30B and 175B.

The 3B and 30B models are available in the [ParlAI model zoo](https://parl.ai/docs/zoo.html).
- BlenderBot 3 3B: `--model-file zoo:bb3/bb3_3B/model`
- BlenderBot 3 30B: `--model-file zoo:bb3/bb3_30B/model`

The BB3 175B model is shared by request [here](https://docs.google.com/forms/d/e/1FAIpQLSfRzw8xVzxaxgRyuodTZtkcYADAjzYjN5gcxx6DMa4XaGwwhQ/viewform).

## <a id="model-cards">Model card</a>

The BB3 model card is available [here](https://github.com/facebookresearch/ParlAI/blob/main/parlai/zoo/bb3/model_card.md).

## <a id="model-cards">Data card</a>

See [here](https://github.com/facebookresearch/ParlAI/blob/main/parlai/zoo/bb3/data_card.md) for the BB3 data card.

## <a id="datasets">Training datasets</a>

We are releasing the new [FITS](https://parl.ai/projects/fits) dataset of Feedback on Internet Talk & Search used to train BB3.

Training is also multi-tasked with all the existing datasets from BB1 and BB2, e.g. the existing [BST tasks](https://parl.ai/projects/bst) from [BlenderBot 1](https://parl.ai/projects/recipes), and [Multi-Session Chat](https://parl.ai/projects/msc) and [Wizard of the Internet](https://parl.ai/projects/sea) from BB2. To train for safety we use the [SaFeRDialogues](https://parl.ai/projects/saferdialogues/) and [BAD dataset](https://parl.ai/projects/safety_recipes). In addition, we use a number of QA tasks and task-oriented dialogue datasets that are all available in [ParlAI](https://parl.ai/docs/tasks.html). See
the [tech report](https://github.com/facebookresearch/ParlAI/blob/main/projects/bb3/BB3_main_tech_report.pdf) for the full list.

See the [ParlAI quickstart](http://www.parl.ai/docs/tutorial_quick.html) for help.


### BB3 Module Tasks

These tasks are used to train BB3's modules, and are hence adapted slightly, e.g. with appropriate control tokens provided in the context (see the paper for a full explanation). We thus provide here the explicit setup used to train BB3. The following multitask teachers train each of the BB3 modules:

- `projects.bb3.tasks.module_level_tasks:AlwaysSearchTeacher`
- `projects.bb3.tasks.module_level_tasks:MaybeSearchTeacher`
- `projects.bb3.tasks.module_level_tasks:MemoryDecisionTeacher`
- `projects.bb3.tasks.module_level_tasks:SearchQueryGenerationTeacher`
- `projects.bb3.tasks.module_level_tasks:MemoryGenerationTeacher`
- `projects.bb3.tasks.module_level_tasks:MemoryKnowledgeGenerationTeacher`
- `projects.bb3.tasks.module_level_tasks:SearchKnowledgeGenerationTeacher`
- `projects.bb3.tasks.module_level_tasks:EntityKnowledgeGenerationTeacher`
- `projects.bb3.tasks.module_level_tasks:SearchDialogueGenerationTeacher`
- `projects.bb3.tasks.module_level_tasks:EntityDialogueGenerationTeacher`
- `projects.bb3.tasks.module_level_tasks:MemoryDialogueGenerationTeacher`
- `projects.bb3.tasks.module_level_tasks:VanillaDialogueGenerationTeacher`

## <a id="code">Code</a>

### BB3 3B Model: Download + Interact

We provide the BB3 3B model in ParlAI's model zoo. You can interact with the model via the following:

```bash
parlai interactive --model-file zoo:bb3/bb3_3B/model --init-opt gen/r2c2_bb3
```

### BB3 30B Model: Download

You can download the BB3 30B model via the following command:

```
wget http://parl.ai/downloads/_models/bb3/bb3_30B/consolidated.pt
```

### BB3 175B Model: Download

You will receive instructions for downloading the 175B model if approved.

### BB3 30B/175B: Interact

(Docs adapted from [OPT docs](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/download_opt175b.md#run-the-api))

After downloading the consolidated BB3 30B or 175B checkpoints, you will need to reshard according to your GPU resources. The 30B model checkpoint requires 56gb of GPU memory, while the 175B checkpoint requires 384GB of GPU memory.

After identifying how many GPUs you will need to run the models, you can use the following commands to reshard appropriately:

**BB3 30B**:
```
CONSOLIDATED=/path/to/bb3_30B/consolidated/
RESHARD=/save/path/to/bb3_30B/resharded/
MP=8
python -m metaseq.scripts.reshard_model_parallel $CONSOLIDATED/consolidated $MP --save-prefix $RESHARD/reshard
```

**BB3 175B**:
```
CONSOLIDATED=/path/to/bb3_175B/consolidated/
RESHARD=/save/path/to/bb3_175B/resharded/
MP=16
python -m metaseq.scripts.reshard_model_parallel $CONSOLIDATED/consolidated $MP --save-prefix $RESHARD/reshard
```

Then, you can follow the instructions for [running an API in `metaseq`](https://github.com/facebookresearch/metaseq/blob/main/docs/api.md) to spin up the API. You will need to update the constants in `metaseq/service/constants.py` to point to the right directories -- specifically, set the [`CHECKPOINT_FOLDER`](https://github.com/facebookresearch/metaseq/blob/c9c817d2a230519c2865264bafdf45931afa02e6/metaseq/service/constants.py#L32) to where you have downloaded the models.

Note that the gpt2-merges.txt and gpt2-vocab.json files in projects/OPT/assets/ will need to be moved to the corresponding directories defined in the constants.py file. You can directly download them with:

```
cd /path/to/resharded-weights
wget https://github.com/facebookresearch/metaseq/raw/main/projects/OPT/assets/gpt2-merges.txt
wget https://github.com/facebookresearch/metaseq/raw/main/projects/OPT/assets/gpt2-vocab.json
```

Once you have an API up and running, you can utilize the BB3 agents we provide to interact with the model:

```
parlai interactive --init-opt gen/opt_bb3 --opt-server API_SERVER --loglevel debug --raw-search-server RELEVANT_SEARCH_SERVER
```

### Holistic Bias
Commands for evaluating the BB3 models on the [HolisticBias](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias) dataset of sentences with demographic terms can be found [here](https://github.com/facebookresearch/ParlAI/blob/main/projects/bb3/holistic_bias/README.md).


## <a id="interaction-data">Live deployment / demo</a>

The live demo is available [here](https://blenderbot.ai/). We have been placing ads&mdash;and conducting user studies&mdash;to allow members of the public to participate in using the system, and to optionally record interaction and feedback data for use by the research community.
See the [tech report](https://github.com/facebookresearch/ParlAI/blob/main/projects/bb3/BB3_main_tech_report.pdf) for full details and evaluation metrics thus far.


## <a id="interaction-data">Sharing interaction data & model improvements: _coming next!_</a>

We are committed to openly sharing participating de-identified organic conversational data collected from the live demo as well as model snapshots **in the future**, as soon as we have collected enough data and assessed quality, safety and other issues. The overall goal of this project is to help the community build ever-improving open AI systems that can interact with people in safer and more helpful ways.


