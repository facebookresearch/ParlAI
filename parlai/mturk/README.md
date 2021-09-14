The code in `parlai/mturk/` has been deprecated. All code for running crowdsourcing tasks can now be found in [`parlai/crowdsourcing/`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing), and all crowdsourcing tasks now utilize the [Mephisto](https://github.com/facebookresearch/Mephisto) platform.

-----

Instructions on how to find all old `parlai/mturk/` code can be found in the sections below.

### core/, scripts/, webapp/

These folders can be restored by switching to the `final_mturk` tag of ParlAI:
```bash
git checkout final_mturk
```


### tasks/acute_eval/

The current Mephisto-based version of the ACUTE-Eval task can be found in [`parlai.crowdsourcing.tasks.acute_eval`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/acute_eval). The obsolete pre-Mephisto version of this task can be restored by switching to the `acute_eval` tag of ParlAI:

```bash
git checkout acute_eval
```


### tasks/convai2_model_eval/

You can restore this task by switching to the `convai2archive`
tag of ParlAI:

```bash
git checkout convai2archive
```


### tasks/dealnodeal/

You can restore this task by switching to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag of ParlAI:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/dealnodeal).


### tasks/image_chat/

The current Mephisto-based version of the human+model image-chat task can be found in [`parlai.crowdsourcing.tasks.model_chat`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/model_chat). The obsolete pre-Mephisto version of this task can be restored by switching to the `v0.10.0` tag of ParlAI:

```bash
git checkout v0.10.0
```


### tasks/light/

You can restore this task by switching to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag of ParlAI:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/light).


### tasks/multi_agent_dialog/

You can restore this task by switching to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag of ParlAI:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/multi_agent_dialog).


### tasks/personachat/

You can restore this task by switching to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag of ParlAI:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/personachat).


### tasks/personality_captions/

You can restore this task by switching to the `v0.10.0` tag of ParlAI:

```bash
git checkout v0.10.0
```


### tasks/qa_data_collection/

The current Mephisto-based version of the QA data collection task can be found in [`parlai.crowdsourcing.tasks.qa_data_collection`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/qa_data_collection). The obsolete pre-Mephisto version of this task can be restored by switching to the `qa_data_collection` tag of ParlAI:

```bash
git checkout qa_data_collection
```


### tasks/qualification_flow_example/

You can restore this task by switching to the `qualification_flow_example` tag of ParlAI:

```bash
git checkout qualification_flow_example
```


### tasks/react_task_demo/

You can restore this task by switching to the `react_task_demo` tag of ParlAI:

```bash
git checkout react_task_demo
```


### tasks/talkthewalk/

You can restore this task by switching to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag of ParlAI:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/talkthewalk).


### tasks/turn_annotations/

The current Mephisto-based version of the turn annotations task can be found in [`parlai.crowdsourcing.tasks.model_chat`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/model_chat). The obsolete pre-Mephisto version of this task can be restored by switching to the `turn_annotations` tag of ParlAI:

```bash
git checkout turn_annotations
```


### tasks/wizard_of_wikipedia/

You can restore this task by switching to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag of ParlAI:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/wizard_of_wikipedia).
