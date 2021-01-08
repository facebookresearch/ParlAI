The code in `parlai/mturk/` has been deprecated. All code for running crowdsourcing tasks has been migrated to the [Mephisto](https://github.com/facebookresearch/Mephisto) platform, and code for running Mephisto tasks can be found in [`parlai/crowdsourcing/`](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing).

Instructions on how to find all old code can be found in the sections below.

### core/, scripts/, webapp/

{{{TODO: add final tag before this version and how to switch to it}}}


### tasks/acute_eval/

The current Mephisto-based version of the ACUTE-Eval crowdsourcing task can be found in [`parlai.crowdsourcing.tasks.acute_eval`](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing/tasks/acute_eval). The obsolete pre-Mephisto version of this task, can be restored by switching to the `acute_eval` tag of ParlAI:

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

If you require need to run
this evaluation, you may rewind back to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/dealnodeal).


### tasks/image_chat/

You can restore this task by switching to the `v0.10.0` tag of ParlAI:

```bash
git checkout v0.10.0
```


### tasks/light/

If you require need to run
this evaluation, you may rewind back to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/light).


### tasks/multi_agent_dialog/

If you require need to run
this evaluation, you may rewind back to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/multi_agent_dialog).


### tasks/personachat/

If you require need to run
this evaluation, you may rewind back to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag:

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

You can restore this task by switching to the `qa_data_collection` tag of ParlAI:

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

If you require need to run
this evaluation, you may rewind back to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/talkthewalk).


### tasks/turn_annotations/

You can restore this task by switching to the `turn_annotations` tag of ParlAI:

```bash
git checkout turn_annotations
```


### tasks/wizard_of_wikipedia/

If you require need to run
this evaluation, you may rewind back to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag:

```bash
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/wizard_of_wikipedia).
