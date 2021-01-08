The code in `parlai/mturk/` has been deprecated. All code for running crowdsourcing tasks has been migrated to the [Mephisto](https://github.com/facebookresearch/Mephisto) platform, and code for running Mephisto tasks can be found in [`parlai/crowdsourcing/`](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing).

Instructions on how to find all old code can be found in the sections below.

### core/, scripts/, webapp/

{{{TODO: add final tag before this version and how to switch to it}}}


### tasks/acute_eval/

The current Mephisto-based version of the ACUTE-Eval crowdsourcing task can be found in [`parlai.crowdsourcing.tasks.acute_eval`](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing/tasks/acute_eval). The obsolete pre-Mephisto version of this task, can be restored by switching to the `acute_eval` tag of ParlAI:

```bash
$ git checkout acute_eval
```


### tasks/convai2_model_eval/

The pre-Mephisto version can be restored by switching to the `convai2archive`
tag of ParlAI:

```bash
$ git checkout convai2archive
```


### tasks/dealnodeal/

{{{TODO}}}


### tasks/image_chat/

{{{TODO}}}


### tasks/light/

{{{TODO}}}


### tasks/multi_agent_dialog/

{{{TODO}}}


### tasks/personachat/

{{{TODO}}}


### tasks/personality_captions/

{{{TODO}}}


### tasks/qa_data_collection/

{{{TODO}}}


### tasks/qualification_flow_example/

{{{TODO}}}


### tasks/react_task_demo/

{{{TODO}}}


### tasks/talkthewalk/

{{{TODO}}}


### tasks/turn_annotations/

{{{TODO}}}


### tasks/wizard_of_wikipedia/

{{{TODO}}}
