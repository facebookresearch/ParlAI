# Crowdsourcing tasks

Code for crowdsourcing tasks that use Mephisto. See https://github.com/facebookresearch/mephisto/blob/master/docs/quickstart.md for a guide to quickly getting started with Mephisto.

## Running tasks

Tasks are launched by calling the appropriate run script: for instance, an ACUTE-Eval run can be launched with `python parlai/crowdsourcing/tasks/acute_eval/run.py`, followed by any appropriate flags. All run parameters are set using [Hydra](https://github.com/facebookresearch/hydra): append the flag `-c job` to your run command to see a list of all available parameters, grouped by their package name (`mephisto.blueprint`, `mephisto.task`, etc.), which determines how they are called. Each run script has a YAML file of default parameters that will be loaded, found in the `conf/` subfolder of each task.
 
### Specifying your own YAML file
 
 The easiest way to specify a different YAML file is to create a new file, say, `my_params.yaml`, in the `conf/` subfolder of the task. Then, you can launch HITs with `python ${TASK_FOLDER}/run.py conf=my_params`.
 
 You also can specify a path to a YAML file existing *outside* of `${TASK_FOLDER}`: you will need to have your YAML file stored at a location `${CUSTOM_FOLDER}/conf/my_params.yaml`, and then you can add a `--config-dir ${CUSTOM_FOLDER}` string to the launch command above.
 
### Setting parameters on the command line

Suppose that your YAML file has a `task_reward` parameter defined as follows:
```
mephisto:
  task:
    task_reward: 0.5
```
If you want to quickly modify this parameter to, say, 0.6 without changing the YAML file, you can add a `mephisto.task.task_reward=0.6` string to your launch command.

### MTurk-specific task configuration

Here is a partial list of MTurk-specific parameters that can be set in YAML files or on the command line:
- `mephisto.task.task_title`: A short and descriptive title about the kind of task that the HIT contains. On the Amazon Mechanical Turk web site, the HIT title appears in search results and everywhere that the HIT is mentioned.
- `mephisto.task.task_description`: Includes detailed information about the kind of task that the HIT contains. On the Amazon Mechanical Turk web site, the HIT description appears in the expanded view of search results, and in the HIT and assignment screens.
- `mephisto.task.task_tags`: One or more words or phrases that describe the HIT, separated by commas. On MTurk website, these words are used in searches to find HITs.
- `mturk.worker_blocklist_paths`: The path to a text file containing a list of IDs of MTurk workers to soft-block, separated by newlines. Multiple paths can be specified, delimited by commas (i.e. `path1,path2,path3`).
 
### Running tasks live

By default, HITs run locally in sandbox mode. To run live HITs, add `mephisto.provider.requester_name=${REQUESTER_NAME} mephisto/architect=heroku` to your launch command, where `${REQUESTER_NAME}` is the MTurk requester name that you specified when setting up Mephisto.
 
## Saving data

By default, Mephisto data is saved in the following directory:
```
<mephisto_root_dir>/data/data/runs/NO_PROJECT/<project_id>/<task_run_id>/<assignment_id>/<agent_id>/data
```
- Internally, `<mephisto_root_dir>` defaults to `/scratch/${USER}/mephisto`.
- The `NO_PROJECT` and `data` subfolders may be renamed in later versions of Mephisto.
- `<agent_id>` can be mapped to MTurk `worker_id` with the `workers` table in the Mephisto SQLite3 database.
