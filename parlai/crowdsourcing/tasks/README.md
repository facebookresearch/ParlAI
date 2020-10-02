# Crowdsourcing tasks

Code for crowdsourcing tasks using Mephisto. See https://github.com/facebookresearch/mephisto/blob/master/docs/quickstart.md for a guide to quickly getting started with Mephisto.

## Running tasks

Tasks are launched by calling the appropriate run script: for instance, an ACUTE-Eval run can be launched with `python parlai/crowdsourcing/tasks/acute_eval/run.py`, followed by the appropriate flags. All flags are set using [Hydra](https://github.com/facebookresearch/hydra), and each run script has a default YAML file of settings that will be loaded, found in the `conf/` subfolder of each task. To specify your own YAML file, pass in the flag `conf=${PATH_TO_FILE}`.
(((DOES THIS WORK IF IT'S NOT IN THE CONF FOLDER?)))

## Saving data

By default, Mephisto data is saved in the following directory:
```
<datapath>/data/runs/NO_PROJECT/<project_id>/<task_run_id>/<assignment_id>/<agent_id>/data
```
- The `NO_PROJECT` and `data` subfolders may be renamed in later versions of Mephisto
- `<agent_id>` can be mapped to MTurk `worker_id` with the `workers` table in the Mephisto SQLite3 database
