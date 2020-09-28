# Crowdsourcing tasks

Code for crowdsourcing tasks using Mephisto. See https://github.com/facebookresearch/mephisto/blob/master/docs/quickstart.md for a guide to quickly getting started with Mephisto.

## Saving data

By default, Mephisto data is saved in the following directory:
```
<datapath>/data/runs/NO_PROJECT/<project_id>/<task_run_id>/<assignment_id>/<agent_id>/data
```
- On the command line, pass in `mephisto.datapath=<new_path> ` to change the default datapath
- The `NO_PROJECT` and `data` subfolders may be renamed in later versions of Mephisto
- `<agent_id>` can be mapped to MTurk `worker_id` with the `workers` table in the Mephisto SQLite3 database