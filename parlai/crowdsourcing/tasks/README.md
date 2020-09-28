# Crowdsourcing tasks

Code for crowdsourcing tasks using Mephisto.

## Saving data

By default, Mephisto data is saved in the following directory:
```
<datapath>/data/runs/NO_PROJECT/<project_id>/<task_run_id>/<assignment_id>/<agent_id>/data
```
- The `NO_PROJECT` and `data` subfolders may be renamed in later versions of Mephisto
- `<agent_id>` can be mapped to MTurk `worker_id` with the `workers` table in the Mephisto SQLite3 database