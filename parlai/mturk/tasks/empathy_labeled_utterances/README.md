Instructions for running a MTurk task to annotate generations with empathy labels. Additional notes are at the bottom.

## File Structure
```
├── conf
│   ├── mturk_sandbox.yaml
│   ├── mturk.yaml
│   └── sample.yaml
├── data
│   └── generations_test.jsonl
├── run.py # main entry point
├── README.md
├── scripts
│   ├── examine_results.py
│   ├── turn_annotations_blueprint.py
│   └── util.py
├── task_config
│   ├── annotation_buckets_radio.json
│   ├── onboarding.json
│   └── task_data.jsonl # this is just default dummy data; see ./data for conversations from generator
└── webapp
```

## Running Tasks
There are three config files, one for running locally, one for Mechanical Turk Sandbox and the other for the "main" or "real" Mechanical Turk. You'll need to register a requester before running live on MTurk or the sandbox. Register separate requesters for the sandbox (name should end with `_sandbox`) and main.
```bash
    # first register requester
    # this will update ~/.aws/credentials
    # also note, you'll probably need to use an older Mephisto tag for the registration (v0.2 worked for me)
    mephisto register mturk \
        name=my_mturk_user \
        access_key_id=$ACCESS_KEY\
        secret_access_key=$SECRET_KEY

    # local on port 8888; if using a jumphost specify a mapping to port 8888 or change port in the config
    python run.py conf=sample

    # sandbox
    python run.py mephisto/architect=heroku mephisto.provider.requester_name=REQUESTER_NAME conf=mturk_sandbox

    # main
    python run.py mephisto/architect=heroku mephisto.provider.requester_name=REQUESTER_NAME conf=mturk
```
### Examining Hits
```bash
    # you'll first be prompted for the requester name used above
    # then you'll be able to view all tasks (a) or unreviewed tasks (u)
    # unreviewed options allows to accepting/rejecting/passing each hit and optionally blocking workers
    python scripts/approve_results.py

    # export annotations to json and examine
    python scripts/export_to_file.py --task_names empathy_main label_utterances_empathy
    python scripts/evaluate_results.py
```

## Discarding Hits
If you have outstanding HITs you no longer need, run
```bash
    MEPHISTO_DIR/mephisto/scripts/mturk/cleanup.py
```
You'll need the requester and task name.

## General Notes
* when testing with sandbox, the onboarding page (defined by the onboarding_qualification name in the config file) is only shown once per user; if making any changes here, just update the name to view again
* run MEPHISTO_DIR/mephisto/scripts/mturk/cleanup.py to remove any launched HITs you no longer want (works for sandbox as well)
* Mephisto creates a Heroku app to "host" the task; if you get an error that there's too many concurrent apps running, just delete the oldest apps [here](https://dashboard.heroku.com/apps)