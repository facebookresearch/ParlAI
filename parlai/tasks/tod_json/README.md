# TOD Json Task Agent

Takes a .jsonl conversation output from model-model chats from `tod_world_script.py`, puts it into the TOD intermediate conversations format so we can use it in a variety of different teachers.

For example, to see the display of the data:
```
parlai dd -t tod_json:SystemTeacher --jsonfile-datapath example_data.jsonl
parlai dd -t tod_json:UserSimulatorTeacher --jsonfile-datapath example_data.jsonl
```

See the file `example_data.json` in this directory for the format.
