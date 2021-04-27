# Option Aliases

This folder contains a set of "option aliases" that are automatically packaged
and provided with ParlAI. They are used as shorthand for

## Adding option aliases

Simply adding `.opt` file to this folder is enough to add an alias. The
directory structure is respected, so `parlai/options/myfolder/myalias.opt` can
be invoked using `-o myfolder/myalias`.

For quality assurance purposes, we request new option aliases respect the
following conventions:

- Only include the very minimum number of options you wish to specify with this
  alias.
- Pipe your opt file through `jq -S .` so keys are always in alphabetical order.
- Add an entry to "docs.py" briefly describing your option. The full expansion
  will be automatically rendered, but a human-friendly summary should be added.
