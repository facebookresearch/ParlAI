# Command Line Usage

This contains the command line usage for each of the standard scripts we
release. These are each included in `parlai/scripts`, and all may
be invoked with the "parlai" supercommand.

The parlai supercommand may be invoked from the command line by running
`parlai` after installing ParlAI. Its default output looks like this:

```none
usage: parlai [-h] [--helpall] COMMAND ...

       _
      /")
     //)
  ==//'=== ParlAI
   /

optional arguments:
  -h, --help               show this help message and exit
  --helpall                show all commands, including advanced ones.

Commands:

  display_data (dd)        Display data from a task
  display_model (dm)       Display model predictions.
  eval_model (em, eval)    Evaluate a model
  train_model (tm, train)  Train a model
  interactive (i)          Interactive chat with a model on the command line
  safe_interactive         Like interactive, but adds a safety filter
  self_chat                Generate self-chats of a model

The remainder of this page describes each of the commands, their possible arguments,
and some examples of their usage.
```

```{include} cli_usage.inc
```
