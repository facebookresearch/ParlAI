# Opt Presets

Opt presets are a way to provide multiple options on the command line as
shorthand. Opt presets are bundled with ParlAI and may be used by simply
invoking the `-o preset_name` option within any ParlAI command.

You may also define your own options by placing them in `~/.parlai/opt_presets/`.
For example, creating `~/.parlai/opt_presets/myfolder/mypreset.opt` allows you to
invoke it via `-o myfolder/mypreset`. These preset files are simple json files
containing a dictionary of files. For example:

```js
{
    "inference": "beam",
    "beam_size": 10,
}
```

## List of presets

The following is a list of all options presets bundled with the latest version
of ParlAI.

```{include} opt_presets_list.inc
```
