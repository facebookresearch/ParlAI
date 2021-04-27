# Mutators

List of ParlAI mutators. Mutators are teacher-independent transformations, and
are useful for writing transformations you want to apply to multiple datasets.

Mutators are available any time teachers are used, i.e. when there is a `--task`
argument. Mutators may also be stacked, e.g. `--mutators flatten,word_shuffle`.

Below, we list all the currently supported mutators and give an example of
their output, as well as their options when available.


```{include} mutators_list.inc
```
