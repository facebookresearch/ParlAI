# Mutators

**Author**: Stephen Roller

Mutators are task-independent data transformations, which are applicable to
_any_ dataset. Examples of mutators include:

- Reversing all the turns in a conversation
- Down-sampling the dataset
- Shuffling the words in a turn
- and much more

Mutators are particularly useful when you want to train or test on different
variants of the data. For example,
[Sundkar et al. (2019)](https://arxiv.org/abs/1906.01603) showed that different
models react differently to having their turns or words shuffled.

For a full list of Mutators existing in ParlAI, check our [mutators][Mutators
Reference].

:::{warning} New feature
Mutators is a brand new feature in ParlAI. If you experience any issues with it,
please [file an issue](https://github.com/facebookresearch/ParlAI/issues/new?assignees=&labels=&template=other.md)
on GitHub.
:::


## Usage

The `--mutators` argument should be available for every script where a `--task`
argument is available. Simply begin adding it to use a mutated dataset.

For example, one of the simplest mutators is `flatten`, which just flattens
the conversation.

```bash
parlai display_data --task dailydialog --mutators flatten
```

### Composability

Mutators are intentionally designed to be composable. That is, we can stack
mututators on top of each other by specifying multiple on the command line:

```bash
parlai display_data -t dailydialog --mutators word_shuffle+flatten
parlai display_data -t dailydialog --mutators word_shuffle,flatten  # equivalent
```

This runs the `word_shuffle` mutator, and pipes the output to the `flatten`
mutator


### Multi-task mutators

Mutators default to being applied on every task. For example, this applies the
same mutators to both tasks (independently):

```bash
parlai display_data -t dailydialog,convai2 --mutators word_shuffle+flatten
```

If necessary, you may also supply mutators to specific tasks. Note that in this
case, you can only use the `+` joiner, and `,` is unavailable.

```bash
parlai display_data -t dailydialog:mutators=word_shuffle,convai2:mutators=flatten+word_shuffle
```


### Mutator arguments

Some mutators have additional arguments. For example, `episode_shuffle` has an
argument `preserve_context`.

```bash
parlai display_data -t dailydialog --mutators episode_shuffle --preserve_context True
```

Unfortunately, mutator arguments cannot be directly specified when using the `--task X:mutators=` format.  Instead, we can pass mutator arguments through the task argument.

```bash
parlai display_data -t dailydialog:mutators=episode_shuffle:preserve_context=True
```


## Writing your own Mutators

Mutators are meant to be added too. Following other patterns in ParlAI, you can
add your own mutators by making sure you decorate your Mutator class with
`@register_mutator("example_name")` before the script runs if you're
using ParlAI in an IPython notebook; if you've checked out ParlAI code locally
to make your own modifications, you can add a new file in
[`parlai/mutators`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/mutators).

ParlAI has 3 base classes for Mutators. Choosing the right base class is only about
making bookkeeping easier.

- `MessageMutator` is used when you need to make changes to individual turns, and
  is no relationship between turns.
- `EpisodeMutator` is when you want to make changes to whole conversations
  (episodes), but you want to keep the number of episodes fixed.
- `ManyEpisodeMutator` is the most powerful setting, and lets you map each episode
  to 0 or more episodes. It is also slightly more complex.

:::{warning} Sharing
Unlike you may expect from [other parts of ParlAI](tutorial_worlds), Mutators do not have
any sort of sharing mechanism. There is always exactly one instance of each
mutator specified on the command line.
:::

For information on writing Mutators, please see the [API
Reference](core/mutators). As additional resources, we provide the following
examples:

- [Word
  Shuffle](https://github.com/facebookresearch/ParlAI/tree/main/parlai/mutators/word_shuffle.py):
  shows how to implement a simple `MessageMutator`.
- [Episode
  Reverse](https://github.com/facebookresearch/ParlAI/tree/main/parlai/mutators/episode_reverse.py):
  shows how to implement a simple `EpisodeMutator`.
- [Flatten](https://github.com/facebookresearch/ParlAI/tree/main/parlai/mutators/flatten.py):
  shows how to implement a `ManyEpisodeMutator`.
