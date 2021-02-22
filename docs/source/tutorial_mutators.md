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

## Usage

The `--mutators` argument should be available for every script where a `--task`
argument is available. Simply begin adding it to use a mutated dataset.

For example, one of the simplest mutators is `flatten`, which just flattens
the conversation.

