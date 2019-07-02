# Internal subfolder

By creating an internal subfolder, you can set up your own custom agents and tasks,
create your own model zoo, and manage it all with a separate git repository.
We've set it up so you can now work inside the ParlAI folder without the risk of accidentally pushing your work to the public ParlAI repo.
We've also added some convenient shortcuts to mirror what we provide in the `parlai` folder.

## How to

Start by creating a new folder named parlai_internal and copying the contents of this folder.
 (We've added this to .gitignore already.)

```bash
cd ~/ParlAI
mkdir parlai_internal
cp -r example_parlai_internal/ parlai_internal
cd parlai_internal
```

We've ignored this folder, but that's it. If you want to set this up as
a separate git repository (e.g. for version control) you can follow the standard steps
for creating a new repo (feel free to do this however you prefer).

```bash
git init
git add .
git commit -m "Initialize parlai_internal"
```

You can connect this to a new github repository if desired.
[Create a new repo](https://github.com/new) (you don't need to initialize
with a README), and then follow the instructions to push
an existing repository from command line.


## Some features

We also provide a number of shortcuts which mirror the public repo.

You can do `from parlai_internal.X.Y import Z` to use your custom modules.

Additionally, you can invoke your internal model agents from command line with `-m internal:model`.
Providing this argument will cause the parser to look for `parlai_internal.agents.model.model.ModelAgent`.
As an example, we provide `parlai_internal/agents/parrot/parrot.py`. You could call
(from the top-level ParlAI folder):

```bash
python examples/display_model.py -t babi:task10k:1 -m internal:parrot
```

Similarly, you can add private tasks under a tasks folder here and invoke them with `-t internal:taskname`.
The parser will look for `parlai_internal.tasks.taskname.taskname.DefaultTeacher`.

You can even create your own model zoo of pretrained models. `parlai_internal/zoo/.internal_zoo_path`
needs to be modified to contain the path to the folder containing all of your models. Once
you've done that, you can use those models by simply adding `-mf /rest/of/modelfilepath`.
For example, if you change `.internal_zoo_path` to be `/private/home/user/checkpoints`
and you have a model at `/private/home/user/checkpoints/model_file/model`, you could use `-mf izoo:model_file/model`.

And you can use as many of these in combination as you would like. For instance, to evaluate a model file that uses
an internal agent definition on an internal task, you would do:

```bash
python examples/eval_model.py -t internal:taskname -m internal:model -mf izoo:model_file/model
```
