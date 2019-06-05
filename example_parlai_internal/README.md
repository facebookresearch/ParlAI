# Internal subfolder

You can set up your own custom agents and tasks, and even create a
separate git repository here to manage your models without the risk of accidentally pushing them to the public ParlAI repo.

Start by creating a new folder named parlai_internal and copying the contents of this folder.
 (We've added this to .gitignore already.)

```bash
cd ~/ParlAI
mkdir parlai_internal
cp -r example_parlai_internal/* parlai_internal
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

Now you can do `from parlai_internal.X.Y import Z` to use your custom modules.
