# Writing Your Own Script

Author: Kurt Shuster


## Custom ParlAI Script
If you have accustomed yourself with the `parlai` command and would like to write your own ParlAI-based script, this tutorial is meant for you. Writing and using a script today in ParlAI is a smooth experience, thanks to recent work in standardizing scripts across the repository.

### `ParlaiScript`

The [`ParlaiScript`](parlai.core.script.ParlaiScript) class is the abstract class which all ParlAI scripts subclass. A script in ParlAI must define two methods:

#### `setup_args`

The `setup_args` function returns a ParlaiParser with relevant command line args specified. As an example, the [`TrainModel`](https://github.com/facebookresearch/ParlAI/blob/master/parlai/scripts/train_model.py) script adds several arguments required for training a model (number of train epochs, validation statistics, etc.).

#### `run`

This is where you run whatever it is you're attempting to run in your script. In this function you'll have access to `self.opt`, which is an [`Opt`](parlai.core.opt.Opt) dictionary with the appropriate options filled in from `setup_args`.

Suppose we want to write a script that loads a file and prints its length. We may start with the following signature:

```python
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript


class LengthScript(ParlaiScript):

    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(True, True)
        parser.add_argument(
            "--length-filepath",
            type=str,
            help="File to analyze in this script"
        )
        return parser

    def run(self):
        with open(self.opt["length_filepath"]) as f:
            file_content = f.read()
        print(f"Your file has {len(file_content)} characters!")
        return len(file_content)

```

### Registering a Script

After writing your script, you can "register" your script to be picked up by the `parlai` global command. This can be achieved via importing the `register_script` function, and wrapping your script with the decorator, e.g. in the following way:

```python
from parlai.core.script import ParlaiScript, register_script


@register_script("length_script", aliases=["length"])
class LengthScript(ParlaiScript):
    ...

```

### Running a script

Now that you've written and registered your script, it's time to run it!

There are three ways you can now run your script with appropriate options:

#### Command Line

Now that you've registered your script, you can run it on the command line:

```bash
$ parlai length_script --length-filepath file_to_measure.txt
```

#### Import and Run with Args

You can also import your script into another file and run it via its `.main` function:

```python
from my_script.module import LengthScript

if __name__ == "__main__":
    LengthScript.main(['--length-filepath', 'file_to_measure.txt'])
```

#### Import and Run with Kwargs

Finally, rather than specifying args as if from the command line, you can run scripts via passing in keyword arguments:

```python
from my_script.module import LengthScript

if __name__ == "__main__":
    LengthScript.main(length_filepath='file_to_measure.txt')
```