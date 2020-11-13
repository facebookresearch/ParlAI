# Tasks and Datasets in ParlAI

__Authors__: Alexander Holden Miller, Filipe de Avila Belbute Peres, Jason Weston, Emily Dinan

ParlAI can support fixed dialogue data for supervised learning (which we
call a dataset) or even dynamic tasks involving an environment, agents
and possibly rewards (we refer to the general case as a task).

In this tutorial we will go over the different ways a new task (or
dataset) can be created.

All setups are handled in pretty much the same way, with the same API,
but there are less steps of course to make a basic dataset.

For a fast way to add a new dataset, go to the **Quickstart** below.

For more complete instructions, or a more complicated setup (like streaming large data), go to the section **Creating a new task: _the more complete way_**.


## Quickstart: Adding a new dataset

Let's look at the easiest way of getting a new dataset into ParlAI
first.

If you have a dialogue, QA or other text-only dataset that you can put
in a text file in the format (called **ParlAI Dialog Format**) we will now describe, you can just load it
directly from there, with no extra code!

Here's an example dataset with a single episode with 2 examples:

    text:hello how are you today?   labels:i'm great thanks! what are you doing?
    text:i've just been biking. labels:oh nice, i haven't got on a bike in years!   episode_done:True

Suppose that data is in the file /tmp/data.txt

:::{note} File format
There are tabs between each field above which are rendered in the
browser as four spaces. Be sure to change them to tabs for the command
below to work.
:::

We could look at that data using the usual display data script:

    python parlai/scripts/display_data.py -t fromfile:parlaiformat --fromfile_datapath /tmp/data.txt
    <.. snip ..>
    [creating task(s): fromfile:parlaiformat]
    [loading parlAI text data:/tmp/data.txt]
    [/tmp/data.txt]: hello how are you today?
    [labels: i'm great thanks! what are you doing?]
       [RepeatLabelAgent]: i'm great thanks! what are you doing?
    ~~
    [/tmp/data.txt]: i've just been biking.
    [labels: oh nice, i haven't got on a bike in years!]
       [RepeatLabelAgent]: oh nice, i haven't got on a bike in years!
    - - - - - - - - - - - - - - - - - - - - -
    ~~
    EPOCH DONE
    [ loaded 1 episodes with a total of 2 examples ]

The text file data format is called ParlAI Dialog format, and is
described in the [teachers documentation](core/teachers); and
in [pyparlai.core.teachers.ParlAIDialogTeacher](https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/teachers.py#L1268).
Essentially, there is one
training example every line, and each field in a ParlAI message is tab
separated with the name of the field, followed by a colon. E.g. the
usual fields like 'text', 'labels', 'label\_candidates' etc. can all be
used, or you can add your own fields too if you have a special use for
them.

:::{danger} Data folds
Data folds are _not_ automatically generated. Using fromfile as above will
result in the same data used for train, validation and test. See the next
section on how to have separate folds.
:::

### Handling Separate Train/Valid/Test data

Once you've gotten the basics of a data working above, you might want to
separate out the data into specific train/valid/test sets, as the above
example uses _the same data for all folds_. This is also easy to do.
Simply separate the data into three separate files: `mydata_train.txt`,
`mydata_valid.txt` and `mydata_test.txt`. Afterwards, modify your parlai
call as follows:

    python parlai/scripts/display_data.py -t fromfile:parlaiformat --fromfile-datapath /tmp/mydata --fromfile-datatype-extension true

This will cause the system to add the `_train.txt`, `_valid.txt`, and
`_test.txt` suffixes at the appropriate times during training,
evaluation, etc.


### Json file format (instead of text file format)

Prefer to use json instead of text files? No problem, the setup is almost the same!
Make a file like this instead (using the same example data as above):

    { "dialog": [ [  {"id": "partner1", "text": "hello how are you today?"},  {"id": "partner2", "text": "i'm great thanks! what are you doing?"},  {"id": "partner1", "text": "i've just been bikinig."},        {"id": "partner2", "text": "oh nice, i haven't got on a bike in years!"} ] ]}

We can then again look at that data using the usual display data script, using the jsonfile teacher:

    python parlai/scripts/display_data.py -t jsonfile --json-datapath /tmp/data.json
    <.. snip ..>
    [creating task(s): jsonfile]
    [loading data from json file into task:/tmp/data.json]
    - - - NEW EPISODE: tmp/data.json - - -
    hello how are you today?
       i'm great thanks! what are you doing?
    i've just been biking.
       oh nice, i haven't got on a bike in years!
    [epoch done]
    [loaded 1 episodes with a total of 2 examples]

The file format consists of one dialogue episode per line, and closely follows the ParlAI messages format. See [here](https://github.com/facebookresearch/ParlAI/tree/master/parlai/utils/conversations.py#L167) for more documentation.

For train/valid/test splits, you can do the same as for text files, using the analogous --jsonfile-datatype-extension true flag.


## Creating a new task: _the more complete way_

Of course after your first hacking around you may want to actually check
this code in so that you can share it with others!

Tasks code is located in the `parlai/tasks` directory.

You will need to create a directory for your new task there.

If your data is in the ParlAI format, you effectively only need a tiny
bit of boilerplate to load it, see e.g. the code for the
[fromfile task agent we just used](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/fromfile).

But right now, let's go through all the steps. You will need to:

0.  Add an `__init__.py` file to make sure imports work correctly.
1.  Implement `build.py` to download and build any needed data (see
    [Part 1: Building the Data](part1)).
2.  Implement `agents.py`, with at least a `DefaultTeacher` which
    extends `Teacher` or one of its children (see [Part 2: Creating the
    Teacher](part2)).
3.  Add the task to the the task list (see [Part 3: Add Task to Task List](part3)).

Below we go into more details for each of these steps.

(part1)=

### Part 1: Building the Data

:::{note} Loading data locally from disk
If you do not intend to commit your task to ParlAI, and instead wish to load your data locally from disk for your own purposes, you can skip this section and go straight to Part 2.
:::

We first need to create functionality for downloading and setting up the
dataset that is going to be used for the task. This is done in the
`build.py` file. Useful functionality for setting up data can be found
in `parlai.core.build_data`.

```python
import parlai.core.build_data as build_data
import os
```

Now we define our build method, which takes in the argument `opt`, which
contains parsed arguments from the command line (or their default),
including the path to the data directory. We can also define a version
string, so that the data is removed and updated automatically for other
ParlAI users if we make changes to this task (here it was just left it
as `None`). We then use the build\_data utilities to check if this data
hasn't been previously built or if the version is outdated. If not, we
proceed to creating the directory for the data, and then downloading and
uncompressing it. Finally, we mark the build as done, so that
`build_data.built()` returns true from now on. Below is an example of
setting up the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset.
```python
RESOURCES = [
    DownloadableFile(
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
        'train-v1.1.json',
        '<checksum for this file>',
        zipped=False,
    ),
    DownloadableFile(
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
        'dev-v1.1.json',
        '<checksum for this file>',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'SQuAD')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES[:2]:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
```

(part2)=

### Part 2: Creating the Teacher

Now that we have our data, we need an agent that understand the task's
structure and is able to present it. In other words, we need a
`Teacher`. Every task requires an `agents.py` file in which we define
the agents for the task. It is there that we will define our teacher.

#### Which base teacher should I use?

We will describe three possible teachers to subclass; you can choose one of them based on your needs:
1. `ParlAIDialogTeacher`: This is the simplest method available, and expects to load a text file of data in **ParlAI Dialog format** (described above). More details are shown in the section [ParlAIDialogTeacher](parlaidialogteacher).
2. `DialogTeacher`: If the data is not in **ParlAI Dialog format**, one can still use the `DialogTeacher` which automates much of the work in setting up a dialog task, but gives the user more flexibility in loading the data from the disk. This is shown in the section [DialogTeacher](dialogteacher).
3. `ChunkTeacher`: Use this teacher if you have a large dataset that cannot fit in memory at once. In the [ChunkTeacher](chunkteacher) section, we show how to break the dataset into smaller "chunks" that are efficiently loaded.


Finally, if the requirements for the task do not fit any of the above,
one can still write a task from scratch without much trouble. This is
shown in the section [Task from Scratch](fromscratch). For example, a dynamic task
which adjusts its response based on the received input rather than using
fixed logs is better suited to this approach.

(parlaidialogteacher)=
#### ParlAIDialogTeacher

For this class, the user must implement at least an `__init__()`
function, and often only that.

In this section we will illustrate the process of using the
`ParlAIDialogTeacher` class by adding the Twitter dataset. This task has
data in textual form and has been formatted to follow the ParlAI Dialog
format. It is thus very simple to implement it using
`ParlAIDialogTeacher`. More information on this class and the dialog
format can be found in the teachers documentation &lt;core/teachers&gt;.

In this task, the agent is presented with questions about movies that
are answerable from Wikipedia. A sample dialog is demonstrated below.

    [twitter]: burton is a fave of mine,even his average films are better than most directors.
    [labels: keeping my fingers crossed that he still has another ed wood in him before he retires.]
    - - - - - - - - - - - - - - - - - - - - -
    ~~
    [twitter]: i saw someone say that we should use glass straws..
    [labels: glass or paper straws - preferably no 'straw' waste. ban !]

Every task requires a `DefaultTeacher`. Since we are subclassing
`ParlAIDialogTeacher`, we only have to initialize the class and set a
few option parameters, as shown below.

```python
class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)

        # get datafile
        opt['parlaidialogteacher_datafile'] = _path(opt, '')

        super().__init__(opt, shared)
```

We can notice there was a call to a `_path()` method, which returns the
path to the correct datafile. The path to the file is then stored in the
options dictionary under the `parlaidialogteacher_datafile` key. This
item is passed to `setup_data()` so that subclasses can just override
the path instead of the function. We still need to implement this
`_path()` method. The version for this example is presented below. It
first ensures the data is built by calling the `build()` method
described in Part 1. It then sets up the paths for the built data.

:::{note} Loading data locally from disk
Note again, that if you are loading data locally from disk, you can skip the call to `build` here, and instead simply return the path to your data file locally given `opt['datatype']`.
:::

```python
from .build import build

def _path(opt, filtered):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'Twitter', dt + '.txt')
```

And this is all that needs to be done to create a teacher for our task
using `ParlAIDialogTeacher`.

To access this data, we can now use the `display_data.py` file in the
`examples` directory:

```bash
parlai display_data -t twitter
```

(dialogteacher)=
#### DialogTeacher

For this class, the user must also implement their own `setup_data()`
function, but the rest of the work of supporting hogwild or batching,
streaming data from disk, processing images, and more is taken care of
for them.

In this section we will demonstrate the process of using the
`DialogTeacher` class by adding the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) dataset. The data on disk downloaded
from the [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not fit the basic `ParlAIDialogTeacher` format described above. Still, using
`DialogTeacher` makes it easy to implement dialog tasks such as this
one.

In this task, the agent is presented with a paragraph from Wikipedia and asked to answer a question about it.

```
[id]: squad
[text]: In October 2014, Beyoncé signed a deal to launch an activewear line of clothing with British fashion retailer Topshop. The 50-50 venture is called Parkwood Topshop Athletic Ltd and is scheduled to launch its first dance, fitness and sports ranges in autumn 2015. The line will launch in April 2016.
When will the full line appear?

[labels]: April 2016
```

We will call our teacher `SquadTeacher`. Let's initialize this class
first.

```python
class SquadTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        # whatever is placed into datafile will be passed as the argument to
        # setup_data in the next section.
        opt['datafile'] = os.path.join(opt['datapath'], 'SQuAD', suffix + '-v1.1.json')
        self.id = 'squad'
        super().__init__(opt, shared)
```

The `id` field names the teacher in the dialog.

By creating `SquadTeacher` as a subclass of `DialogTeacher`, the job
of creating a teacher for this task becomes much simpler: most of the
work that needs to be done will limit itself to defining a `setup_data`
method. This method is a generator that will take in a path to the data
and yield a pair of elements for each call. The first element of the
pair is a dictionary containing a dialogue act (with required fields
`text` and `labels` and any other additional field required by your task,
such as `label_candidates`). In this case,
we *only* return `text` and `labels`. The second
is a boolean flag `new_episode?` which indicates if the current query
starts a new episode or not.

More information on this format can be found in the documentation under
`DialogData` in the teachers documentation &lt;core/teachers&gt;
(`setup_data` is provided as a data\_loader to `DialogData`).

The sample `setup_data` method for our task is presented below.

```python
def setup_data(self, path):
    # note that path is the value provided by opt['datafile']
    print('loading: ' + path)
    with PathManager.open(path) as data_file:
        self.squad = json.load(data_file)['data']
    for article in self.squad:
        # each paragraph is a context for the attached questions
        for paragraph in article['paragraphs']:
            # each question is an example
            for qa in paragraph['qas']:
                question = qa['question']
                answers = tuple(a['text'] for a in qa['answers'])
                context = paragraph['context']
                yield {"text": content + "\n" + question, "labels": answers}, True
```

As we can see from the code above, for this task, each
episode consists of only one query, thus `new_episode?` is always true
(i.e., each query is the start of its episode). This could also vary
depending on the task.

Finally, one might notice that no reward, label candidates, or a path
to an image were
provided in the tuple (all are set to `None`). These fields are not relevant to this task.

The only thing left to be done for this part is to define a
`DefaultTeacher` class. This is a requirement for any task, as the
`create_agent` method looks for a teacher named this. We can simply
default to the class we have built so far.

```python
class DefaultTeacher(SquadTeacher):
    pass
```

And we have finished building our task.

(chunkteacher)=
#### Chunk Teacher

Chunk Teacher is useful for streaming large amounts of data
(*read: does not fit into memory*), that naturally separate
into several separate files (or chunks). The data is separated into chunks and
loaded one chunk at a time. Loads the data off of the main thread.

To implement a chunk teacher, you have to write the following functions:
- `_get_data_folder`: Returns the path to the directory containing the chunks.
- `get_num_samples`: Given `opt`, returns a tuple containing `(num_episodes, num_examples)`. Since we are streaming this data, we must know the number of examples a priori.
- `get_fold_chunks`: Given `opt`, returns a list of chunks indices. For example, we might
separate the chunks based on the data split, given by `opt['datatype']`.
- `load_from_chunk`: Given a chunk index, loads the associated file and returns a list of samples.
- `create_message`: Given a single sample item from the list returned by `load_from_chunk`, create a Message to return.

We create an example teacher to demonstrate. Let's suppose that
`/tmp/path_to_my_chunks/` is the directory containing our chunks, and
each chunk file (e.g. `/tmp/path_to_my_chunks/1.txt`) has the following format:

```
<input 1>\t<output 1>
<input 2>\t<output 2>
<input 3>\t<output 3>
...
<input 100>\t<output 100>
```


Then our chunk teacher would look like the following:
```python
class ExampleChunkTeacher(ChunkTeacher):
    def _get_data_folder(self):
        # return the path to directory containing your chunks
        return '/tmp/path_to_my_chunks/'

    def get_num_samples(self, opt) -> Tuple[int, int]:
        # return the number of episodes and examples
        # in this case, all of our episodes are single examples
        # so they are the same number
        datatype = opt['datatype']
        if 'train' in datatype:
            return 300, 300  # each chunk contains 100 examples
        elif 'valid' in datatype:
            return 100, 100
        elif 'test' in datatype:
            return 100, 100

    def get_fold_chunks(self, opt) -> List[int]:
        # in this case, our train split contains 3 chunks and
        # valid and test each contain 1
        datatype = opt['datatype']
        if 'train' in datatype:
            return [1, 2, 3]
        elif 'valid' in datatype:
            return [4]
        elif 'test' in datatype:
            return [5]

    def load_from_chunk(self, chunk_idx: int):
        # we load the chunk specified by chunk_idx and return a
        # list of outputs
        chunk_path = os.path.join(self._get_data_folder(), f'{chunk_idx}.txt')
        output = []
        with open(chunk_path, 'r') as f:
            for line in f.readlines():
                txt_input, txt_output = line.split('\t')
                output.append((txt_input, txt_output))

        return output

    def create_message(self, sample_item, entry_idx=0):
        # finally, we return a message given an element from the list
        # returned by `load_from_chunk`
        text, label = sample_item
        return {'id': 'Example Chunk Teacher', 'text': text, 'labels': [label], 'episode_done': True}
```


:::{note} Streaming data
Chunk Teacher only works with streaming data, so make sure to run with
`-dt train:stream` (or `-dt valid:stream` or `-dt test:stream`) when using
your chunk data.
:::


(fromscratch)=
#### Task from Scratch

In this case, one would inherit from the Teacher class. For this class,
at least the `act()` method and probably the `observe()` method must be
overriden. Quite a bit of functinoality will not be built in, such as a
support for hogwild and batching, but metrics will be set up and can be
used to track stats like the number of correctly answered examples.

In general, extending Teacher directly is not recommended unless the
above classes definitely do not fit your task. We still have a few
remnants which do this in our code base instead of using
FixedDialogTeacher, but this will require one to do extra work to
support batching and hogwild if desired.

However, extending teacher directly is necessary for non-fixed data. For
example, one might have a like the full negotiation version of the
`dealnodeal` task, where episodes are variable-length (it continues
until one player ends the discussion).

In this case, just implement the `observe()` function to handle seeing
the text from the other agent, and the `act()` function to take an
action (such as sending text or other fields such as reward to the other
agent).

(part3)=
### Part 3: Add Task to Task List

Now that our task is complete, we must add an entry to the
`task_list.py` file in `parlai/tasks`. This file just contains a
json-formatted list of all tasks, with each task represented as a
dictionary. Sample entries for our tasks are presented below.

```python
[
    # other tasks...
    {
        "id": "MTurkWikiMovies",
        "display_name": "MTurk WikiMovies",
        "task": "mturkwikimovies",
        "tags": [ "all",  "QA" ],
        "description": "Closed-domain QA dataset asking MTurk-derived questions about movies, answerable from Wikipedia. From Li et al. '16. Link: https://arxiv.org/abs/1611.09823"
    },
    {
        "id": "MNIST_QA",
        "display_name": "MNIST_QA",
        "task": "mnist_qa",
        "tags": [ "all", "Visual" ],
        "description": "Task which requires agents to identify which number they are seeing. From the MNIST dataset."
    },
    {
        "id": "VQAv2",
        "display_name": "VQAv2",
        "task": "vqa_v2",
        "tags": [ "all", "Visual" ],
        "description": "Bigger, more balanced version of the original VQA dataset. From Goyal et al. '16. Link: https://arxiv.org/abs/1612.00837"
    },
    # other tasks...
]
```

(part4)=

### Part 4: Executing the Task

A simple way of testing the basic functionality in a task is to run the
`display_data.py` example in the `examples` directory. Now that the work
is done, we can pass it to ParlAI by using the `-t` flag. For example,
to execute the MTurk WikiMovies task we should call:

`python display_data.py -t mturkwikimovies`

To run the MNIST\_QA task, while displaying the images in ascii format,
we could call:

`python display_data.py -t mnist_qa -im ascii`

And for VQAv2:

`python display_data.py -t vqa_v2`
