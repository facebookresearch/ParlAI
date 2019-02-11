..
  Copyright (c) Facebook, Inc. and its affiliates.
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.

Tasks and Datasets in ParlAI
============================
**Authors**: Alexander Holden Miller, Filipe de Avila Belbute Peres, Jason Weston

ParlAI can support fixed dialogue data for supervised learning (which we call a dataset) or even dynamic tasks involving an environment, agents and possibly rewards (we refer to the general case  as a task).

In this tutorial we will go over the different ways a new task (or dataset) can be created.

All setups are handled in pretty much the same way, with the same API, but there are less steps of course to make a basic dataset.


Getting a New Dataset Into ParlAI: *the simplest way*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's look at the easiest way of getting a new dataset into ParlAI first.

If you have a dialogue, QA or other text-only dataset that you can put
in a text file in the format we will now describe, you can just load it directly from
there, with no extra code!

Here's an example dataset with a single episode with 2 examples:

::

	text:hello how are you today?	labels:i'm great thanks! what are you doing?
	text:i've just been biking.	labels:oh nice, i haven't got on a bike in years!	episode_done:True

Suppose that data is in the file /tmp/data.txt

.. note::

	There are tabs between each field above which are rendered in the browser as four spaces.
	Be sure to change them to tabs for the command below to work.

We could look at that data using the usual display data script:

::

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

The text file data format is called ParlAI Dialog format, and is described
in the :doc:`teachers documentation <teachers>` (parlai.core.teachers.ParlAIDialogTeacher)
and
in the `core/teachers.py file <https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/teachers.py#L1098>`_.
Essentially, there is one training example every line, and each field in a
ParlAI message is tab separated with the name of the field, followed by a colon.
E.g. the usual fields like 'text', 'labels', 'label_candidates' etc. can all
be used, or you can add your own fields too if you have a special use for them.


Creating a New Task: *the more complete way*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Of course after your first hacking around you may want to actually check this code in so that you can share it with others!

Tasks code is located in the ``parlai/tasks`` directory.

You will need to create a directory for your new task there.

If your data is in the ParlAI format, you effectively only need a tiny bit of boilerplate to load it, see e.g. the code for the `fromfile task agent we just used <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/fromfile>`_.

But right now, let's go through all the steps. You will need to:

0. Add an ``__init__.py`` file to make sure imports work correctly.
1. Implement ``build.py`` to download and build any needed data (see `Part 1: Building the Data`_).
2. Implement ``agents.py``, with at least a ``DefaultTeacher`` which extends ``Teacher`` or one of its children (see `Part 2: Creating the Teacher`_).
3. Add the task to the the task list (see `Part 3: Add Task to Task List`_).

Below we go into more details for each of these steps.


Part 1: Building the Data
^^^^^^^^^^^^^^^^^^^^^^^^^

We first need to create functionality for downloading and setting up the dataset
that is going to be used for the task. This is done in the ``build.py`` file.
Useful functionality for setting up data can be found in ``parlai.core.build_data``.

.. code-block:: python

    import parlai.core.build_data as build_data
    import os

Now we define our build method, which takes in the argument ``opt``,
which contains parsed arguments from the command line (or their default),
including the path to the data directory. We can also define a version string,
so that the data is removed and updated automatically for other ParlAI users
if we make changes to this task (here it was just left it as ``None``).
We then use the build_data utilities to check if this data hasn't been
previously built or if the version is outdated. If not, we proceed to creating
the directory for the data, and then downloading and uncompressing it.
Finally, we mark the build as done, so that ``build_data.built()`` returns
true from now on. Below is an example of setting up the MNIST dataset.

.. code-block:: python

    def build(opt):
        # get path to data directory
        dpath = os.path.join(opt['datapath'], 'mnist')
        # define version if any
        version = None

        # check if data had been previously built
        if not build_data.built(dpath, version_string=version):
            print('[building data: ' + dpath + ']')

            # make a clean directory if needed
            if build_data.built(dpath):
                # an older version exists, so remove these outdated files.
                build_data.remove_dir(dpath)
            build_data.make_dir(dpath)

            # download the data.
            fname = 'mnist.tar.gz'
            url = 'http://parl.ai/downloads/mnist/' + fname # dataset URL
            build_data.download(url, dpath, fname)

            # uncompress it
            build_data.untar(dpath, fname)

            # mark the data as built
            build_data.mark_done(dpath, version_string=version)



Part 2: Creating the Teacher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we have our data, we need an agent that understand the task's structure
and is able to present it. In other words, we need a ``Teacher``.
Every task requires an ``agents.py`` file in which we define the agents for the task.
It is there that we will define our teacher.

Teachers already in the ParlAI system use a series of subclasses, each with
additional functionality (and fewer methods to implement). These follow the path
``Agent`` => ``Teacher`` => ``FixedDialogTeacher`` => ``DialogTeacher`` => ``ParlAIDialogTeacher``.

(Note there is also a FbDialogTeacher, but this is deprecated -- although some datasets in ParlAI still currently use it.)

The simplest method available for creating a teacher is to use the
``ParlAIDialogTeacher`` class, which makes the process very simple if the text
data is already formatted in the ParlAI Dialog format.
(In fact, even if your text data is not in the ParlAI Dialog format, it might
be simpler to parse it into this format and use the ``ParlAIDialogTeacher``.)
This is shown in the section `ParlAIDialogTeacher`_.

If the data is not in this format, one can still use the ``DialogTeacher``
which automates much of the work in setting up a dialog task,
but gives the user more flexibility in loading the data from the disk.
This is shown in the section `DialogTeacher`_.

If the data is still a fixed set (e.g. is not dynamic, is based on fixed files)
and even more functionality is needed, such as providing extra information
like the answer indices for the SQuAD dataset, one can use the
``FixedDialogTeacher`` class. This is shown in the section `FixedDialogTeacher`_.

Finally, if the requirements for the task do not fit any of the above,
one can still write a task from scratch without much trouble.
This is shown in the section `Task from Scratch`_. For example, a dynamic task
which adjusts its response based on the received input rather than using fixed
logs is better suited to this approach.

The methods for a teacher to implement for each class are as follows:

:class Teacher: ``__init__()``, ``observe()``, ``act()``

:class FixedDialogTeacher: ``__init__()``, ``get()``, ``num_examples()``, ``num_episodes()``

:class DialogTeacher: ``__init__()``, ``setup_data()``

:class ParlAIDialogTeacher: ``__init__()``



ParlAIDialogTeacher
~~~~~~~~~~~~~~~~~~~

For this class, the user must implement at least an ``__init__()`` function, and
often only that.

In this section we will illustrate the process of using the ``ParlAIDialogTeacher``
class by adding the Twitter dataset.
This task has data in textual form and has been formatted to follow the ParlAI Dialog format.
It is thus very simple to implement it using ``ParlAIDialogTeacher``.
More information on this class and the dialog format can be found in the :doc:`teachers documentation <teachers>`.

In this task, the agent is presented with questions about movies that are answerable from Wikipedia.
A sample dialog is demonstrated below.

::

	[twitter]: burton is a fave of mine,even his average films are better than most directors.
	[labels: keeping my fingers crossed that he still has another ed wood in him before he retires.]
	- - - - - - - - - - - - - - - - - - - - -
	~~
	[twitter]: i saw someone say that we should use glass straws..
	[labels: glass or paper straws - preferably no 'straw' waste. ban !]

Every task requires a ``DefaultTeacher``. Since we are subclassing ``ParlAIDialogTeacher``,
we only have to initialize the class and set a few option parameters, as shown below.

.. code-block:: python

    class DefaultTeacher(ParlAIDialogTeacher):
        def __init__(self, opt, shared=None):
            super().__init__(opt, shared)
            opt = copy.deepcopy(opt)

            # get datafile
            opt['datafile'] = _path(opt, '')

We can notice there was a call to a ``_path()`` method, which returns the path to the correct datafile.
The path to the file is then stored in the options dictionary under the ``datafile`` key.
This item is passed to ``setup_data()`` so that subclasses can just override the path instead of the function.
We still need to implement this ``_path()`` method. The version for this example is presented below.
It first ensures the data is built by calling the ``build()`` method described in Part 1.
It then sets up the paths for the built data.

.. code-block:: python

    from .build import build

    def _path(opt, filtered):
        # build the data if it does not exist
        build(opt)

        # set up path to data (specific to each dataset)
        dt = opt['datatype'].split(':')[0]
        return os.path.join(opt['datapath'], 'Twitter', dt + '.txt')

And this is all that needs to be done to create a teacher for our task using ``ParlAIDialogTeacher``.

To access this data, we can now use the ``display_data.py`` file in the ``examples`` directory:

.. code-block:: bash

    python examples/display_data.py -t twitter


DialogTeacher
~~~~~~~~~~~~~

For this class, the user must also implement their own ``setup_data()`` function,
but the rest of the work of supporting hogwild or batching, streaming data from
disk, processing images, and more is taken care of for them.

In this section we will demonstrate the process of using the ``DialogTeacher``
class by adding a simple question-answering task based on the MNIST dataset.
This task depends on visual data and so does not fit the basic ``ParlAIDialogTeacher``
class described above. Still, using ``DialogTeacher`` makes it easy to
implement dialog tasks such as this one.

In this task, the agent is presented with the image of a digit and then asked to
answer which number it is seeing. A sample episode is demonstrated below.
Note that we display an ASCII rendition here for human-viewing,
and while you could try to train a model on the ASCII,
the pixel values and several preprocessing options are available instead.

::

    [mnist_qa]: Which number is in the image?
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@83 c@@@@@@@@@@
    @@@@@@@@@@@@@h:  ,@@@@@@@@@@
    @@@@@@@@@@@@c    .&@@@@@@@@@
    @@@@@@@@@@@:  .,  :@@@@@@@@@
    @@@@@@@@@@A  c&@2  8@@@@@@@@
    @@@@@@@@@H  ;@@@H  h@@@@@@@@
    @@@@@@@@9: ,&@@G.  #@@@@@@@@
    @@@@@@@@h ,&@@A    @@@@@@@@@
    @@@@@@@@; H@&s    r@@@@@@@@@
    @@@@@@@@: ::.     #@@@@@@@@@
    @@@@@@@@h        ;@@@@@@@@@@
    @@@@@@@@h        G@@@@@@@@@@
    @@@@@@@@@A,:2c  :@@@@@@@@@@@
    @@@@@@@@@@@@@:  3@@@@@@@@@@@
    @@@@@@@@@@@@&, r@@@@@@@@@@@@
    @@@@@@@@@@@@:  A@@@@@@@@@@@@
    @@@@@@@@@@@@   2@@@@@@@@@@@@
    @@@@@@@@@@@@  ,@@@@@@@@@@@@@
    @@@@@@@@@@@@  3@@@@@@@@@@@@@
    @@@@@@@@@@@@ ,&@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@

    [labels: 9|nine]
    [cands: seven|six|one|8|two| ...and 15 more]
       [Agent]: nine


We will call our teacher ``MnistQATeacher``. Let's initialize this class first.

.. code-block:: python

    class MnistQATeacher(DialogTeacher):
        def __init__(self, opt, shared=None):
            # store datatype
            self.datatype = opt['datatype'].split(':')[0]

            # store identifier for the teacher in the dialog
            self.id = 'mnist_qa'

            # strings for the labels in the class (digits)
            # (information specific to this task)
            self.num_strs = ['zero', 'one', 'two', 'three', 'four', 'five',
                    'six', 'seven', 'eight', 'nine']

            # store paths to images and labels
            opt['datafile'], self.image_path = _path(opt)

            super().__init__(opt, shared)

The ``id`` field names the teacher in the dialog. The ``num_strs`` field is
specific to this example task. It is being used simply to store the text
version of the digits.

We also call our ``_path()`` method (defined below). The ``opt['datafile']`` item
is passed to ``setup_data()`` when it is called by DialogTeacher, which we will
also define below.

The version of ``_path()`` for this example is presented below.
It first ensures the data is built by calling the ``build()`` method described above.
It then sets up the paths for the built data. This should be specific to the dataset being used.
If your dataset does not use images, the ``image_path`` is not necessary, for example.
Or if your task will use data other than labels, the path to the file containing this information can also be returned.
You do not need to put this in a separate function like we do here, but could also encode directly in the class.

.. code-block:: python

    def _path(opt):
        # ensure data is built
        build(opt)

        # set up paths to data (specific to each dataset)
        dt = opt['datatype'].split(':')[0]
        labels_path = os.path.join(opt['datapath'], 'mnist', dt, 'labels.json')
        image_path = os.path.join(opt['datapath'], 'mnist', dt)
        return labels_path, image_path

By creating ``MnistQATeacher`` as a subclass of ``DialogTeacher``,
the job of creating a teacher for this task becomes much simpler:
most of the work that needs to be done will limit itself to defining a ``setup_data`` method.
This method is a generator that will take in a path to the data and yield a
pair of elements for each call.
The first element of the pair is a tuple containing the following information:
``(query, labels, reward, label_candidates, path_to_image)``.
The second is a boolean flag ``new_episode?`` which indicates if the current
query starts a new episode or not.

More information on this format can be found in the documentation under ``DialogData``
in the :doc:`teachers documentation <teachers>`
(``setup_data`` is provided as a data_loader to ``DialogData``).

The sample ``setup_data`` method for our task is presented below.

.. code-block:: python

    def setup_data(self, path):
        print('loading: ' + path)

        # open data file with labels
        # (path will be provided to setup_data from opt['datafile'] defined above)
        with open(path) as labels_file:
            self.labels = json.load(labels_file)

        # define standard question, since it doesn't change for this task
        self.question = 'Which number is in the image?'
        # every episode consists of only one query in this task
        new_episode = True

        # define iterator over all queries
        for i in range(len(self.labels)):
            # set up path to curent image
            img_path = os.path.join(self.image_path, '%05d.bmp' % i)
            # get current label, both as a digit and as a text
            label = [self.labels[i], self.num_strs[int(self.labels[i])]]
            # yield tuple with information and new_episode? flag (always True)
            yield (self.question, label, None, None, img_path), new_episode

As we can see from the code above, for this specific task the question is always the same,
and thus it is fixed. For different tasks, this would likely change at each iteration.
Similarly, for this task, each episode consists of only one query, thus
``new_episode?`` is always true (*i.e.*, each query is the start of its episode).
This could also vary depending on the task.

Looking at the tuple provided by the iterator at each yield,
we can see that we defined a query, a label and an image path.
When working with ``DialogTeacher`` in visual tasks, we provide the path to the
image on disk so that the dialog teacher can automatically load and process it.
The "image-mode" command line argument allows for a number of post-processing
options, including returning the raw pixels, extracting features using
pre-trained image models (which are cached and loaded from file the next time)
or as above converted to ASCII.

Finally, one might notice that no reward or label candidates were provided in
the tuple (both are set to ``None``). The reward is not specified because it is
not useful for this supervised-learning task. The label candidates, however,
were not specified per-example for this task because we instead use a single set
of universal candidates for every example in this task (the digits from '0' to '9').
For cases like this, with fixed label candidates, one can simply define a method
``label_candidates()`` that returns the unchanging candidates, as demonstrated below.
For cases where the label candidates vary for each query, the field in the tuple can be used.

.. code-block:: python

    def label_candidates(self):
        return [str(x) for x in range(10)] + self.num_strs

The only thing left to be done for this part is to define a ``DefaultTeacher`` class.
This is a requirement for any task, as the ``create_agent`` method looks for a teacher named this.
We can simply default to the class we have built so far.

.. code-block:: python

    class DefaultTeacher(MnistQATeacher):
        pass

And we have finished building our task.


FixedDialogTeacher
~~~~~~~~~~~~~~~~~~

For this class the user must define at least ``__init__()``, a ``get()`` function,
and ``num_examples()`` and ``num_episodes()``. The user must also handle data
loading and storage on their own, which can be done during intialization.
However, like with its child DialogTeacher, batching and hogwild will still be
handled automatically, as well as metric updating and reporting, example iteration,
and more.

In this section we will demonstrate the use of this class with the VQAv2
visual question-answering task. Since we want to return additional fields apart
from the standard ones used in DialogTeacher (text, labels, reward, candidates,
an image, and whether the episode is done), we'll extend FixedDialogTeacher instead.
We'll also demonstrate the use of the multithreaded loader that is available,
which can be helpful for speeding up image loading by beginning to load the next
example while the current one is being looked at by the model.

In this task, the agent is presented with an image of a scene and then asked
to answer a question about that scene. A sample episode is demonstrated below.

.. image:: _static/img/task_tutorial_skateboard.jpg

::

    [vqa_v2]: What is this man holding?
    [labels: skateboard]
       [Agent]: skateboard


We will call our teacher ``OeTeacher`` (for open-ended teacher, since it doesn't provide the agent with label candidates).
Let's initialize this class first.

.. code-block:: python

    class OeTeacher(FixedDialogTeacher):
        """VQA v2.0 Open-Ended teacher, which loads the json VQA data and
        implements the ``get`` method to return additional metadata.
        """
        def __init__(self, opt, shared=None):
            super().__init__(opt)
            self.image_mode = opt.get('image_mode', 'none')

            if shared and 'ques' in shared:
                # another instance was set up already, just reference its data
                self.ques = shared['ques']
                if 'annotation' in shared:
                    self.annotation = shared['annotation']
                self.image_loader = shared['image_loader']
            else:
                # need to set up data from scratch
                data_path, annotation_path, self.image_path = _path(opt)
                self._setup_data(data_path, annotation_path)
                self.image_loader = ImageLoader(opt)

            self.reset()


There are a few parts to this initialization.
First, we store the image mode so the we know how to preprocess images.
Then, we check if this teacher is being initialized with a ``shared`` parameter.
This is used during hogwild or batching to share data within a batch or between
threads without each instance having to initialize from scratch. See the
**Batching and Hogwild** tutorial for more information on this.
If ``shared`` is empty, then we'll move on to loading our data.

Finally we'll reset the class so parents can initialize class fields to
support threaded loading, metrics, and more.

Let's take a quick look at the fucntions which set up the data and share it
between instances just so we see how those are set up.

.. code-block:: python

    def _setup_data(self, data_path, annotation_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.ques = json.load(data_file)

        if not self.datatype.startswith('test'):
            print('loading: ' + annotation_path)
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)

    def share(self):
        shared = super().share()
        shared['ques'] = self.ques
        if hasattr(self, 'annotation'):
            shared['annotation'] = self.annotation
        shared['image_loader'] = self.image_loader
        return shared


Next up, we need to implement ``num_examples()`` and ``num_episodes`` for the
FixedDialogTeacher teacher to work correctly. These are very straightforward,
and we only have one question per episode, so we can reuse that definition.

.. code-block:: python

    def num_examples(self):
        return len(self.ques['questions'])

    def num_episodes(self):
        return self.num_examples()


Next we need to implement the ``get()`` function. This has two arguments: which
episode we want to pull from, and then the index within that episode of the
specific example we want. Since every episode has only one entry in this dataset,
we provide a default for the keyword and ignore it.

We also define the DefaultTeacher class to refer to this one.
This task also includes another teacher which includes multiple choice candidates,
but we don't include that in this tutorial.

.. code-block:: python

    def get(self, episode_idx, entry_idx=0):
        qa = self.ques['questions'][episode_idx]
        question = qa['question']

        action = {
            'id': self.id,
            'text': question,
            'image_id': qa['image_id'],
            'episode_done': True
        }

        if not self.datatype.startswith('test'):
            # test set annotations are not available for this dataset
            anno = self.annotation['annotations'][episode_idx]
            action['labels'] = [ans['answer'] for ans in anno['answers']]

        return action


    class DefaultTeacher(OeTeacher):
        pass


At this point, the class is done! However, we'll extend it a little further to
take advantage of a few utility methods that allow for loading the next image
in the background by overriding the ``next_example()`` method of FixedDialogTeacher
(the method that calls our ``get()`` method).

.. code-block:: python

    def reset(self):
        super().reset()  # call parent reset so other fields can be set up
        self.example = None  # set up caching fields
        self.next_example()  # call this once to get the cache moving

    def next_example(self):
        """Returns the next example from this dataset after starting to queue
        up the next example.
        """
        ready = None
        # pull up the currently queued example
        if self.example is not None:
            if self.image_mode != 'none':
                # move the image we loaded in the background into the example
                image = self.data_queue.get()
                self.example['image'] = image
            ready = (self.example, self.epochDone)
        # get the next base example: super().next_example() calls self.get()
        self.example, self.epochDone = super().next_example()
        if self.image_mode != 'none' and 'image_id' in self.example:
            # load the next image in the background
            image_id = self.example['image_id']
            self.submit_load_request(image_id)
        # return the previously cached example
        return ready

This method uses the ``submit_load_request()`` method to start a background
thread loading the next image, loading previously finished work with with
``self.data_queue.get()``. It calls ``super().next_example()`` to prepare the
next example it's going to return, which calls the ``get()`` method we defined,
and then returns the previously cached example. Note that here we also call
``next_example()`` in the ``reset()`` function to start filling the cache.

This extra functionality helps in particular with loading images--for this task,
adding the threading helped a model to process an epoch approximately 2.5x faster.
Further speedups can be accomplished with the Pytorch dataloader, adding another
6.5x speedup. A tutorial on how to use this dataloader is forthcoming.


Task from Scratch
~~~~~~~~~~~~~~~~~

In this case, one would inherit from the Teacher class.
For this class, at least the ``act()`` method and probably the ``observe()``
method must be overriden. Quite a bit of functinoality will not be built in,
such as a support for hogwild and batching, but metrics will be set up and
can be used to track stats like the number of correctly answered examples.

In general, extending Teacher directly is not recommended unless the above
classes definitely do not fit your task. We still have a few remnants which
do this in our code base instead of using FixedDialogTeacher, but this will
require one to do extra work to support batching and hogwild if desired.

However, extending teacher directly is necessary for non-fixed data.
For example, one might have a like the full negotiation version of the
``dealnodeal`` task, where episodes are variable-length (it continues until one
player ends the discussion).

In this case, just implement the ``observe()`` function to handle seeing the
text from the other agent, and the ``act()`` function to take an action
(such as sending text or other fields such as reward to the other agent).


Part 3: Add Task to Task List
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that our task is complete, we must add an entry to the ``task_list.py`` file in ``parlai/tasks``.
This file just contains a json-formatted list of all tasks, with each task represented as a dictionary.
Sample entries for our tasks are presented below.

.. code-block:: python

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

Part 4: Executing the Task
^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple way of testing the basic functionality in a task is to run the
``display_data.py`` example in the ``examples`` directory.
Now that the work is done, we can pass it to ParlAI by using the ``-t`` flag.
For example, to execute the MTurk WikiMovies task we should call:

``python display_data.py -t mturkwikimovies``

To run the MNIST_QA task, while displaying the images in ascii format, we could call:

``python display_data.py -t mnist_qa -im ascii``

And for VQAv2:

``python display_data.py -t vqa_v2``
