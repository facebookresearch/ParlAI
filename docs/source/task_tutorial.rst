..
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.
  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.

Creating a New Task
===================
**Author**: Filipe de Avila Belbute Peres

Adding new tasks to ParlAI is a simple process. In this tutorial we will go over the different ways a new task can be created.

Tasks code is located in the ``parlai/tasks`` directory. You will need to create a directory for your new task there. (Don't forget to create an ``__init__.py`` file.) The code for the tasks in this tutorial can also be found in this directory.


Summary
^^^^^^^

In brief, to add your own task you need to:

1. Implement ``build.py`` to `download and build any needed data <http://parl.ai/static/docs/task_tutorial.html#part-1-building-the-data>`__.
2. Implement ``agents.py``, with at least a ``DefaultTeacher`` (extending ``Teacher`` or one of its children)

    - if your data is in FB Dialog format, subclass `FbDialogTeacher`_.
    - if your data consists of fixed logs, you can extend `DialogTeacher`_, in which case you just need to write your own ``setup_data()`` function, which provides an iterable over the data.
    - if your data uses other fields, build your `task from scratch`_, by subclassing ``Teacher`` and writing your own ``act()`` method, which will provide observations from your task each time it's called.

3. Add the task to the `task list <http://parl.ai/static/docs/task_tutorial.html#part-3-add-task-to-task-list>`__.

Below we go into more details for each of these steps.


Part 1: Building the Data
^^^^^^^^^^^^^^^^^^^^^^^^^

We first need to create functionality for downloading and setting up the dataset that is going to be used for the task. This is done in the ``build.py`` file. Useful functionality for setting up data can be found in ``parlai.core.build_data``. We thus start by importing it:

.. code-block:: python

    import parlai.core.build_data as build_data
    import os

Now we define our build method, which takes in the argument ``opt``, which contains parsed arguments from the command line (or their default), including the path to the data directory. We can also define a version string, so that the data is updated automatically in case there is a new version (here it was just left as ``None`` as the MNIST dataset doesn't have a version). We then use the build_data utilities to check if this data hasn't been previously built or if the version is outdated. If not, we proceed to creating the directory for the data, and then downloading and uncompressing it. Finally, we mark the build as done, so that ``build_data.built`` returns true from now on. Below is an example of setting up the MNIST dataset.

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
            url = 'https://s3.amazonaws.com/fair-data/parlai/mnist/' + fname # dataset URL
            build_data.download(url, dpath, fname)

            # uncompress it
            build_data.untar(dpath, fname)

            # mark the data as built
            build_data.mark_done(dpath, version_string=version)



Part 2: Creating the Teacher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we have our data, we need an agent that understand the task's structure and is able to present it. In other words, we need a ``Teacher``. Every task requires an ``agents.py`` file in which we define the agents for the task. It is there that we will define our teacher.

The simplest method available for creating a teacher is to use the ``FbDialogTeacher`` class, which makes the process very simple for text data already formatted in the Facebook Dialog format. (In fact, even if your text data is not in the Facebook Dialog format, it might be simpler to parse it into this format and use the ``FbDialogTeacher``.) This is shown in the section `FbDialogTeacher`_.

If the data is not in this format or there are different requirements, one can still use the ``DialogTeacher`` which automates much of the work in setting up a dialog task, but gives the user more flexibility in setting up the data. This is shown in the section `DialogTeacher`_.

Finally, if the requirements for the task do not fit any of the above, one can still write a task from scratch without much trouble. This is shown in the section `Task from Scratch`_.


FbDialogTeacher
~~~~~~~~~~~~~~~

In this section we will illustrate the process of using the ``FbDialogTeacher`` class by adding the `MTurk WikiMovies <http://parl.ai/static/docs/tasks.html#mturk-wikimovies>`__ question-answering task. This task has data in textual form and has been formatted to follow the Facebook Dialog format. It is thus very simple to implement it using ``FbDialogTeacher``. More information on this class and the dialog format can be found `here <http://parl.ai/static/docs/fbdialog.html>`__.

In this task, the agent is presented with questions about movies that are answerable from Wikipedia. A sample dialog is demonstrated below.

::

    [mturkwikimovies]: Which directors collaborated for the film Flushed Away?
    [labels: David Bowers, Sam Fell]
    [cands: David Rose|Ismail Kadare|Alexis Díaz de Villegas|emily blunt|Glory| ...and 75537 more]
       [Agent]: David Bowers, Sam Fell

Every task requires a ``DefaultTeacher``. We will thus create one for this task. Since we are subclassing ``FbDialogTeacher``, we only have to initialize the class and set a few option parameters, as shown below.

.. code-block:: python

    class DefaultTeacher(FbDialogTeacher):
        def __init__(self, opt, shared=None):
            opt = copy.deepcopy(opt)

            # get datafile
            opt['datafile'] = _path(opt, '')

            # get file with candidate answers
            opt['cands_datafile'] = os.path.join(opt['datapath'], 'WikiMovies',
                                                 'movieqa', 'knowledge_source',
                                                 'entities.txt')
            super().__init__(opt, shared)

We can notice there was a call to a ``_path()`` method, which returns the path to the correct datafile. The path to the file is then stored in the options dictionary under the ``'datafile'`` key. We still need to implement this ``_path()`` method. The version for this example is presented below. It first ensures the data is built by calling the ``build()`` method described above. It then sets up the paths for the built data.

.. code-block:: python

    def _path(opt, filtered):
        # ensure data is built
        build(opt)

        # set up path to data (specific to each dataset)
        dt = opt['datatype'].split(':')[0]
        if dt == 'valid':
            dt = 'dev'
        return os.path.join(opt['datapath'], 'MTurkWikiMovies', 'mturkwikimovies',
                            'qa-{type}.txt'.format(type=dt))

And this is all that needs to be done to create a teacher for our task using ``FbDialogTeacher``.


DialogTeacher
~~~~~~~~~~~~~

In this section we will demonstrate the process of using the ``DialogTeacher`` class by adding a simple question-answering task based on the MNIST dataset. This task depends on visual data and so does not fit the ``FbDialogTeacher`` class described above. Still, using ``DialogTeacher`` makes it easy to implement dialog tasks such as this one.

In this task, the agent is presented with the image of a digit and then asked to answer which number it is seeing. A sample episode is demonstrated below.

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

            # _path method explained below, returns paths to images and labels
            labels_path, self.image_path = _path(opt)

            # store path to label data in options dictionary
            opt['datafile'] = labels_path

            # store identifier for the teacher in the dialog
            self.id = 'mnist_qa'

            # strings for the labels in the class (digits)
            # (information specific to this task)
            self.num_strs = ['zero', 'one', 'two', 'three', 'four', 'five',
                    'six', 'seven', 'eight', 'nine']

            super().__init__(opt, shared)

The ``id`` field names the teacher in the dialog. The ``num_strs`` field is specific to this example task. It is being used simply to store the text version of the digits.

More importantly, we can notice there was a call to a ``_path()`` method, which returns the paths to the image files and the labels. The path to the file is then stored in the options dictionary under the ``'datafile'`` key. This key should be used to store data that will be useful for performing the task.

We still need to implement this ``_path()`` method. The version for this example is presented below. It first ensures the data is built by calling the ``build()`` method described above. It then sets up the paths for the built data. This should be specific to the dataset being used. If your dataset does not use images, the ``image_path`` is not necessary, for example. Or if your task will use data other than labels, the path to the file containing this information can also be returned.

.. code-block:: python

    def _path(opt):
        # ensure data is built
        build(opt)

        # set up paths to data (specific to each dataset)
        dt = opt['datatype'].split(':')[0]
        labels_path = os.path.join(opt['datapath'], 'mnist', dt, 'labels.json')
        image_path = os.path.join(opt['datapath'], 'mnist', dt)
        return labels_path, image_path

By creating ``MnistQATeacher`` as a subclass of ``DialogTeacher``, the job of creating a teacher for this task becomes much simpler: most of the work that needs to be done will limit itself to defining a ``setup_data`` method. This method is a generator that will take in a path to the data and yield a pair of elements for each call. The first element of the pair is a tuple containing the following information: ``(query, labels, reward, label_candidates, path_to_image)``. The second is a boolean flag ``episode_done?`` which indicates if the current query marks the end of an episode or not.

More information on this format can be found in the documentation on ``data_loader`` in `DialogData <http://parl.ai/static/docs/dialog.html#parlai.core.dialog_teacher.DialogData>`__ (``setup_data`` is provided as a data_loader to ``DialogData``).

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
        episode_done = True

        # define iterator over all queries
        for i in range(len(self.labels)):
            # set up path to curent image
            img_path = os.path.join(self.image_path, '%05d.bmp' % i)
            # get current label, both as a digit and as a text
            label = [self.labels[i], self.num_strs[int(self.labels[i])]]
            # yield tuple with information and episode_done? flag
            yield (self.question, label, None, None, img_path), episode_done

As we can see from the code above, for this specific task the question is always the same, and thus it is fixed. For different tasks, this might change at each iteration. Similarly, for this task, each episode consists of only one query, thus ``episode_done?`` is always true (*i.e.*, each query is the end of its episode). This could also vary depending on the task.

Looking at the tuple provided by the iterator at each yield, we can see that we defined a query, a label and an image path. When working with ``DialogTeacher`` in visual tasks, it is important to provide the path to the image in the ``setup_data`` tuple. This allows one to inherit functionality around the "image-mode" command line parameter, such as automatically returning ascii versions of images if -im ascii is set.

Finally, one might notice that no reward or label candidates were provided in the tuple (both are set to ``None``). The reward is not specified because it is not useful for this task. The label candidates, however, were not specified per-example for this task because we instead use a single set of universal candidates for every example in this task (the digits from '0' to '9'). For cases like this, with fixed label candidates, one can simply define a method ``label_candidates()`` that returns the unchanging candidates, as demonstrated below. For cases where the label candidates vary for each query, the field in the tuple can be used.

.. code-block:: python

    def label_candidates(self):
        return [str(x) for x in range(10)] + self.num_strs

The only thing left to be done for this part is to define a ``DefaultTeacher`` class. This is a requirement for any task, since it defaults to this teacher when no one is specified. We can simply default to the class we have built so far.

.. code-block:: python

    class DefaultTeacher(MnistQATeacher):
        pass

And we have finished building our task.


Task from Scratch
~~~~~~~~~~~~~~~~~

In this section we will demonstrate the process of creating a task from scratch by adding the VQAv2 visual question-answering task. To implement this task we will inherit directly from the base ``Teacher`` class instead of using ``DialogTeacher``. This is usually not necessary, but it is done here as an example of creating a task from scratch.

In this task, the agent is presented with an image of a scene and then asked to answer a question about that scene. A sample episode is demonstrated below.

.. image:: _static/img/task_tutorial_skateboard.jpg

::

    [vqa_v2]: What is this man holding?
    [labels: skateboard]
       [Agent]: skateboard


We will call our teacher ``OeTeacher`` (for open-ended teacher, since it doesn't provide the agent with label candidates). Let's initialize this class first.

.. code-block:: python

    class OeTeacher(Teacher):
        def __init__(self, opt, shared=None):
            super().__init__(opt)
            # store datatype
            self.datatype = opt['datatype']
            # _path method explained below, returns paths to images and labels
            data_path, annotation_path, self.image_path = _path(opt)

            # setup data if it hasn't been provided in shared
            if shared and 'ques' in shared:
                self.ques = shared['ques']
                if 'annotation' in shared:
                    self.annotation = shared['annotation']
            else:
                self._setup_data(data_path, annotation_path)
            self.len = len(self.ques['questions'])

            # for ordered data in batch mode (especially, for validation and
            # testing), each teacher in the batch gets a start index and a step
            # size so they all process disparate sets of the data
            self.step_size = opt.get('batchsize', 1)
            self.data_offset = opt.get('batchindex', 0)

            # instantiate image loader for later usage
            self.image_loader = ImageLoader(opt)

            self.reset()

There are three important parts to this initialization. First, the call to the ``_path()`` method, which returns the paths to the data, annotation and image files. Second, setting up the data and handling the ``shared`` argument, which is used when initializing multiple teachers (*e.g.*, for batch training). It is a dictionary containing data that can be shared across instances of the class. Third, defining step sizes and offsets for walking over the data in batch mode. Let's look at each of these in order.

First, we need to implement the ``_path()`` method. The version for this example is presented below. It first ensures the data is built by calling the ``build()`` method described above. In this case, it also calls a ``buildImage()`` method, which downloads the images for this task. This method is analogous to ``build()`` and can be found in the same ``build.py`` file. It then sets up the paths for the built data. This should be specific to the dataset being used. If your dataset does not use images, the ``image_path`` is not necessary, for example. (The same applies to the ``image_loader``.)

.. code-block:: python

    def _path(opt):
        # ensure data is built
        build(opt)
        buildImage(opt)
        dt = opt['datatype'].split(':')[0]

        # verify datatype to decide which sub-dataset to load
        if dt == 'train':
            ques_suffix = 'v2_OpenEnded_mscoco_train2014'
            annotation_suffix = 'v2_mscoco_train2014'
            img_suffix = os.path.join('train2014', 'COCO_train2014_')
        elif dt == 'valid':
            ques_suffix = 'v2_OpenEnded_mscoco_val2014'
            annotation_suffix = 'v2_mscoco_val2014'
            img_suffix = os.path.join('val2014', 'COCO_val2014_')
        elif dt == 'test':
            ques_suffix = 'v2_OpenEnded_mscoco_test2015'
            annotation_suffix = 'None'
            img_suffix = os.path.join('test2015', 'COCO_test2015_')
        else:
            raise RuntimeError('Not valid datatype.')

        # set up paths to data
        data_path = os.path.join(opt['datapath'], 'VQA-v2',
                                 ques_suffix + '_questions.json')

        annotation_path = os.path.join(opt['datapath'], 'VQA-v2',
                                       annotation_suffix + '_annotations.json')

        image_path = os.path.join(opt['datapath'], 'COCO-IMG', img_suffix)

        return data_path, annotation_path, image_path

Now, we can look at how to setup the data and handle the ``shared`` argument. If an ``OeTeacher`` instance is the first one being created in a task execution, ``shared`` will be ``None``, and thus it will need to set up it's data. This is done in the ``_setup_data()`` method, pasted below. In the case of this task, ``_setup_data()`` simply loads the data (and possibly the annotations) and stores them as class attributes.

.. code-block:: python

    def _setup_data(self, data_path, annotation_path):
        # loads data
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.ques = json.load(data_file)
        # if testing load annotations
        if self.datatype != 'test':
            print('loading: ' + annotation_path)
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)

However, if the ``OeTeacher`` instance being created is not the first one for a certain task execution, we want to avoid having to reload the same data many times. For this to work we need to do two things. First, we define a ``share()`` method, which will set up the task-specific contents of the ``shared`` parameter. This method is presented below. It places the data we have just loaded in ``_setup_data()`` in the shared dictionary and returns it.

.. code-block:: python

    def share(self):
        shared = super().share()
        shared['ques'] = self.ques
        if hasattr(self, 'annotation'):
            shared['annotation'] = self.annotation
        return shared

Now that the data sharing is properly set up, when other instances of ``OeTeacher`` are created for a task execution, they will be able to use the ``shared`` argument passed to ``__init__()`` in order to use the already loaded data, as seen before.

We have also seen that we have set up ``self.step_size`` to the size of the batch and ``self.data_offset`` to the batch index, so that different teachers in a batch access diferent parts of the data. A method ``reset()`` is then called to initialize the data loading. Let's look at that method below. It first sets the attribute ``self.lastY`` to ``None``. This attribute will be used to hold the label for the last example seen by the instance. Then, ``self.episode_idx`` is set to a ``step_size`` below the ``data_offset``, so that when the first action is executed, it is incremented and starts exactly at the ``data_offset`` index.

.. code-block:: python

    def reset(self):
        # Reset the dialog so that it is at the start of the epoch,
        # and all metrics are reset.
        super().reset()
        self.lastY = None
        self.episode_idx = self.data_offset - self.step_size

Now that we are done with the class initialization, there are only a few steps left in creating the task. First, the ``OeTeacher`` requires a ``__len__()`` method that returns the size of the data it is presenting. Since ``self.len`` had already been defined in the initialization, this is easy to achieve.

.. code-block:: python

    def __len__(self):
        return self.len

The final step is to define the important ``act()`` and ``observe()`` methods, which are required of all agents in parlai. In the observe method we simply check if a prediction was made in the last step and if so update the metrics with the last observation and label and clear ``lastY``. This is important because it is the job of the ``Teacher`` to update the metrics.

.. code-block:: python

    def observe(self, observation):
        """Process observation for metrics."""
        if self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

In the act method we need to return the ``Teacher``'s action, which will then be presented to the agent(s) performing the task. In this case, this includes an image and a question. We first select which example to use: randomly in the case of training or sequentially in the case of validation/testing. The ``OeTeacher`` then loads the appropriate question, which is placed in the ``text`` field of the dict. The image_path is also constructed and an image object (loaded utilizing the ``ImageLoader`` class) is passed in the ``image`` field. The ``episode_done`` flag is always set to true in this task specifically due to the fact that all episodes consist of only one example.

.. code-block:: python

    def act(self):
        # pick random example if training, else proceed sequentially
        if self.datatype == 'train':
            self.episode_idx = random.randrange(self.len)
        else:
            self.episode_idx = (self.episode_idx + self.step_size) % len(self)
            if self.episode_idx == len(self) - self.step_size:
                self.epochDone = True
        # get question and image path for current example
        qa = self.ques['questions'][self.episode_idx]
        question = qa['question']
        image_id = qa['image_id']

        img_path = self.image_path + '%012d.jpg' % (image_id)
        # build action dict, all episodes consist of 1 example in this task
        action = {
            'image': self.image_loader.load(img_path),
            'text': question,
            'episode_done': True
        }
        # if not testing get annotations and set lastY
        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][self.episode_idx]
            self.lastY = [ans['answer'] for ans in anno['answers']]
        # if training, set fill labels field
        if self.datatype.startswith('train'):
            action['labels'] = self.lastY

        return action

The only thing left to be done for this part is to define a ``DefaultTeacher`` class. This is a requirement for any task, since it defaults to this teacher when no one is specified. We can simply default to the class we have built so far.

.. code-block:: python

    class DefaultTeacher(OeTeacher):
        pass

And we have finished building a task from scratch.



Part 3: Add Task to Task List
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that our task is complete, we must add an entry to the ``task_list.py`` file in ``parlai/tasks``. This file just contains a json-formatted list of all tasks, with each task represented as a dictionary. Sample entries for our tasks are presented below.

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

A simple way of testing the basic functionality in a task is to run the ``display_data.py`` example in the ``examples`` directory. Now that the work is done, we can pass it to ParlAI by using the ``-t`` flag. For example, to execute the MTurk WikiMovies task we should call:

``python display_data.py -t mturkwikimovies``

To run the MNIST_QA task, while displaying the images in ascii format, we could call:

``python display_data.py -t mnist_qa -im ascii``

And for VQAv2:

``python display_data.py -t vqa_v2``
