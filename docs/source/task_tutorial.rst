..
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.
  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.

Creating a New Task
===================

Adding new tasks to ParlAI is a simple process. In this tutorial we will go over the process of adding a simple question-answering task based on the MNIST dataset in order to exemplify how a new task can be created. Using the ``DialogeTeacher`` class makes it very easy to implement dialog tasks such as this one. 

In this task, the agent is presented with the image of a digit and then be asked to answer which number it is seeing. A sample episode is demonstrated below. 

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

The first thing to do is to create a directory for our task under the ``parlai/tasks`` directory. The code for this task can be found in the ``mnist_qa`` directory.


Part 1: Building the Data
^^^^^^^^^^^^^^^^^^^^^^^^^

Now we need to create functionality for downloading and setting up the dataset that is going to be used for the task. This is done in the ``build.py`` file. Useful functionality for setting up data can be found in ``parlai.core.build_data``. We thus start by importing it: 

.. code-block:: python

    import parlai.core.build_data as build_data
    import os

Now we define our build method, which takes in the argument ``opt``, which contains parsed arguments from the command line (or their default), including the path to the data directory. We then use the build_data utilities to check if this data has been previously built, so that work is only done once. If not, we proceed to creating the directory for the data, and then downloading and uncompressing it. Finally, we mark the build as done, so that ``build_data.built`` returns true from now on.

.. code-block:: python

    def build(opt):
        # get path to data directory
        dpath = os.path.join(opt['datapath'], 'mnist')
        
        # check if data had been previously built
        if not build_data.built(dpath):
            print('[building data: ' + dpath + ']')
            
            # make a clean directory
            build_data.remove_dir(dpath)
            build_data.make_dir(dpath)

            # download the data.
            fname = 'mnist.tar.gz'
            url = 'https://s3.amazonaws.com/fair-data/parlai/mnist/' + fname # dataset URL
            build_data.download(url, dpath, fname)

            # uncompress it
            build_data.untar(dpath, fname)

            # mark the data as built
            build_data.mark_done(dpath)


Part 2: Creating the Teacher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we have our data, we need an agent that understand the task's structure and is able to present it. In other words, we need a ``Teacher``. 

Every task requires an ``agents.py`` file in which we define the agents for the task. It is there that we will define our teacher, which we will call ``MnistQATeacher``. Let's initialize this class first.

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

More information on this format can be found in the documentation on ``data_loader`` in `DialogueData <http://parl.ai/static/docs/dialog.html#parlai.core.dialog_teacher.DialogData>`__ (``setup_data`` is provided as a data_loader to ``DialogueData``).

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

And we have finished building our task. (Don't forget to create an ``__init__.py`` file in the task directory, though.)

Part 3: Add Task to Task List
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that our task is complete, we must add an entry to the ``task_list.py`` file in ``parlai/tasks``. This file just contains a json-formatted list of all tasks, with each task represented as a dictionary. A sample entry for our task is presented below.

.. code-block:: python

    [
        # other tasks...
        {
                "id": "MNIST_QA",
                "display_name": "MNIST_QA",
                "task": "mnist_qa",
                "tags": [ "all", "Visual" ],
                "description": "Task which requires agents to identify which number they are seeing. From the MNIST dataset."
        },
        # other tasks...
     ]

Part 4: Executing the Task
^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple way of testing the basic functionality in the task is to run the ``display_data.py`` example in the ``examples`` directory. Now that our task is done, we can pass it to ParlAI by using the ``-t`` flag. For example, to execute this example, while displaying the images in ascii format, we could call:

``python display_data.py -t mnist_qa -im ascii``

This flag will work with any other example as well.
