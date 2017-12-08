# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""This module provides a set of teachers that deal with dialog:

    ``FixedDialogTeacher(Teacher)``
    Base class for teachers in tasks that have fixed dialog - i.e., dialog
    that is not dynamically generated but rather is pulled from set examples.
    However, the class can be extended to all tasks involved fixed data.
    Implements much of the basic functionality of these teachers, including
    ``observe()``, ``act()``, ``next_example()``

    ``DialogTeacher(FixedDialogTeacher)``
     Base teacher class for doing dialog specifically with fixed chat logs.

    ``FbDialogTeacher(DialogTeacher)``
     Teacher class that provides access to data in the Facebook Dialog format.
     See the class description for more details.


This module also includes ``DataLoader``, a threadpool data loader for ``FixedDialogTeacher``,
and ``DialogData``/``StreamDialogData``, data structures for accessing textual
dialog data and utilized by ``DialogTeacher``



"""
from .agents import Teacher, create_task_agent_from_taskname
from .image_featurizers import ImageLoader
from .utils import flatten, sort_data, make_batches

import concurrent.futures
import multiprocessing
from multiprocessing import Value
from threading import Thread
import queue
import random
import sys
import time
import os


class DataLoader(Thread):
    """A worker thread that provides a threadpool for data loading.

    A teacher may submit a request to the loader, which will return the
    appropriate data.

    To submit a request, a teacher should call ``request_load`` with the
    following arguments:
        - ``receive_fn`` - a receive function (for receiving the data)
        - ``load_fn`` - a load function (for loading the data)
        - ``args`` - arguments for the load function
            -> args can be either a dictionary of arguments for a function, or
               a list of positional arguments
    """
    def __init__(self, opt):
        Thread.__init__(self, daemon=True)
        self.num_workers = opt.get('numthreads', 1)
        self.request_queue = queue.Queue()

    def request_load(self, receive_fn, load_fn, args):
        self.request_queue.put((receive_fn, load_fn, args))

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            while True:
                receive_fn, load_fn, args = self.request_queue.get()
                if type(args) == dict:
                    future = executor.submit(load_fn, **args)
                else:
                    future = executor.submit(load_fn, *args)
                receive_fn(future)


class FixedDialogTeacher(Teacher):
    """A teacher agent for all teachers involved in tasks with fixed data.

    This class provides the following functionality for its subclasses:

    - Resets a teacher
    - Provides an observe method
    - Computes and retrieves the next episode index for a teacher
    - Provides a threadpool option for loading data (especially useful for
      large data, e.g. images)

    To utilize the DataLoader for threadpool loading, a teacher should
    implement the ``submit_load_request`` function to send a load request
    to the DataLoader by calling ``self.data_loader.request_load`` with the
    appropriate arguments (``receive_fn, load_fn, args``). The DataLoader then
    returns the data to the teacher's ``data_queue``, which the teacher can
    poll in its ``act`` method.

    The following is an example of the DataLoader usage in the VQA-V1 teacher.

        1. In the teacher's ``init`` function, the teacher calls its
           ``submit_load_request`` function to preload an image.
        2. The ``submit_load_request`` function gets the next ``episode_idx``,
           and computes the image path for the load request.
        3. At the end of ``submit_load_request``, the teacher calls
           ``self.data_loader.request_load`` with three args:
           - ``self.receive_data`` - the function that the DataLoader calls to
               return the the loaded object
           - ``self.image_loader.load`` - the function used to load the image
               from the image path
           - ``[img_path]`` - a list of arguments for the load function, which
               in this case is the path of the image.
         4. In the teacher's ``act`` function, the teacher loads the data from
            its data queue.
         5. At the end of the ``act`` function, the teacher calls
            ``submit_load_request`` to preload an image for the next example.


    """
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if not hasattr(self, 'datatype'):
            self.datatype = opt['datatype']
        if not hasattr(self, 'random'):
            self.random = self.datatype == 'train'
        if not hasattr(self, 'training'):
            self.training = self.datatype.startswith('train')

        # set up support for multithreaded data loading
        self.data_queue = queue.Queue()
        if shared:
            self.data_loader = shared['data_loader']
            self.index = shared['index']
        else:
            self.data_loader = DataLoader(opt)
            self.data_loader.start()
            self.index = FixedDialogTeacher.AttrDict(value=-1)

        # set up batching
        self.bsz = opt.get('batchsize', 1)
        self.batchindex = opt.get('batchindex', 0)

        dt = opt.get('datatype', '').split(':')
        self.use_batch_act = (opt.get('batch_sort', False) and self.bsz > 1
                              and 'stream' not in dt)

        if self.use_batch_act:
            if shared:
                self.lastYs = shared['lastYs']
            else:
                self.lastYs = [None] * self.bsz
                ordered_opt = opt.copy()
                ordered_opt['datatype'] = ':'.join((dt[0], 'ordered'))
                ordered_opt['batchsize'] = 1
                ordered_opt['numthreads'] = 1
                ordered_teacher = create_task_agent_from_taskname(ordered_opt)[0]

                clen = opt.get('context_length', -1)
                incl = opt.get('include_labels', True)

                if ordered_teacher.num_examples() > 1000000:  # one million
                    print('WARNING: this dataset is large, and batch sorting '
                          'may use too much RAM or take too long to set up. '
                          'Consider disabling batch sorting, setting '
                          'context-length to a small integer (if this dataset '
                          'has episodes of multiple examples), or streaming '
                          'the data using a streamed data mode if supported.')

                flatdata = flatten(ordered_teacher,
                                   context_length=clen, include_labels=incl)
                self.sorted_data = sort_data(flatdata)
                self.batches = make_batches(self.sorted_data, self.bsz)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def _lock(self):
        if hasattr(self.index, 'get_lock'):
            return self.index.get_lock()
        else:
            return self

    def reset(self):
        """Reset the dialog so that it is at the start of the epoch,
        and all metrics are reset.
        """
        super().reset()
        self.metrics.clear()
        self.lastY = None
        self.episode_done = True
        self.epochDone = False
        self.data_queue = queue.Queue()

        self.episode_idx = -1
        with self._lock():
            self.index.value = -1
        if self.use_batch_act and self.random and hasattr(self, 'batches'):
            random.shuffle(self.batches)

    def submit_load_request(self):
        """An agent should implement this method to submit requests to the
        data loader. At the end of this method, the agent should call
        ``self.data_loader.request_load()`` with the appropriate args.
        """
        pass

    def receive_data(self, future):
        """Function for receiving data from the data loader."""
        data = future.result()
        self.data_queue.put(data)

    def share(self):
        shared = super().share()
        shared['data_loader'] = self.data_loader

        if hasattr(self, 'lastYs'):
            # share lastYs to communicate between batch_act and observe
            shared['lastYs'] = self.lastYs

        if (self.opt.get('numthreads') > 1 and
            type(self.index) is not multiprocessing.sharedctypes.Synchronized):
            # for multithreading need to move index into shared / locked memory
            self.index = Value('l', -1)
        shared['index'] = self.index

        return shared

    def next_episode_idx(self, num_eps=None, loop=None):
        if num_eps is None:
            num_eps = self.num_episodes()
        if loop is None:
            loop = self.training
        if self.random:
            new_idx = random.randrange(num_eps)
        else:
            with self._lock():
                self.index.value += 1
                if loop:
                    self.index.value %= num_eps
                new_idx = self.index.value
        return new_idx

    def next_example(self):
        if self.episode_done:
            self.episode_idx = self.next_episode_idx()
            self.entry_idx = 0
        else:
            self.entry_idx += 1

        if self.episode_idx >= self.num_episodes():
            return {'episode_done': True}, True

        ex = self.get(self.episode_idx, self.entry_idx)
        self.episode_done = ex['episode_done']

        if (not self.random and self.episode_done
                and self.episode_idx + 1 >= self.num_episodes()):
            epoch_done = True
        else:
            epoch_done = False

        return ex, epoch_done

    def num_episodes(self):
        """Get the number of episodes in this dataset."""
        if self.use_batch_act:
            # when using batch_act, this is length of sorted data
            return len(self.sorted_data)
        raise RuntimeError('"num_episodes" must be overriden by children.')

    def num_examples(self):
        """Get the total number of examples in this dataset."""
        if self.use_batch_act:
            # when using batch_act, this is length of sorted data
            return len(self.sorted_data)
        raise RuntimeError('"num_examples" must be overriden by children.')

    def get(self, episode_idx, entry_idx=0):
        """Get the specified episode and the specified entry in that episode.

        Many datasets have only single-entry episodes, so entry_idx defaults to
        zero. Children must override this method in order to inherit the
        `next_example` method.
        """
        raise RuntimeError('"Get" method must be overriden by children.')

    def observe(self, observation):
        """Process observation for metrics."""
        if self.use_batch_act:
            self.lastY = self.lastYs[self.batchindex]

        if hasattr(self, 'lastY') and self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    def batch_act(self, observations):
        # we ignore observations
        if not hasattr(self, 'epochDone'):
            # reset if haven't yet
            self.reset()

        # get next batch
        with self._lock():
            self.index.value += 1
            if self.training:
                self.index.value %= len(self.batches)
            batch_idx = self.index.value

            if batch_idx + 1 >= len(self.batches):
                if self.random:
                    random.shuffle(self.batches)
                self.epochDone = True
            else:
                self.epochDone = False

        if batch_idx >= len(self.batches):
            return [{'episode_done': True, 'id': self.getID()}] * self.bsz

        batch = self.batches[batch_idx]

        # pad batch
        if len(batch) < self.bsz:
            batch += [{'episode_done': True, 'id': self.getID()}] * (self.bsz - len(batch))

        # remember correct answer if available (for padding, None)
        for i, ex in enumerate(batch):
            self.lastYs[i] = ex.get('labels', ex.get('eval_labels'))

        return batch

    def act(self):
        """Send new dialog message."""
        if not hasattr(self, 'epochDone'):
            # reset if haven't yet
            self.reset()

        # get next example, action is episode_done dict if already out of exs
        action, self.epochDone = self.next_example()
        action['id'] = self.getID()

        # remember correct answer if available
        self.lastY = action.get('labels', None)
        if not self.datatype.startswith('train') and 'labels' in action:
            # move labels to eval field so not used for training
            # but this way the model can use the labels for perplexity or loss
            action['eval_labels'] = action.pop('labels')

        return action


class DialogTeacher(FixedDialogTeacher):
    """A base teacher class for doing dialog with fixed chat logs.

    This class provides a set a basic functionality:

    - uses data class to store and query text data
    - generates action tables to send to the student agent from the data
    - metrics tracking count of sent vs correctly answered queries

    If you have ``opt.numthreads > 1``, this also activates a shared memory
    array for the data and lock-protected shared-memory metrics.

    In order to subclass this class, you must implement ``setup_data()`` in your
    class (or subclass another class which does, like ``FbDialogTeacher``), which
    reads your data file as an iterator.
    """

    def __init__(self, opt, shared=None):
        # Check for setup_data
        if not hasattr(self, 'setup_data'):
            raise RuntimeError('Must implement setup_data or subclass a class' +
                               ' which implements it (e.g. FbDialogTeacher)' +
                               ' in order to use this class.')
        super().__init__(opt, shared)

        self.startTime = time.time()
        self.datatype = opt['datatype']
        self.training = self.datatype.startswith('train')
        self.stream = 'stream' in self.datatype.split(':')

        if not self.use_batch_act:
            # first initialize any shared objects
            data_class = StreamDialogData if self.stream else DialogData
            kwargs = {'cycle': self.training} if self.stream else {}
            if shared and shared.get('data'):
                self.data = data_class(opt, shared=shared['data'], **kwargs)
            else:
                self.data = data_class(opt, data_loader=self.setup_data,
                    cands=self.label_candidates(), **kwargs)

        self.reset()

    def reset(self):
        # Reset the dialog so that it is at the start of the epoch,
        # and all metrics are reset.
        super().reset()
        if self.stream:
            self.data.reset()
            self.epochDone = False

    def share(self):
        shared = super().share()
        if hasattr(self, 'data'):
            shared['data'] = self.data.share()
        return shared

    def label_candidates(self):
        """Returns ``None`` by default, but override this in children (such as
        ``FbDialogTeacher``) to load up candidate labels for every example.
        """
        return None

    def num_episodes(self):
        try:
            return self.data.num_episodes()
        except AttributeError:
            return super().num_episodes()

    def num_examples(self):
        try:
            return self.data.num_examples()
        except AttributeError:
            return super().num_examples()

    def get(self, episode_idx, entry_idx=0):
        return self.data.get(episode_idx, entry_idx)[0]

    def next_example(self):
        if self.stream:
            action, epoch_done = self.data.get()
        else:
            action, epoch_done = super().next_example()
        return action, epoch_done


class DialogData(object):
    """Provides a data structure for accessing textual dialog data.
    This can be used whenever the dialog data is a fixed log of chats
    (i.e not a simulator setting). The logs can include dialog text and possibly
    supervised labels, candidate labels and rewards.

    All these are stored in this internal data format which is used by the
    ``DialogTeacher`` class.

    ``data_loader`` is an iterable, with each call returning:

        ``(x, ...), new_episode?``

        Where

        - ``x`` is a query and possibly context

        ``...`` can contain additional fields, specifically

        - ``y`` is an iterable of label(s) for that query
        - ``r`` is the str reward for getting that query correct
        - ``c`` is an iterable of label candidates that the student can choose from
        - ``i`` is a str path to an image on disk, which will be loaded by the data
          class at request-time. should always point to the raw image file.
        - ``new_episode?`` is a boolean value specifying whether that example is the start of a new episode. If you don't use episodes set this to ``True`` every time.


    ``cands`` can be set to provide a list of candidate labels for every example
    in this dataset, which the agent can choose from (the correct answer
    should be in this set).


    ``random`` tells the data class whether or not to visit episodes sequentially
    or randomly when returning examples to the caller.
    """

    def __init__(self, opt, data_loader=None, cands=None, shared=None, **kwargs):
        # self.data is a list of episodes
        # each episode is a tuple of entries
        # each entry is a tuple of values for the action/observation table
        if shared:
            self.image_loader = shared.get('image_loader', None)
            self.data = shared.get('data', [])
            self.cands = shared.get('cands', None)
        else:
            self.image_loader = ImageLoader(opt)
            self.data = []
            self._load(data_loader, opt['datafile'])
            self.cands = None if cands == None else set(sys.intern(c) for c in cands)
        self.addedCands = []
        self.copied_cands = False

    def share(self):
        shared = {
            'data': self.data,
            'cands': self.cands,
            'image_loader': self.image_loader
        }
        return shared

    def _read_episode(self, data_generator):
        """Reads one episode at a time from the provided iterator over entries.
        """
        episode = []
        last_cands = None
        for entry, new in data_generator:
            if new and len(episode) > 0:
                yield tuple(episode)
                episode = []
                last_cands = None

            # intern all strings so we don't store them more than once
            new_entry = []
            if len(entry) > 0:
                # process text if available
                if entry[0] is not None:
                    new_entry.append(sys.intern(entry[0]))
                else:
                    new_entry.append(None)
                if len(entry) > 1:
                    # process labels if available
                    if entry[1] is None:
                        new_entry.append(None)
                    elif hasattr(entry[1], '__iter__') and type(entry[1]) is not str:
                        # make sure iterable over labels, not single string
                        new_entry.append(tuple(sys.intern(e) for e in entry[1]))
                    else:
                        raise TypeError('Must provide iterable over labels, not a single string.')
                    if len(entry) > 2:
                        # process reward if available
                        if entry[2] is not None:
                            new_entry.append(entry[2])
                        else:
                            new_entry.append(None)
                        if len(entry) > 3:
                            # process label candidates if available
                            if entry[3] is None:
                                new_entry.append(None)
                            elif last_cands and entry[3] is last_cands:
                                # if cands are shared, say "same" so we
                                # don't store them again
                                new_entry.append(
                                    sys.intern('same as last time'))
                            elif hasattr(entry[3], '__iter__') and type(entry[3]) is not str:
                                # make sure iterable over candidates, not single string
                                last_cands = entry[3]
                                new_entry.append(tuple(
                                    sys.intern(e) for e in entry[3]))
                            else:
                                raise TypeError('Must provide iterable over label candidates, not a single string.')
                            if len(entry) > 4 and entry[4] is not None:
                                new_entry.append(sys.intern(entry[4]))

            episode.append(tuple(new_entry))

        if len(episode) > 0:
            yield tuple(episode)

    def _load(self, data_loader, datafile):
        """Loads up data from an iterator over tuples described in the class
        docs.
        """
        for episode in self._read_episode(data_loader(datafile)):
            self.data.append(episode)

    def num_episodes(self):
        """Return number of episodes in the dataset."""
        return len(self.data)

    def num_examples(self):
        """Returns total number of entries available. Each episode has at least
        one entry, but might have many more.
        """
        return sum(len(episode) for episode in self.data)

    def get(self, episode_idx, entry_idx=0):
        """Returns a specific entry from the dataset."""
        # first look up data
        episode = self.data[episode_idx]
        entry = episode[entry_idx]
        episode_done = entry_idx == len(episode) - 1
        end_of_data = episode_done and episode_idx == len(self.data) - 1

        # now pack it in a action-observation dictionary
        table = self.build_table(entry)

        # last entry in this episode
        table['episode_done'] = episode_done
        return table, end_of_data

    def build_table(self, entry):
        """Packs an entry into an action-observation dictionary."""
        table = {}
        if entry[0] is not None:
            table['text'] = entry[0]
        if len(entry) > 1:
            if entry[1] is not None:
                table['labels'] = entry[1]
            if len(entry) > 2:
                if entry[2] is not None:
                    table['reward'] = entry[2]
                if len(entry) > 3:
                    if entry[3] is not None:
                        table['label_candidates'] = entry[3]
                    if len(entry) > 4 and entry[4] is not None:
                        img = self.image_loader.load(entry[4])
                        if img is not None:
                            table['image'] = img

        if (table.get('labels', None) is not None
                and self.cands is not None):
            if self.addedCands:
                # remove elements in addedCands
                self.cands.difference_update(self.addedCands)
                self.addedCands.clear()
            for label in table['labels']:
                if label not in self.cands:
                    # add labels, queue them for removal next time
                    if not self.copied_cands:
                        self.cands = self.cands.copy()
                        self.copied_cands = True
                    self.cands.add(label)
                    self.addedCands.append(label)
            table['label_candidates'] = self.cands

        if 'labels' in table and 'label_candidates' in table:
            if table['labels'][0] not in table['label_candidates']:
                raise RuntimeError('true label missing from candidate labels')
        return table


class StreamDialogData(DialogData):
    """Provides a data structure for streaming textual dialog data.
    This can be used whenever the dialog data follows the format described in
    DialogData but cannot fit entirely into memory.

    Additional keyword-argument cycle defines if the stream should restart from
    the beginning after an epoch is finished (defaults to True).
    """

    def __init__(self, opt, data_loader=None, cands=None, shared=None, **kwargs):
        # super() call initiates stream in self.data by calling _load()
        super().__init__(opt, data_loader, cands, shared, **kwargs)
        self.cycle = kwargs['cycle'] if 'cycle' in kwargs else True
        if shared:
            # auxiliary instances hold pointer to main datastream (in self.data)
            self.reset_data = shared['reset']
            # Share datafile and data_loader for computing num_exs and num_eps
            self.datafile = shared['datafile']
            self.data_loader = shared['data_loader']
        else:
            # main instance holds the stream and shares pointer to it
            self.data_loader = data_loader
            self.datafile = opt['datafile']
            self.reset_data = None
            self.is_reset = True
        self.entry_idx = 0
        self.next_episode = None
        self.num_eps = None
        self.num_exs = None

    def share(self):
        shared = super().share()
        # also share reset method to allow datastream to be reset
        shared['reset'] = self.reset
        # share datafile and data for loading length if necessary
        shared['datafile'] = self.datafile
        shared['data_loader'] = self.data_loader

        return shared

    def _load(self, data_loader, datafile):
        """Load data generator into data field."""
        self.data = self._data_generator(data_loader, datafile)

    def _data_generator(self, data_loader, datafile):
        """Generates data using the iterator over tuples constructed
        by data_loader.
        """
        self.is_reset = False
        while True:
            for episode in self._read_episode(data_loader(datafile)):
                yield episode
            yield -1
            while not self.cycle:
                yield -1

    def load_length(self):
        datafiles = self.datafile if type(self.datafile) is tuple else [self.datafile]
        length_file = datafiles[0] + ".lengths"
        if not os.path.isfile(length_file):
            num_eps = 0
            num_exs = 0
            for episode in self._read_episode(self.data_loader(self.datafile)):
                num_eps += 1
                num_exs += len(episode)
            with open(length_file, 'w') as f:
                f.write("{}\n{}".format(num_eps, num_exs))
        else:
            with open(length_file, 'r') as f:
                num_eps, num_exs = f.readlines()
        return int(num_eps), int(num_exs)

    def num_examples(self):
        if not self.num_exs:
            self.num_eps, self.num_exs = self.load_length()
        return self.num_exs

    def num_episodes(self):
        if not self.num_eps:
            self.num_eps, self.num_exs = self.load_length()
        return self.num_eps

    def get(self):
        """Returns a the next entry from the stream in the current episode for
        this instance. When episode is done returns first entry of next episode.
        """
        # first look up data
        if self.next_episode is None:
            self.next_episode = next(self.data)
        if self.entry_idx == 0:
            self.cur_episode = self.next_episode
            self.next_episode = next(self.data)
        entry = self.cur_episode[self.entry_idx]

        # now pack it in a action-observation dictionary
        table = self.build_table(entry)

        episode_done = self.entry_idx == len(self.cur_episode) - 1
        if episode_done:
            self.entry_idx = 0
        else:
            self.entry_idx += 1
        end_of_data = episode_done and self.next_episode is -1
        if end_of_data and self.cycle:
            self.next_episode = next(self.data)

        # last entry in this episode
        table['episode_done'] = episode_done
        return table, end_of_data

    def reset(self):
        """Reset the datastream to its beginning"""
        if self.reset_data is not None:
            # auxiliary instance, reset main datastream
            self.data = self.reset_data()
            self.next_episode = None
        elif not self.is_reset:
            # if main instance is not reset, reset datastream
            self._load(self.data_loader, self.datafile)
            self.is_reset = True
            self.next_episode = None
        self.entry_idx = 0
        return self.data


class FbDialogTeacher(DialogTeacher):
    """This module provides access to data in the Facebook Dialog format.


    Subclasses ``DialogTeacher`` for functionality and provides an implementation
    of ``setup_data()`` which iterates over datasets in the "fbdialog" format.

    The way FB Dialog data is set up is as follows:

    ::

        1 Sam went to the kitchen.
        2 Pat gave Sam the milk.
        3 Where is the milk?<TAB>kitchen<TAB>1<TAB>hallway|kitchen|bathroom
        4 Sam went to the hallway
        5 Pat went to the bathroom
        6 Where is the milk?<TAB>hallway<TAB>1<TAB>hallway|kitchen|bathroom

    Lines 1-6 represent a single episode, with two different examples: the first
    example is lines 1-3, and the second is lines 4-6.

    Lines 1,2,4, and 5 represent contextual information.

    Lines 3 and 6 contain a query, a label, a reward for getting the question
    correct, and three label candidates.

    Since both of these examples are part of the same episode, the information
    provided in the first example is relevant to the query in the second example
    and therefore the agent must remember the first example in order to do well.

    In general dialog in this format can be any speech, not just QA pairs:

    ::

        1 Hi how's it going?<TAB>It's going great. What's new?
        2 Well I'm working on a new project at work.<TAB>Oh me too!
        3 Oh cool!<TAB>Tell me about yours.

    etc.
    """

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.cloze = opt.get('cloze', False)
        if shared and 'cands' in shared:
            self.cands = shared['cands']
        else:
            self.cands = self.load_cands(opt.get('cands_datafile', None))
        super().__init__(opt, shared)

    def share(self):
        shared = super().share()
        shared['cands'] = self.cands
        return shared

    def label_candidates(self):
        return self.cands

    def load_cands(self, path):
        """Load global fixed set of candidate labels that the teacher provides
        every example (the true labels for a specific example are also added to
        this set, so that it's possible to get the right answer).
        """
        if path is None:
            return None
        cands = []
        lines_have_ids = False
        cands_are_replies = False
        cnt = 0
        with open(path) as read:
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) > 0:
                    cnt = cnt + 1
                    # If lines are numbered we strip them of numbers.
                    if cnt == 1 and line[0:2] == '1 ':
                        lines_have_ids = True
                    # If tabs then the label_candidates are all the replies.
                    if '\t' in line and not cands_are_replies:
                        cands_are_replies = True
                        cands = []
                    if lines_have_ids:
                        space_idx = line.find(' ')
                        line = line[space_idx + 1:]
                        if cands_are_replies:
                            sp = line.split('\t')
                            if len(sp) > 1 and sp[1] != '':
                                cands.append(sp[1])
                        else:
                            cands.append(line)
                    else:
                        cands.append(line)
        return cands

    def setup_data(self, path):
        """Reads data in the fbdialog format.

        Returns ``((x,y,r,c), new_episode?)`` tuples.

        ``x`` represents a query, ``y`` represents the labels, ``r`` represents any reward,
        and ``c`` represents any label_candidates.

        The example above will be translated into the following tuples:

        ::

            x: 'Sam went to the kitchen\\nPat gave Sam the milk\\nWhere is the milk?'
            y: ['kitchen']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = True (this is the first example in the episode)


        ::

            x: 'Sam went to the hallway\\nPat went to the bathroom\\nWhere is the
                milk?'
            y: ['hallway']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = False (this is the second example in the episode)
        """
        print("[loading fbdialog data:" + path + "]")
        with open(path) as read:
            start = True
            x = ''
            reward = 0
            dialog_index = 0
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) == 0:
                    continue

                # first, get conversation index -- '1' means start of episode
                space_idx = line.find(' ')
                conv_id = line[:space_idx]

                # split line into constituent parts, if available:
                # x<tab>y<tab>reward<tab>label_candidates
                # where y, reward, and label_candidates are optional
                split = line[space_idx + 1:].split('\t')

                # remove empty items and strip each one
                for i in range(len(split)):
                    word = split[i].strip()
                    if len(word) == 0:
                        split[i] = ''
                    else:
                        split[i] = word
                # Empty reward string same as None
                if len(split) > 2 and split[2] == '':
                    split[2] = None

                # now check if we're at a new episode
                if conv_id == '1':
                    dialog_index += 1
                    x = x.strip()
                    if x:
                        yield [x, None, reward], start
                    start = True
                    reward = 0
                    # start a new episode
                    if self.cloze:
                        x = 'Fill in the blank in the last sentence.\n{x}'.format(
                            x=split[0]
                        )
                    else:
                        x = split[0]
                else:
                    if x:
                        # otherwise add current x to what we have so far
                        x = '{x}\n{next_x}'.format(x=x, next_x=split[0])
                    else:
                        x = split[0]
                if len(split) > 2 and split[2]:
                    reward += float(split[2])

                if len(split) > 1 and split[1]:
                    # only generate an example if we have a y
                    split[0] = x
                    # split labels
                    split[1] = split[1].split('|')
                    if len(split) > 3:
                        # split label_candidates
                        split[3] = split[3].split('|')
                    if len(split) > 2:
                        split[2] = reward
                    else:
                        split.append(reward)
                    if start:
                        yield split, True
                        start = False
                    else:
                        yield split, False
                    # reset x in case there is unlabeled data still left
                    x = ''
                    reward = 0
            if x:
                yield [x, None, reward], start
