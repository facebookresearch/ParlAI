import numpy 
import os, gc
import cPickle
import copy
import logging

import threading
import Queue

import collections

logger = logging.getLogger(__name__)

class SSFetcher(threading.Thread):
    def __init__(self, parent, init_offset=0, init_reshuffle_count=1, eos_sym=-1,
                 skip_utterance=False, skip_utterance_predict_both=False):
        threading.Thread.__init__(self)
        self.parent = parent
        self.rng = numpy.random.RandomState(self.parent.seed)
        self.indexes = numpy.arange(parent.data_len)

        self.init_offset = init_offset
        self.init_reshuffle_count = init_reshuffle_count
        self.offset = 0
        self.reshuffle_count = 0

        self.eos_sym = eos_sym
        self.skip_utterance = skip_utterance
        self.skip_utterance_predict_both = skip_utterance_predict_both

    def apply_reshuffle(self):
        self.rng.shuffle(self.indexes)
        self.offset = 0
        self.reshuffle_count += 1

    def run(self):
        diter = self.parent
        # Initialize to previously set reshuffles and offset position
        while (self.reshuffle_count < self.init_reshuffle_count):
            self.apply_reshuffle()

        self.offset = self.init_offset

        while not diter.exit_flag:
            last_batch = False
            dialogues = []

            while len(dialogues) < diter.batch_size:
                if self.offset == diter.data_len:
                    if not diter.use_infinite_loop:
                        last_batch = True
                        break
                    else:
                        # Infinite loop here, we reshuffle the indexes
                        # and reset the self.offset
                        self.apply_reshuffle()

                index = self.indexes[self.offset]
                s = diter.data[index]

                # Flatten if this is a list of lists
                if len(s) > 0:
                    if isinstance(s[0], list):
                        s = [item for sublist in s for item in sublist]

                # Standard dialogue preprocessing
                if not self.skip_utterance:
                    # Append only if it is shorter than max_len
                    if diter.max_len == -1 or len(s) <= diter.max_len:
                        dialogues.append([s, self.offset, self.reshuffle_count])

                # Skip-utterance preprocessing
                else:
                    s = copy.deepcopy(s)
                    eos_indices = numpy.where(numpy.asarray(s) == self.eos_sym)[0]

                    if not s[0] == self.eos_sym:
                        eos_indices = numpy.insert(eos_indices, 0, [self.eos_sym])
                    if not s[-1] == self.eos_sym:
                        eos_indices = numpy.append(eos_indices, [self.eos_sym])
                    if len(eos_indices) > 2:
                        # Compute forward and backward targets
                        first_utterance_index = self.rng.randint(0, len(eos_indices)-2)
                        s_forward = s[eos_indices[first_utterance_index]:eos_indices[first_utterance_index+2]+1]

                        s_backward_a = s[eos_indices[first_utterance_index+1]:eos_indices[first_utterance_index+2]]
                        s_backward_b = s[eos_indices[first_utterance_index]:eos_indices[first_utterance_index+1]+1]

                        # Sometimes an end-of-utterance token is missing at the end.
                        # Therefore, we need to insert it here.
                        if s_backward_a[-1] == self.eos_sym or s_backward_b[0] == self.eos_sym:
                            s_backward = s_backward_a + s_backward_b
                        else:
                            s_backward = s_backward_a + [self.eos_sym] + s_backward_b

                    else:
                        s_forward = [self.eos_sym]
                        s_backward = [self.eos_sym]

                    if self.skip_utterance_predict_both:
                        # Append only if it is shorter than max_len
                        if diter.max_len == -1 or len(s_forward) <= diter.max_len:
                            dialogues.append([s_forward, self.offset, self.reshuffle_count])
                        if diter.max_len == -1 or len(s_backward) <= diter.max_len:
                            dialogues.append([s_backward, self.offset, self.reshuffle_count])
                    else:
                        # Append only if it is shorter than max_len
                        if self.rng.randint(0, 2) == 0:
                            if diter.max_len == -1 or len(s_forward) <= diter.max_len:
                                dialogues.append([s_forward, self.offset, self.reshuffle_count])
                        else:
                            if diter.max_len == -1 or len(s_backward) <= diter.max_len:
                                dialogues.append([s_backward, self.offset, self.reshuffle_count])

                self.offset += 1


            if len(dialogues):
                diter.queue.put(dialogues)

            if last_batch:
                diter.queue.put(None)
                return

class SSIterator(object):
    def __init__(self,
                 dialogue_file,
                 batch_size,
                 seed,
                 max_len=-1,
                 use_infinite_loop=True,
                 init_offset=0,
                 init_reshuffle_count=1,
                 eos_sym=-1,
                 skip_utterance=False,
                 skip_utterance_predict_both=False):

        self.dialogue_file = dialogue_file
        self.batch_size = batch_size
        self.init_offset = init_offset
        self.init_reshuffle_count = init_reshuffle_count
        self.eos_sym = eos_sym
        self.skip_utterance = skip_utterance
        self.skip_utterance_predict_both = skip_utterance_predict_both

        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.load_files()
        self.exit_flag = False

    def load_files(self):
        self.data = cPickle.load(open(self.dialogue_file, 'r'))
        self.data_len = len(self.data)
        logger.debug('Data len is %d' % self.data_len)

    def start(self):
        self.exit_flag = False
        self.queue = Queue.Queue(maxsize=1000)
        self.gather = SSFetcher(self, self.init_offset, self.init_reshuffle_count,
                                self.eos_sym, self.skip_utterance, self.skip_utterance_predict_both)
        self.gather.daemon = True
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        if self.exit_flag:
            return None
        
        batch = self.queue.get()
        if not batch:
            self.exit_flag = True
        return batch


