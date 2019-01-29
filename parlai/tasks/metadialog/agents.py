import copy
import json
import os
import re

import numpy as np

from .build import build

from parlai.core.agents import MultiTaskTeacher
from parlai.core.metrics import aggregate_metrics
from parlai.core.teachers import FbDialogTeacher, ParlAIDialogTeacher
from projects.metadialog.utils import add_person_tokens

def _path(opt, filename_override=None):
    build(opt)
    dataroot = opt['dataroot']
    subtask = opt['subtask']
    st = None
    if subtask == 'dialog':
        st = 'hh'
    elif subtask == 'sentiment':
        st = 'st'
    elif subtask == 'feedback' or 'explanation':
        st = 'fb_a'
    else:
        print(f'######## Not found! {subtask}')
    dp = os.path.join(opt['datapath'], 'dialogue_sf', 'dialogue_sf_v01')
    dt = filename_override or opt.get('datatype', 'train').split(':')[0]
    filename = f'{dt}_{st}.txt'
    return os.path.join(dp, filename)

class MetadialogTeacher(ParlAIDialogTeacher):
    """Teacher for the MetadialogAgent

    opt['datatype'] determines whether we use the designated filepath ('train') or
        one of the eval files ('valid', 'test'), which are identical regardless of
        what training set is being used.

    Example:
    -t metadialog:dialog:train_a
        train on data/convai2meta/dialog/train_a.txt
        eval on data/convai2meta/dialog/valid.txt
        test on data/convai2meta/dialog/test.txt
    """
    def __init__(self, opt, shared):
        opt = copy.deepcopy(opt)

        if 'train' in opt['datatype']: # Use 'in' to also capture 'train:ordered:stream'
            # Use the filename explicitly given with the flag if available
            # Otherwise, use the filename passed in the task flag
            train_file_flag = f"{opt['subtask'][:3]}_train"
            if opt.get(train_file_flag, None):
                filename = opt[train_file_flag]
            else:
                filename = opt['task'].split(':')[-1]
            path = _path(opt, filename)
        else:
            # Use the filename explicitly given with the flag if available
            # Otherwise, use the datatype (valid.txt or test.txt)
            eval_file_flag = f"{opt['subtask'][:3]}_{opt['datatype']}"
            if opt.get(eval_file_flag, None):
                filename = opt[eval_file_flag]
            else:
                filename = opt['datatype'].split(':')[0]
            path = _path(opt, filename)

        if not os.path.exists(path):
            raise ValueError("Unrecognized filepath: {}".format(path))

        opt['parlaidialogteacher_datafile'] = path
        opt['datafile'] = path
        super().__init__(opt, shared)

    def _setup_data(self, path): # Make private method for ParlAIDialogTeacher
        """Reads data in the fbdialog format.

        Returns ``((x,y,r,c), new_episode?)`` tuples.
        """
        print("[ Loading metadialog text data:" + path + "]")
        self.episodes = []
        self.num_exs = 0
        self.max_train = self.opt.get('max_train', 0)
        with open(path, 'r') as f:
            for line in f.readlines():
                if self.max_train and self.num_exs >= self.max_train:
                    break
                parley = json.loads(line)

                # NOTE: History is trimmed here, not by TorchAgent (except in interactive mode)
                if self.opt['history_size'] == 0:
                    parley['context'] = '__null__'
                elif self.opt['history_size'] > 0:
                    utterances = re.split(r'__p\d__', parley['context'])[1:]
                    trimmed = utterances[-self.opt['history_size']:]
                    parley['context'] = add_person_tokens(trimmed, last_speaker=1)

                # WARNING: STRIPPING AWAY MEMORIES
                parley['memories'] = []

                episode = {
                    # 'text': '\n'.join(parley.get('memories', [])) + '\n' + parley['context'],
                    'text': parley['context'],
                    'labels': [parley['response']],
                    'label_candidates': parley.get('candidates', []),
                    'reward': parley.get('reward', 0),
                    'episode_done': True,
                }

                # Convert integer labels (e.g., polarization dataset) to strings
                episode['labels'] = [str(l) for l in episode['labels']]

                self.num_exs += 1
                self.episodes.append([episode])


class MetadialogMTLTeacher(MultiTaskTeacher):
    """Creates a teacher that is actually a set of teachers each based on
    a task string--each of these teachers will get called in turn,
    either randomly or in order.
    They are all in the same world (they are the same agent switching tasks).

    More specifically, this child class of MultiTaskTeacher supports multitask learning
    with batches (ensuring that all batches only have data from a single task at a time)
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        # TODO: allow user to specify other strategies for creating sampling_prob
        # uniform, batch_proportion, etc.
        num_batches = np.array([t.num_examples() / t.bsz for t in self.tasks])
        self.sampling_prob = num_batches / np.sum(num_batches)

        self.task_idx_assignment = -1  # This will be updated by BatchWorld
        self.new_task = True
        self.random = opt.get('datatype') == 'train'

    def observe(self, observation):
        return self.tasks[self.task_idx].observe(observation)

    def act(self):
        self.task_idx = self.get_task_index()
        if self.task_idx < 0:
            return {'episode_done': True}
        action = self.tasks[self.task_idx].act()
        # Pass the name of the task currently being worked on
        action['subtask'] = self.tasks[self.task_idx].opt['subtask']
        return action

    def get_task_index(self):
        if self.task_idx_assignment >= 0:
            # Use the assignment from the BatchWorld
            return self.task_idx_assignment
        else:
            # Just go through the tasks in order
            for i in range(len(self.tasks)):
                if not self.tasks[i].epoch_done():
                    return i
            # If this is reached, all tasks are done, so return sentinel
            return -1

    # We get most metrics from the agents, not the teachers
    def report(self):
        m = {'exs': sum(t.report()['exs'] for t in self.tasks)}
        return m

class DialogTeacher(MetadialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtask'] = 'dialog'
        super().__init__(opt, shared)

class ExplanationTeacher(MetadialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtask'] = 'explanation'
        super().__init__(opt, shared)

class SentimentTeacher(MetadialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtask'] = 'sentiment'
        super().__init__(opt, shared)

class DiaexpTeacher(MetadialogMTLTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtasks'] = ['dialog', 'explanation']
        # Expand abbreviated task name ('both') into full task names
        if opt['task'].split(':')[-2] == 'diaexp':
            train_files = [opt['dia_train'], opt['exp_train']]
            assert(len(opt['subtasks']) == len(train_files))
            tasks = [f'metadialog:{subtask}:{train_file}' for subtask, train_file
                in zip(opt['subtasks'], train_files)]
            opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)

class DiasenTeacher(MetadialogMTLTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtasks'] = ['dialog', 'sentiment']
        # Expand abbreviated task name ('both') into full task names
        if opt['task'].split(':')[-2] == 'diasen':
            train_files = [opt['dia_train'], opt['sen_train']]
            assert(len(opt['subtasks']) == len(train_files))
            tasks = [f'metadialog:{subtask}:{train_file}' for subtask, train_file
                in zip(opt['subtasks'], train_files)]
            opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)

class AllTeacher(MetadialogMTLTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtasks'] = ['dialog', 'explanation', 'sentiment']
        # Expand abbreviated task name ('all') into full task names
        if opt['task'].split(':')[-2] == 'all':
            train_files = [opt['dia_train'], opt['exp_train'], opt['sen_train']]
            assert(len(opt['subtasks']) == len(train_files))
            tasks = [f'metadialog:{subtask}:{train_file}' for subtask, train_file
                in zip(opt['subtasks'], train_files)]
            opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)

class Convai2Teacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        raise NotImplementedError

class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        raise NotImplementedError
