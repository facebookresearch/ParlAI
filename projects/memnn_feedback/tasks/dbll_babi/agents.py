#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Accessing the tasks can be done with something like:
#
#   python examples/display_data.py -t "projects.memnn_feedback.tasks.dbll_babi.agents:taskTeacher:3_p0.5:feedback"
#
# which specifies task 2, and policy with 0.5 answers correct, see the paper
# for more details: https://arxiv.org/abs/1604.06045

from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os

tasks = {}
tasks[1] = 'rl1_pure_imitation'
tasks[2] = 'rl2_pos_neg'
tasks[3] = 'rl3_with_ans'
tasks[4] = 'rl4_with_hints'
tasks[5] = 'rl5_told_sf'
tasks[6] = 'rl6_only_some_rewards'
tasks[7] = 'rl7_no_feedback'
tasks[8] = 'rl8_imitation_plus_rl'
tasks[9] = 'rl9_ask_for_answer'
tasks[10] = 'rl10_ask_for_sf'

_suffixes = {
    'train': 'train',
    'test': 'test',
    'valid': 'dev'
}


def _path(subdir, task, opt, dt=''):
    build(opt)
    if dt == '':
        dt = opt['datatype'].split(':')[0]
    task_name = '%s_%s' % (task.split('_')[1],
                           tasks[int(task.split('_')[0])])
    return os.path.join(opt['datapath'], 'DBLL', 'dbll',
                        '{subdir}_{task}_{suffix}.txt'.format(
                            subdir=subdir, task=task_name,
                            suffix=_suffixes[dt]))


class TaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        params = opt['task'].split(':')[2]
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(os.path.join('babi', 'babi1'), params, opt)
        opt['cands_datafile'] = _path(os.path.join('babi', 'babi1'), params,
                                      opt, 'train')
        self.opt = opt
        super().__init__(opt, shared)

    def setup_data(self, path):
        """ Reads feedback for an example along with text and labels
        if 'feedback' argument is specified 
        """
        if self.opt['task'].split(':')[-1] == 'feedback':
            return self.setup_data_with_feedback(path)
        else:
            return super().setup_data(path)

    def setup_data_with_feedback(self, path):
        """Reads data in the fbdialog format.
        This method is very similar to FbDialogTeacher.setup_data(..). 
        The difference is that in this method the feedback is appended to the query 
        from the current example; in the default setup the feedback is appended to 
        the x from the next example.

        The data would look something like this:
        
        Mary moved to the bedroom.
        Mary travelled to the garden.
        Where is John?
        No, that's wrong.
        [labels: garden]
        
        To append feedback to the current example, modify the task name like this:
          python examples/display_data.py -t dbll_babi:task:2_p0.5:f 
        Default setup: 
          python examples/display_data.py -t dbll_babi:task:2_p0.5 

        """
        print("[loading fbdialog data:" + path + "]")
        with open(path) as read:
            start = True
            x = ''

            y = None

            reward = 0
            dialog_index = 0
            read_feedback = False
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
                    read_feedback = True
                    # split labels
                    y = split[1].split('|')

                if read_feedback and not split[1]:
                    split[0] = x
                    split[1] = y
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
                    y = None
                    read_feedback = False

# Defaults to task 2 with p=0.5.
class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = '2_p0.5'
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(os.path.join('babi', 'babi1'), task, opt)
        opt['cands_datafile'] = _path(os.path.join('babi', 'babi1'), task,
                                      opt, 'train')
        super().__init__(opt, shared)
