#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
# Accessing the tasks can be done with something like:
#
#   python examples/train_model.py --setting 'RBI' -m "projects.memnn_feedback.agent.memnn_feedback:MemnnFeedbackAgent" 
# -t "projects.memnn_feedback.tasks.dialog_babi_feedback.agents:taskTeacher:1_p0.5:feedback"
#
# which specifies task 1, and policy with 0.5 answers correct with reward-based learning, see the papers 
# for more details: https://arxiv.org/abs/1604.06045 and https://arxiv.org/abs/1605.07683

from parlai.core.teachers import FbDialogTeacher
from .build import build

import os

tasks = {}
tasks[1] = 'rl1_API_calls_with_ans'
tasks[2] = 'rl2_API_refine_with_ans'
tasks[3] = 'rl3_options_with_ans'
tasks[4] = 'rl4_phone_address_with_ans'
tasks[5] = 'rl5_full_dialogs_with_ans'

def _path(task, opt):
    # Build the data if it doesn't exist.
    build(opt)
    task_name = '%s_%s' % (task.split('_')[1],
                           tasks[int(task.split('_')[0])])
    task = task.split('_')[0]
    task_name = 'dialog-babi_' + task_name
    prefix = os.path.join(opt['datapath'], 'dialog-bAbI-feedback', 'dialog-bAbI-feedback')
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'trn'
    elif dt == 'test':
        suffix = 'tst'
    elif dt == 'valid':
        suffix = 'dev'
    datafile = os.path.join(prefix,
            '{task}_{type}.txt'.format(task=task_name, type=suffix))

    cands_datafile = os.path.join(prefix, 'dialog-babi-candidates.txt')
    return datafile, cands_datafile

# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'dialog-bAbI-feedback',
                                       'dialog-bAbI-feedback-tasks',
                                       'dialog-babi-kb-all.txt')
        super().__init__(opt, shared)


# Single task.
class TaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        paths = _path(opt['task'].split(':')[2], opt)
        opt['datafile'], opt['cands_datafile'] = paths
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
