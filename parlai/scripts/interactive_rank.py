#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Does human evaluation on a task with label_candidates.

Human can exit with ctrl + c and metrics will be computed and displayed.

Examples
--------

.. code-block:: shell

  python examples/interactive_rank.py -t babi:task10k:1 -dt valid

When prompted, enter the index of the label_candidate you think is correct.
Candidates are shuffled for each example.
During datatype train, examples are randomly sampled with replacement; use
train:ordered to not repeat examples.
During datatype valid or test, examples are shown in order, not shuffled.
"""
from parlai.core.metrics import Metrics
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent, create_task_agent_from_taskname

import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser()
    parser.add_pytorch_datateacher_args()
    parser.set_params(model='parlai.agents.local_human.local_human:LocalHumanAgent')
    return parser


def interactive_rank(opt, print_parser=None):
    # Create model and assign it to the specified task
    human = create_agent(opt)
    task = create_task_agent_from_taskname(opt)[0]

    metrics = Metrics(opt)
    episodes = 0

    def print_metrics():
        report = metrics.report()
        report['episodes'] = episodes
        print(report)

    # Show some example dialogs:
    try:
        while not task.epoch_done():
            msg = task.act()
            print('[{id}]: {text}'.format(id=task.getID(), text=msg.get('text', '')))
            cands = list(msg.get('label_candidates', []))
            random.shuffle(cands)
            for i, c in enumerate(cands):
                print('    [{i}]: {c}'.format(i=i, c=c))

            print('[ Please choose a response from the list. ]')

            choice = None
            while choice is None:
                choice = human.act().get('text')
                try:
                    choice = int(choice)
                    if choice >= 0 and choice < len(cands):
                        choice = cands[choice]
                    else:
                        print('[ Try again: you selected {i} but the '
                              'candidates are indexed from 0 to {j}. ]'
                              ''.format(i=choice, j=len(cands) - 1))
                        choice = None
                except (TypeError, ValueError):
                    print('[ Try again: you did not enter a valid index. ]')
                    choice = None

            print('[ You chose ]: {}'.format(choice))
            reply = {'text_candidates': [choice]}
            labels = msg.get('eval_labels', msg.get('labels'))
            metrics.update(reply, labels)
            if msg.get('episode_done'):
                episodes += 1
            print_metrics()
            print('------------------------------')
            print('[ True reply ]: {}'.format(labels[0]))
            if msg.get('episode_done'):
                print('******************************')

    except KeyboardInterrupt:
        pass

    print()
    print_metrics()


if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    interactive_rank(parser.parse_args(print_args=False))
