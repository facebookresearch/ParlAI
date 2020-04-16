#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Convert a ParlAI teacher to acute-eval format.
Examples
--------
.. code-block:: shell
py parlai/mturk/tasks/acute_eval/dump_task_to_acute_format.py  -t  convai2
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.conversations import Conversations
from parlai.utils.misc import TimeLogger
import random
import tempfile


def setup_args():
    """
    Set up conversion args.
    """
    parser = ParlaiParser()
    parser.add_argument(
        '-n',
        '--num-episodes',
        default=-1,
        type=int,
        help='Total number of episodes to convert, -1 to convert \
                                all examples',
    )
    parser.add_argument(
        '-of',
        '--outfile',
        default=None,
        type=str,
        help='Output file where to save, by default will be \
                                created in /tmp',
    )
    parser.add_argument(
        '-s1id', '--speaker-0-id', type=str, help='Speaker id of agent who speaks first'
    )
    parser.add_argument(
        '-s1id',
        '--speaker-1-id',
        type=str,
        help='Speaker id of agent who speaks second',
    )
    parser.add_argument(
        '--prepended-context',
        type='bool',
        default=False,
        help='specify if the context is prepended to the first act',
    )
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=10)
    parser.set_defaults(datatype='train:ordered')

    return parser


def dump_data(opt):
    """
    Dump task data to ACUTE-Eval.
    """
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    task = opt.get('task')
    speaker_0_id = opt.get('speaker_0_id') or f'{task}_as_human'
    speaker_1_id = opt.get('speaker_1_id') or f'{task}_as_model'
    if opt['outfile'] is None:
        outfile = tempfile.mkstemp(
            prefix='{}_{}_'.format(opt['task'], opt['datatype']), suffix='.txt'
        )[1]
    else:
        outfile = opt['outfile']

    num_episodes = (
        world.num_episodes()
        if opt['num_episodes'] == -1
        else min(opt['num_episodes'], world.num_episodes())
    )
    log_timer = TimeLogger()

    print(f'[ starting to convert, saving output to {outfile} ]')
    dialogues = []
    for _ in range(num_episodes):
        episode = []
        episode_done = False
        while not episode_done:
            world.parley()
            acts = world.get_acts()
            text = acts[0].get('text')
            split_text = text.split('\n')
            label = random.choice(
                acts[0].get('labels', acts[0].pop('eval_labels', None))
            )
            if not episode and opt.get('prepended_context'):
                # first turn
                context = split_text[:-1]
                text = split_text[-1]
                context_turn = [
                    {'text': context, 'episode_done': False, 'id': 'context'}
                    for _ in range(2)
                ]
                episode.append(context_turn)
            turn = [
                {'text': text, 'episode_done': False, 'id': speaker_0_id},
                {'text': label, 'episode_done': False, 'id': speaker_1_id},
            ]
            episode.append(turn)
            if acts[0].get('episode_done', False):
                episode[-1][-1]['episode_done'] = True
                episode_done = True
                dialogues.append(episode)

            if log_timer.time() > opt['log_every_n_secs']:
                text, _log = log_timer.log(world.total_parleys, world.num_examples())
                print(text)

        if world.epoch_done():
            break

    Conversations.save_conversations(dialogues, outfile, opt)


def main():
    random.seed(42)
    # Get command line arguments
    parser = setup_args()
    opt = parser.parse_args()
    dump_data(opt)


if __name__ == '__main__':
    main()
