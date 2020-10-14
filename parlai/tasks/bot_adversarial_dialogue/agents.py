#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from parlai.core.opt import Opt
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.tasks.bot_adversarial_dialogue.build import (
    build_dialogue_datasets,
    build_human_safety_eval_dataset,
    get_adversarial_dialogue_folder,
    get_human_safety_eval_folder,
)
import parlai.utils.logging as logging
from parlai.utils.io import PathManager
from parlai.utils.misc import str_to_msg
from parlai.tasks.dialogue_safety.agents import OK_CLASS, NOT_OK_CLASS

# Constants
SAFETY_DICT = {'safe': OK_CLASS, 'unsafe': NOT_OK_CLASS}


def _adversarial_dialogue_datapath(opt: Opt) -> str:
    """
    Return the filepath for the specified datatype of the specified adversarial dialogue
    task.
    """
    build_dialogue_datasets(opt)
    # Build the data if it doesn't exist.
    dt = opt['datatype'].split(':')[0]
    data_path = os.path.join(
        get_adversarial_dialogue_folder(opt['datapath']),
        'bot_adversarial_dialogue_datasets',
        dt + '.txt',
    )
    return data_path


class BotAdversarialDialogueTeacher(ParlAIDialogTeacher):
    """
    Teacher for bot_adversarial_dialogue.
    """

    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Bot Adversarial Dialogue options')
        agent.add_argument('--bad-num-turns', type=int, default=-1)
        agent.add_argument(
            '--bad-speaker-to-eval',
            type=str,
            default='all',
            choices=['bot', 'human', 'all'],
        )
        agent.add_argument(
            '--bad-safety-mix',
            type=str,
            default='all',
            choices=['safe', 'unsafe', 'all'],
        )

    def __init__(self, opt, shared=None):
        opt['parlaidialogteacher_datafile'] = _adversarial_dialogue_datapath(opt=opt)
        super().__init__(opt, shared=shared)
        self.id = '{}.numTurns_{}_speakerToEval_{}_safetyMix_{}'.format(
            self.id,
            self.opt['bad_num_turns'],
            self.opt['bad_speaker_to_eval'],
            self.opt['bad_safety_mix'],
        )

    def _setup_data(self, path):
        logging.info(f"Loading ParlAI text data: {path}")

        self.episodes = []
        self.num_exs = 0
        eps = []
        with PathManager.open(path, newline='\n', encoding='utf-8') as read:
            for line_no, line in enumerate(read, 1):
                msg = str_to_msg(line.rstrip('\n'))
                if msg and 'eval_labels' in msg:
                    raise ValueError(
                        f"It looks like you've written eval_labels as a key in your "
                        f"data file. This is not appropriate; labels will be converted "
                        f"for you automatically. This is happening on Line {line_no} "
                        f"in {path}. The line is:\n\t{line}"
                    )
                if msg and 'text' not in msg:
                    raise ValueError(
                        f'ParlaiDialogTeacher requires a "text" field in every '
                        f'entry, but one is missing in Line {line_no} in {path}. '
                        f'The line is:\n\t{line}'
                    )
                if msg and 'labels' not in msg:
                    raise ValueError(
                        f'ParlaiDialogTeacher requires a "labels" field in every '
                        f'entry, but one is missing in Line {line_no} in {path}. '
                        f'The line is:\n\t{line}'
                    )

                if (
                    self.opt['bad_speaker_to_eval'] != 'all'
                    and self.opt['bad_speaker_to_eval'] != msg['speaker_to_eval']
                ):
                    continue
                if (
                    self.opt['bad_safety_mix'] != 'all'
                    and SAFETY_DICT[self.opt['bad_safety_mix']] != msg['labels'][0]
                ):
                    continue
                if self.opt['bad_num_turns'] > 0:
                    dialog = msg['text'].split('\n')
                    msg.force_set(
                        'text', '\n'.join(dialog[-self.opt['bad_num_turns'] :])
                    )
                if msg:
                    self.num_exs += 1
                    eps.append(msg)
                    if msg.get('episode_done', False):
                        self.episodes.append(eps)
                        eps = []
        if len(eps) > 0:
            # add last episode
            eps[-1].force_set('episode_done', True)
            self.episodes.append(eps)
        if len(self.episodes) == 1 and line_no > 100:
            logging.error(
                f'The data in {path} looks like one very long episode. If this '
                f'is intentional, you may ignore this, but you MAY have a bug in '
                f'your data.'
            )


def _human_safety_eval_datapath(opt: Opt) -> str:
    """
    Return the filepath for the specified datatype of the specified human evaluation
    task on bot adversarial dialogue.
    """
    build_human_safety_eval_dataset(opt)
    # Build the data if it doesn't exist.
    logging.info(
        f'The data for human safety evaluation is test set only '
        f'regardless of your chosen datatype, which is {opt["datatype"]} '
    )
    data_path = os.path.join(
        get_human_safety_eval_folder(opt['datapath']), 'human_safety_eval', 'test.txt'
    )
    return data_path


class HumanSafetyEvaluationTeacher(ParlAIDialogTeacher):
    """
    Teacher for human safety evaluation on bot adversarial dialogues.
    """

    def __init__(self, opt, shared=None):
        opt['parlaidialogteacher_datafile'] = _human_safety_eval_datapath(opt=opt)
        super().__init__(opt, shared=shared)


class DefaultTeacher(BotAdversarialDialogueTeacher):
    pass
