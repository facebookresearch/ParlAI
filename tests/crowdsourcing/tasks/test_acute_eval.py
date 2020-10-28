#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the ACUTE-Eval crowdsourcing task.
"""

import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List

import hydra
from mephisto.core.hydra_config import register_script_config
from mephisto.core.local_database import LocalMephistoDB
from mephisto.core.operator import Operator
from mephisto.utils.scripts import augment_config_from_db
from omegaconf import DictConfig, OmegaConf

from parlai.crowdsourcing.tasks.acute_eval.run import (
    ScriptConfig,
    defaults,
    TASK_DIRECTORY,
)


test_defaults = defaults + [
    {'mephisto/architect': 'mock'},
    {'mephisto/provider': 'mock'},
]

relative_task_directory = os.path.relpath(TASK_DIRECTORY, os.path.dirname(__file__))


# Params
DESIRED_INPUTS = [
    {
        'task_specs': {
            's1_choice': 'I would prefer to talk to <Speaker 1>',
            's2_choice': 'I would prefer to talk to <Speaker 2>',
            'question': 'Who would you prefer to talk to for a long conversation?',
            'is_onboarding': True,
            'model_left': {
                'name': 'modela',
                'dialogue': [
                    {'id': 'modela', 'text': 'Hello how are you?'},
                    {'id': 'human_evaluator', 'text': "I'm well, how about yourself?"},
                    {"id": "modela", "text": "Good, just reading a book."},
                    {"id": "human_evaluator", "text": "What book are you reading?"},
                    {
                        'id': 'modela',
                        'text': 'An English textbook. Do you like to read?',
                    },
                    {
                        'id': 'human_evaluator',
                        'text': 'Yes, I really enjoy reading, but my favorite thing to do is dog walking.',
                    },
                    {
                        'id': 'modela',
                        'text': "Do you have a dog? I don't have any pets",
                    },
                    {
                        'id': 'human_evaluator',
                        'text': 'Yes, I have a labrador poodle mix.',
                    },
                ],
            },
            'model_right': {
                'name': 'modelc',
                'dialogue': [
                    {'id': 'modelc', 'text': 'Hello hello hello'},
                    {'id': 'human_evaluator', 'text': 'How are you?'},
                    {'id': 'modelc', 'text': 'Hello hello hello'},
                    {'id': 'human_evaluator', 'text': 'Hello back'},
                    {'id': 'modelc', 'text': 'Hello hello hello'},
                    {'id': 'human_evaluator', 'text': 'You must really like that word'},
                    {'id': 'modelc', 'text': 'Hello hello hello'},
                    {'id': 'human_evaluator', 'text': 'Ok'},
                ],
            },
        },
        'pairing_dict': {
            'is_onboarding': True,
            'speakers_to_eval': ['modela', 'modelc'],
            'correct_answer': 'modela',
            'tags': ['onboarding1'],
            'dialogue_dicts': [
                {
                    'speakers': ['modela', 'human_evaluator'],
                    'id': 'ABCDEF',
                    'evaluator_id_hashed': 'HUMAN1',
                    'oz_id_hashed': None,
                    'dialogue': [
                        {'id': 'modela', 'text': 'Hello how are you?'},
                        {
                            'id': 'human_evaluator',
                            'text': "I'm well, how about yourself?",
                        },
                        {'id': 'modela', 'text': 'Good, just reading a book.'},
                        {'id': 'human_evaluator', 'text': 'What book are you reading?'},
                        {
                            'id': 'modela',
                            'text': 'An English textbook. Do you like to read?',
                        },
                        {
                            'id': 'human_evaluator',
                            'text': 'Yes, I really enjoy reading, but my favorite thing to do is dog walking.',
                        },
                        {
                            'id': 'modela',
                            'text': "Do you have a dog? I don't have any pets",
                        },
                        {
                            'id': 'human_evaluator',
                            'text': 'Yes, I have a labrador poodle mix.',
                        },
                    ],
                },
                {
                    'speakers': ['modelc', 'human_evaluator'],
                    'id': 'ZYX',
                    'evaluator_id_hashed': 'HUMAN3',
                    'oz_id_hashed': None,
                    'dialogue': [
                        {'id': 'modelc', 'text': 'Hello hello hello'},
                        {'id': 'human_evaluator', 'text': 'How are you?'},
                        {'id': 'modelc', 'text': 'Hello hello hello'},
                        {'id': 'human_evaluator', 'text': 'Hello back'},
                        {'id': 'modelc', 'text': 'Hello hello hello'},
                        {
                            'id': 'human_evaluator',
                            'text': 'You must really like that word',
                        },
                        {'id': 'modelc', 'text': 'Hello hello hello'},
                        {'id': 'human_evaluator', 'text': 'Ok'},
                    ],
                },
            ],
        },
        'pair_id': 0,
    },
    {
        'task_specs': {
            's1_choice': 'I would prefer to talk to <Speaker 1>',
            's2_choice': 'I would prefer to talk to <Speaker 2>',
            'question': 'Who would you prefer to talk to for a long conversation?',
            'is_onboarding': False,
            'model_left': {
                'name': 'modelb',
                'dialogue': [
                    {
                        'id': 'human_evaluator',
                        'text': 'Hi, I love food, what about you?',
                    },
                    {
                        'id': 'modelb',
                        'text': "I love food too, what's your favorite? Mine is burgers.",
                    },
                    {
                        'id': 'human_evaluator',
                        'text': "I'm a chef and I love all foods. What do you do?",
                    },
                    {'id': 'modelb', 'text': "I'm retired now, but I was a nurse."},
                    {
                        'id': 'human_evaluator',
                        'text': "Wow, that's really admirable. My sister is a nurse.",
                    },
                    {'id': 'modelb', 'text': 'Do you have any hobbies?'},
                    {'id': 'human_evaluator', 'text': 'I like to paint and play piano'},
                    {
                        'id': 'modelb',
                        'text': "You're very artistic. I wish I could be so creative.",
                    },
                ],
            },
            'model_right': {
                'name': 'modela',
                'dialogue': [
                    {'id': 'modela', 'text': 'Hi how are you doing?'},
                    {'id': 'human_evaluator', 'text': "I'm doing ok."},
                    {'id': 'modela', 'text': "Oh, what's wrong?"},
                    {
                        'id': 'human_evaluator',
                        'text': 'Feeling a bit sick after my workout',
                    },
                    {'id': 'modela', 'text': 'Do you workout a lot?'},
                    {
                        'id': 'human_evaluator',
                        'text': 'Yes, I go to the gym every day. I do a lot of lifting.',
                    },
                    {'id': 'modela', 'text': "That's cool, I like to climb."},
                    {'id': 'human_evaluator', 'text': "I've never been."},
                ],
            },
        },
        'pairing_dict': {
            'is_onboarding': False,
            'speakers_to_eval': ['modelb', 'modela'],
            'tags': ['example1'],
            'dialogue_ids': [0, 1],
            'dialogue_dicts': [
                {
                    'speakers': ['modelb', 'human_evaluator'],
                    'id': 'AGHIJK',
                    'evaluator_id_hashed': 'HUMAN2',
                    'oz_id_hashed': None,
                    'dialogue': [
                        {
                            'id': 'human_evaluator',
                            'text': 'Hi, I love food, what about you?',
                        },
                        {
                            'id': 'modelb',
                            'text': "I love food too, what's your favorite? Mine is burgers.",
                        },
                        {
                            'id': 'human_evaluator',
                            'text': "I'm a chef and I love all foods. What do you do?",
                        },
                        {'id': 'modelb', 'text': "I'm retired now, but I was a nurse."},
                        {
                            'id': 'human_evaluator',
                            'text': "Wow, that's really admirable. My sister is a nurse.",
                        },
                        {'id': 'modelb', 'text': 'Do you have any hobbies?'},
                        {
                            'id': 'human_evaluator',
                            'text': 'I like to paint and play piano',
                        },
                        {
                            'id': 'modelb',
                            'text': "You're very artistic. I wish I could be so creative.",
                        },
                    ],
                },
                {
                    'speakers': ['modela', 'human_evaluator'],
                    'id': '123456',
                    'evaluator_id_hashed': 'HUMAN1',
                    'oz_id_hashed': None,
                    'dialogue': [
                        {'id': 'modela', 'text': 'Hi how are you doing?'},
                        {'id': 'human_evaluator', 'text': "I'm doing ok."},
                        {'id': 'modela', 'text': "Oh, what's wrong?"},
                        {
                            'id': 'human_evaluator',
                            'text': 'Feeling a bit sick after my workout',
                        },
                        {'id': 'modela', 'text': 'Do you workout a lot?'},
                        {
                            'id': 'human_evaluator',
                            'text': 'Yes, I go to the gym every day. I do a lot of lifting.',
                        },
                        {'id': 'modela', 'text': "That's cool, I like to climb."},
                        {'id': 'human_evaluator', 'text': "I've never been."},
                    ],
                },
            ],
        },
        'pair_id': 1,
    },
]
DESIRED_OUTPUTS = {
    "final_data": [
        {"speakerChoice": "modela", "textReason": "Turn 1"},
        {"speakerChoice": "modelb", "textReason": "Turn 2"},
        {"speakerChoice": "modelb", "textReason": "Turn 3"},
        {"speakerChoice": "modelb", "textReason": "Turn 4"},
        {"speakerChoice": "modelb", "textReason": "Turn 5"},
    ]
}


@dataclass
class TestScriptConfig(ScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: test_defaults)


@hydra.main(config_path=relative_task_directory, config_name="test_script_config")
def main(cfg: DictConfig) -> Dict[str, Any]:

    # Set up the mock server
    data_dir = tempfile.mkdtemp()
    database_path = os.path.join(data_dir, "mephisto.db")
    db = LocalMephistoDB(database_path)
    cfg = augment_config_from_db(cfg, db)
    cfg.mephisto.architect.should_run_server = True
    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
    sup = operator.supervisor
    assert len(sup.channels) == 1
    channel = list(sup.channels.keys())[0]
    server = sup.channels[channel].job.architect.server
    task_runner = sup.channels[channel].job.task_runner
    print(OmegaConf.to_yaml(cfg))
    # TODO: remove print statement

    # Create a mock worker agent
    mock_worker_name = "MOCK_WORKER"
    server.register_mock_worker(mock_worker_name)
    workers = db.find_workers(worker_name=mock_worker_name)
    worker_id = workers[0].db_id
    mock_agent_details = "FAKE_ASSIGNMENT"
    server.register_mock_agent(worker_id, mock_agent_details)
    agent_1 = db.find_agents()[0]
    agent_id_1 = agent_1.db_id

    # print("SENDING_THREAD: ", sup.sending_thread)
    # print('RECEIVED MESSAGES: ', server.received_messages)
    # TODO: remove block

    # Set initial data
    _ = task_runner.get_init_data_for_agent(agent_1)
    import pdb

    pdb.set_trace()  # TODO: remove

    # Send response
    server.send_agent_act(
        agent_id_1, {'MEPHISTO_is_submit': True, 'task_data': DESIRED_OUTPUTS}
    )

    # Give up to 1 seconds for the actual operations to occur
    # start_time = time.time()
    TIMEOUT_TIME = 1
    # while time.time() - start_time < TIMEOUT_TIME:
    #     if len(agent_1_data["acts"]) > 0:
    #         break
    #     time.sleep(0.1)
    import time

    time.sleep(TIMEOUT_TIME)
    # TODO: clean up this

    import pdb

    pdb.set_trace()  # TODO: remove

    # sup.shutdown()  # TODO: we still want this, right?

    return agent_1.state.state


class TestAcuteEval(unittest.TestCase):
    """
    Test the ACUTE-Eval crowdsourcing task.
    """

    def test_base_task(self):
        register_script_config(name='test_script_config', module=TestScriptConfig)
        agent_state = main()
        self.assertEqual(DESIRED_INPUTS, agent_state['inputs'])
        self.assertEqual(DESIRED_OUTPUTS, agent_state['outputs'])


if __name__ == "__main__":
    unittest.main()
