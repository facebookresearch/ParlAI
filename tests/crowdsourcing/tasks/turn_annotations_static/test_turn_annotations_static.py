#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the chat demo crowdsourcing task.
"""

import os
import unittest


SAMPLE_CONVERSATIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'task_config',
    'sample_conversations.jsonl',
)


try:

    from parlai.crowdsourcing.tasks.turn_annotations_static.run import TASK_DIRECTORY
    from parlai.crowdsourcing.tasks.turn_annotations_static.run_in_flight_qa import (
        TASK_DIRECTORY as TASK_DIRECTORY_IN_FLIGHT_QA,
    )
    from parlai.crowdsourcing.tasks.turn_annotations_static.turn_annotations_blueprint import (
        STATIC_BLUEPRINT_TYPE,
        STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.utils.tests import CrowdsourcingTestMixin

    class TestTurnAnnotationsStatic(CrowdsourcingTestMixin, unittest.TestCase):
        """
        Test the turn annotations crowdsourcing tasks.
        """

        def test_no_in_flight_qa(self):

            # # Setup

            # Set up the config and database
            overrides = [
                '+mephisto.blueprint.annotation_indices_jsonl=null',
                '+mephisto.blueprint.conversation_count=null',
                f'mephisto.blueprint.data_jsonl={SAMPLE_CONVERSATIONS_PATH}',
                'mephisto.blueprint.onboarding_qualification=null',
                '+mephisto.blueprint.random_seed=42',
            ]
            # TODO: remove all of these params once Hydra 1.1 is released with support
            #  for recursive defaults
            # TODO: test onboarding as well, and don't nullify the onboarding_qualification param
            self._set_up_config(
                blueprint_type=STATIC_BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )

            # Set up the operator and server
            self._set_up_server()

            # Set up the mock human agents
            agent_id = self._register_mock_agents(num_agents=1)[0]

            # # Feed messages to the agent

            # Set initial data
            self.server.request_init_data(agent_id)

            state = self.db.find_agents()[0].state.get_data()
            print('foooooooooo')
            print(state)
            # TODO: remove block

            import pdb

            pdb.set_trace()
            # TODO: remove
            # TODO: revise below
            # Have agents talk to each other
            for agent_0_text, agent_1_text in AGENT_MESSAGES:
                self._send_agent_message(
                    agent_id=agent_0_id,
                    agent_display_id=AGENT_0_DISPLAY_ID,
                    text=agent_0_text,
                )
                self._send_agent_message(
                    agent_id=agent_1_id,
                    agent_display_id=AGENT_1_DISPLAY_ID,
                    text=agent_1_text,
                )

            # Have agents fill out the form
            self.server.send_agent_act(
                agent_id=agent_0_id,
                act_content={
                    'text': FORM_PROMPTS['agent_0'],
                    'task_data': {'form_responses': FORM_RESPONSES['agent_0']},
                    'id': AGENT_0_DISPLAY_ID,
                    'episode_done': False,
                },
            )
            self.server.send_agent_act(
                agent_id=agent_1_id,
                act_content={
                    'text': FORM_PROMPTS['agent_1'],
                    'task_data': {'form_responses': FORM_RESPONSES['agent_1']},
                    'id': AGENT_1_DISPLAY_ID,
                    'episode_done': False,
                },
            )

            # Submit the HIT
            self.server.send_agent_act(
                agent_id=agent_0_id,
                act_content={
                    'task_data': {'final_data': {}},
                    'MEPHISTO_is_submit': True,
                },
            )
            self.server.send_agent_act(
                agent_id=agent_1_id,
                act_content={
                    'task_data': {'final_data': {}},
                    'MEPHISTO_is_submit': True,
                },
            )

            # # Check that the inputs and outputs are as expected

            state_0 = self.db.find_agents()[0].state.get_data()
            actual_and_desired_states = [
                (state_0, DESIRED_STATE_AGENT_0),
                (state_1, DESIRED_STATE_AGENT_1),
            ]
            for actual_state, desired_state in actual_and_desired_states:
                assert actual_state['inputs'] == desired_state['inputs']
                assert len(actual_state['outputs']['messages']) == len(
                    desired_state['outputs']['messages']
                )
                for actual_message, desired_message in zip(
                    actual_state['outputs']['messages'],
                    desired_state['outputs']['messages'],
                ):
                    for key, desired_value in desired_message.items():
                        if key == 'timestamp':
                            pass  # The timestamp will obviously be different
                        elif key == 'data':
                            for key_inner, desired_value_inner in desired_message[
                                key
                            ].items():
                                if key_inner == 'message_id':
                                    pass  # The message ID will be different
                                else:
                                    self.assertEqual(
                                        actual_message[key][key_inner],
                                        desired_value_inner,
                                    )
                        else:
                            self.assertEqual(actual_message[key], desired_value)


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
