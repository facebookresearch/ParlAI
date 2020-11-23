#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the chat demo crowdsourcing task.
"""

import unittest

# Desired inputs/outputs
# {{{TODO: inputs/outputs}}}
# TODO: move this all to a YAML file given the upcoming pytest regressions framework


try:

    import parlai.crowdsourcing.tasks.turn_annotations.worlds as world_module
    from parlai.crowdsourcing.tasks.turn_annotations.run import TASK_DIRECTORY
    from parlai.crowdsourcing.tasks.turn_annotations.turn_annotations_blueprint import (
        SharedTurnAnnotationsTaskState,
        BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.utils.tests import AbstractParlAIChatTest

    class TestChatDemo(AbstractParlAIChatTest):
        """
        Test the chat demo crowdsourcing task.
        """

        def test_base_task(self):

            # # Setup

            # Set up the config and database
            # {{{TODO: define overrides}}}
            # TODO: remove all of these params once Hydra 1.1 is released with support
            #  for recursive defaults
            self._set_up_config(
                blueprint_type=BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )

            # Set up the operator and server
            shared_state = SharedTurnAnnotationsTaskState(world_module=world_module)
            self._set_up_server(shared_state=shared_state)

            # Check that the agent states are as they should be
            self._test_agent_states(
                agent_display_ids=AGENT_DISPLAY_IDS,
                agent_messages=AGENT_MESSAGES,
                form_prompts=FORM_PROMPTS,
                form_responses=FORM_RESPONSES,
                expected_states=EXPECTED_STATES,
            )

            # Check that the contents of the chat data file are as expected
            # {{{TODO}}}


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
