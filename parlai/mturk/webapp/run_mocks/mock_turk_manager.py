#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import threading
import time
import uuid

from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.socket_manager import Packet
from parlai.mturk.webapp.run_mocks.mock_turk_agent import MockTurkAgent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils

parent_dir = os.path.dirname(os.path.abspath(__file__))


class MockTurkManager:
    """
    Manages interactions between MTurk agents as well as direct interactions between a
    world and the MTurk server.
    """

    current_manager = None

    def __init__(self, opt, mturk_agent_ids, is_test=False, use_db=False):
        """
        Fake an MTurk manager that has the functionality to run a task, but not on
        mturk.
        """
        self.opt = opt
        self.mturk_agent_ids = mturk_agent_ids
        self.has_run = False
        self.sandbox = True
        self.db_logger = None

    # Required lifecycle functions below
    def setup_server(self, task_directory_path=None):
        """
        Noop, we aren't connecting to a server.
        """
        print('[mock] setup_server called')

    def start_new_run(self):
        """
        Initialize expected state to not cause crashes.
        """
        self.run_id = str(int(time.time()))
        self.task_group_id = '{}_{}'.format(self.opt['task'], self.run_id)
        print('[mock] start_new_run called')

    def ready_to_accept_workers(self, timeout_seconds=None):
        """
        No threads, as there is no sustained worker pool.

        Instead we instantiate x MockTurkAgents in onboarding
        """
        self.id_to_agent = {
            agent_id: MockTurkAgent(
                self.opt,
                self,
                'hit_id_{}'.format(agent_id),
                'assignment_id_{}'.format(agent_id),
                agent_id,
            )
            for agent_id in self.mturk_agent_ids
        }
        self.agents = list(self.id_to_agent.values())
        MockTurkManager.current_manager = self
        print('[mock] ready_to_accept_workers called')

    def set_onboard_function(self, onboard_function):
        self.onboard_function = onboard_function
        print('[mock] set_onboard_function called')

    def start_task(self, eligibility_function, assign_role_function, task_function):
        """
        Handle running a task by checking to see when enough agents are in the pool to
        start an instance of the task.

        Continue doing this until the desired number of conversations is had.
        """
        print('[mock] start_task called')
        if callable(eligibility_function):
            # Convert legacy eligibility_functions to the new format
            eligibility_function = {'multiple': False, 'func': eligibility_function}
        else:
            # Ensure the eligibility function is valid
            if 'func' not in eligibility_function:
                shared_utils.print_and_log(
                    logging.CRITICAL, "eligibility_function has no 'func'. Cancelling."
                )
                raise Exception(
                    'eligibility_function dict must contain a `func` field '
                    'containing the actual function.'
                )
            elif not callable(eligibility_function['func']):
                shared_utils.print_and_log(
                    logging.CRITICAL,
                    "eligibility_function['func'] not a function. Cancelling.",
                )
                raise Exception(
                    "eligibility_function['func'] must contain a function. "
                    "If eligibility_function['multiple'] is set, it should "
                    "filter through the list of workers and only return those "
                    "that are currently eligible to participate. If it is not "
                    "set, it should take in a single worker and return whether"
                    " or not they are eligible."
                )
            if 'multiple' not in eligibility_function:
                eligibility_function['multiple'] = False

        valid_agents = [a for a in self.agents if a.mock_status == 'waiting']
        needed_agents = len(self.mturk_agent_ids)
        while len(valid_agents) < needed_agents:
            valid_agents = [a for a in self.agents if a.mock_status == 'waiting']

        # Add the required number of valid agents to the conv
        agents = [a for a in valid_agents[:needed_agents]]
        assign_role_function(agents)
        # Allow task creator to filter out agents and run
        # versions of the task that require fewer agents
        agents = [a for a in agents if a.id is not None]
        for agent in agents:
            agent.mock_status = AssignState.STATUS_IN_TASK
            agent.set_status(AssignState.STATUS_IN_TASK)
            agent.conversation_id = 'in_task'

        try:
            task_function(mturk_manager=self, opt=self.opt, workers=agents)
        except Exception as e:
            import sys
            import traceback

            print(e)
            traceback.print_exc(file=sys.stdout)
            raise e

        for agent in agents:
            agent.mock_status = AssignState.STATUS_DONE
            agent.set_status(AssignState.STATUS_DONE)
            agent.task_done = True

    def shutdown(self, force=False):
        """
        No servers, nothing to clean up.
        """
        print('[mock] shutdown called')

    def move_agents_to_waiting(self, agents):
        """
        Mock moving to a waiting world.
        """
        for agent in agents:
            agent.mock_status = AssignState.STATUS_WAITING
            agent.set_status(AssignState.STATUS_WAITING)
            agent.conversation_id = 'waiting'

    def disconnect_agent(self, worker_id, assignment_id):
        """
        Set an agent to status disconnect, and all other agents to partner disconnect.

        send them the correct message. Mocks MTurkManager._handle_agent_disconnect
        """
        worker = self.id_to_agent[worker_id]
        worker.disconnected = True
        for agent in self.agents:
            if not agent.disconnected:
                agent.some_agent_disconnected = True

    def worker_alive(self, worker_id, hit_id, assign_id):
        """
        Mocks baseline worker_alive status changes for mock agents.
        """
        agent = self.id_to_agent[worker_id]
        if agent.mock_status == AssignState.STATUS_NONE:
            agent.status = AssignState.STATUS_ONBOARDING
            agent.set_status(AssignState.STATUS_ONBOARDING)
            self.onboard_new_agent(agent)
        else:
            if agent.status in [
                AssignState.STATUS_ONBOARDING,
                AssignState.STATUS_IN_TASK,
            ]:
                pass
            elif (
                agent.status == AssignState.STATUS_DISCONNECT
                or agent.status == AssignState.STATUS_DONE
                or agent.status == AssignState.STATUS_EXPIRED
                or agent.status == AssignState.STATUS_RETURNED
                or agent.status == AssignState.STATUS_PARTNER_DISCONNECT
            ):
                # reconnect is an inactive command
                data = agent.get_inactive_command_data()
                self.send_command(worker_id, assign_id, data)

    def on_new_message(self, worker_id, msg):
        agent = self.id_to_agent[worker_id]
        agent.put_data(msg.id, msg.data)
        agent.append_message(msg.data)

    def onboard_new_agent(self, agent):
        """
        Creates an onboarding thread for the given agent.
        """
        # get state variable in question
        worker_id = agent.worker_id
        assignment_id = agent.assignment_id

        def _onboard_function(agent):
            """
            Onboarding wrapper to set state to onboarding properly.
            """
            if self.onboard_function:
                agent.id = 'Onboarding'
                self.onboard_function(agent)

            # once onboarding is done, move into a waiting world
            self.move_agents_to_waiting([agent])

        # Start the onboarding thread and run it
        onboard_thread = threading.Thread(
            target=_onboard_function,
            args=(agent,),
            name='onboard-{}-{}'.format(worker_id, assignment_id),
        )
        onboard_thread.daemon = True
        onboard_thread.start()
        return True

    # MTurk Agent Interaction Functions #

    def send_message(
        self, receiver_id, assignment_id, data, blocking=True, ack_func=None
    ):
        """
        'Send' a message directly by updating the queue of messages not yet recieved
        that the agent can pull from.
        """
        data = data.copy()  # Ensure data packet is sent in current state
        data['type'] = data_model.MESSAGE_TYPE_MESSAGE
        # Force messages to have a unique ID
        if 'message_id' not in data:
            data['message_id'] = str(uuid.uuid4())
        conversation_id = None
        agent = self.id_to_agent[receiver_id]
        conversation_id = agent.conversation_id
        event_id = shared_utils.generate_event_id(receiver_id)
        packet = Packet(
            event_id,
            Packet.TYPE_MESSAGE,
            'world',
            receiver_id,
            assignment_id,
            data,
            conversation_id=conversation_id,
            blocking=blocking,
            ack_func=ack_func,
        )

        shared_utils.print_and_log(
            logging.INFO,
            'Manager sending: {}'.format(packet),
            should_print=self.opt['verbose'],
        )
        # Push message to restore queue and incoming queue
        agent.append_message(packet.data)
        agent.unread_messages.append(packet)
        return data['message_id']

    def send_command(
        self, receiver_id, assignment_id, data, blocking=True, ack_func=None
    ):
        """
        Commands aren't actually sent this way, as state updates are read.
        """
        return None

    def timeout_all_agents(self):
        """
        Set all agent statuses to disconnect to kill the world.
        """
        for agent in self.agents:
            agent.disconnected = True

    # BELOW ARE STUBS THAT EXIST TO HOPEFULLY MAKE RUN FILES NOT CRASH
    # NONE OF THEM DO ANYTHING (though some return success values)

    def mark_workers_done(self, workers):
        pass

    def free_workers(self, workers):
        pass

    def get_agent_work_status(self, assignment_id):
        pass

    def get_qualification_list(self, qualifications=None):
        return []

    def create_additional_hits(self, num_hits, qualifications=None):
        return 'fake_page_url'

    def create_hits(self, qualifications=None):
        return 'fake_page_url'

    def get_hit(self, hit_id):
        pass

    def get_assignment(self, assignment_id):
        pass

    def get_assignments_for_hit(self, hit_id):
        pass

    def expire_all_unassigned_hits(self):
        pass

    def approve_work(self, assignment_id, override_rejection=False):
        print('[mock] Assignment {} approved'.format(assignment_id))

    def reject_work(self, assignment_id, reason):
        print('[mock] Assignment {} rejected for {}'.format(assignment_id, reason))

    def approve_assignments_for_hit(self, hit_id, override_rejection=False):
        print('[mock] HIT {} approved'.format(hit_id))

    def block_worker(self, worker_id, reason):
        print('[mock] Worker {} blocked for reason {}'.format(worker_id, reason))

    def soft_block_worker(self, worker_id, qual='block_qualification'):
        print('[mock] Worker {} given qual {}'.format(worker_id, qual))

    def un_soft_block_worker(self, worker_id, qual='block_qualification'):
        print('[mock] Worker {} revoked qual {}'.format(worker_id, qual))

    def give_worker_qualification(self, worker_id, qual_name, qual_value=None):
        print('[mock] Worker {} given qual {}'.format(worker_id, qual_name))

    def remove_worker_qualification(self, worker_id, qual_name, reason=''):
        print('[mock] Worker {} revoked qual {}'.format(worker_id, qual_name))

    def create_qualification(self, qualification_name, description, can_exist=True):
        pass

    def pay_bonus(
        self, worker_id, bonus_amount, assignment_id, reason, unique_request_token
    ):
        print('[mock] Worker {} paid bonus {}'.format(worker_id, bonus_amount))

    def email_worker(self, worker_id, subject, message_text):
        print('[mock] Worker {} emailed {}'.format(worker_id, message_text))
        return {'success': True}
