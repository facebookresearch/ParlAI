#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pickle
import time

from botocore.exceptions import ClientError

from parlai.mturk.core.agents import MTurkAgent, AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.mturk_utils as mturk_utils
import parlai.mturk.core.shared_utils as shared_utils


# Time to persist a disconnect before forgetting about it. Combined with the
# above this will block workers that disconnect at least 10 times in a day
DISCONNECT_PERSIST_LENGTH = 60 * 60 * 24

# Max number of conversation disconnects before a turker should be blocked
MAX_DISCONNECTS = 5

DISCONNECT_FILE_NAME = 'disconnects.pickle'

parent_dir = os.path.dirname(os.path.abspath(__file__))


class WorkerState():
    """Class for holding state information about an mturk worker"""
    def __init__(self, worker_id, disconnects=0):
        """Create a new worker state for the given worker_id. Number of
        prior disconnects is optional.
        """
        self.worker_id = worker_id
        self.agents = {}
        self.disconnects = disconnects

    def active_conversation_count(self):
        """Return the number of conversations within this worker state
        that aren't in a final state
        """
        count = 0
        for assign_id in self.agents:
            if not self.agents[assign_id].is_final():
                count += 1
        return count

    def add_agent(self, mturk_agent):
        """Add an assignment to this worker state with the given assign_it"""
        assert mturk_agent.worker_id == self.worker_id, \
            "Can't add agent that does not match state's worker_id"
        self.agents[mturk_agent.assignment_id] = mturk_agent

    def get_agent_for_assignment(self, assignment_id):
        return self.agents.get(assignment_id, None)

    def has_assignment(self, assign_id):
        """Returns true if this worker has an assignment for the given id"""
        return assign_id in self.agents

    def completed_assignments(self):
        """Returns the number of assignments this worker has completed"""
        complete_count = 0
        for agent in self.agents.values():
            if agent.get_status() == AssignState.STATUS_DONE:
                complete_count += 1
        return complete_count

    def disconnected_assignments(self):
        """Returns the number of assignments this worker has completed"""
        disconnect_count = 0
        for agent in self.agents.values():
            if agent.get_status() in [AssignState.STATUS_DISCONNECT,
                                      AssignState.STATUS_RETURNED]:
                disconnect_count += 1
        return disconnect_count


class WorkerManager():
    """Class used to keep track of workers state, as well as processing
    messages that come from the web client.
    """

    def __init__(self, mturk_manager, opt):
        self.opt = opt
        self.mturk_manager = mturk_manager
        self.mturk_workers = {}
        self.conv_to_agent = {}
        self.assignment_to_worker_id = {}
        self.hit_id_to_agent = {}
        self.time_blocked_workers = []
        self.load_disconnects()
        self.is_sandbox = mturk_manager.is_sandbox

    def get_worker_data_package(self):
        workers = []
        for mturk_worker in self.mturk_workers.values():
            worker = {
                'id': mturk_worker.worker_id,
                'accepted': len(mturk_worker.agents),
                'completed': mturk_worker.completed_assignments(),
                'disconnected': mturk_worker.disconnected_assignments(),
            }
            workers.append(worker)
        return workers

    def _create_agent(self, hit_id, assignment_id, worker_id):
        """Initialize an agent and return it"""
        return MTurkAgent(
            self.opt, self.mturk_manager, hit_id, assignment_id, worker_id)

    def _get_worker(self, worker_id):
        """A safe way to get a worker by worker_id"""
        return self.mturk_workers.get(worker_id, None)

    def _get_agent(self, worker_id, assignment_id):
        """A safe way to get an agent by worker and assignment_id"""
        worker = self._get_worker(worker_id)
        if worker is not None:
            return worker.agents.get(assignment_id, None)
        return None

    def route_packet(self, pkt):
        """Put an incoming message into the queue for the agent specified in
        the packet, as they have sent a message from the web client.
        """
        worker_id = pkt.sender_id
        assignment_id = pkt.assignment_id
        agent = self._get_agent(worker_id, assignment_id)
        if agent is not None:
            shared_utils.print_and_log(
                logging.INFO,
                'Manager received: {}'.format(pkt),
                should_print=self.opt['verbose']
            )
            # Push the message to the message thread to send on a reconnect
            agent.append_message(pkt.data)

            # Clear the send message command, as a message was recieved
            agent.set_last_command(None)
            agent.put_data(pkt.id, pkt.data)

    def map_over_agents(self, map_function, filter_func=None):
        """Take an action over all the agents we have access to, filters if
        a filter_func is given"""
        for worker in self.mturk_workers.values():
            for agent in worker.agents.values():
                if filter_func is None or filter_func(agent):
                    map_function(agent)

    def get_agent_for_assignment(self, assignment_id):
        """Returns agent for the assignment, none if no agent assigned"""
        if assignment_id not in self.assignment_to_worker_id:
            return None
        worker_id = self.assignment_to_worker_id[assignment_id]
        worker = self.mturk_workers[worker_id]
        return worker.get_agent_for_assignment(assignment_id)

    def time_block_worker(self, worker_id):
        self.time_blocked_workers.append(worker_id)
        self.mturk_manager.soft_block_worker(worker_id, 'max_time_qual')

    def un_time_block_workers(self, workers=None):
        if workers is None:
            workers = self.time_blocked_workers
            self.time_blocked_workers = []
        for worker_id in workers:
            self.mturk_manager.un_soft_block_worker(worker_id, 'max_time_qual')

    def load_disconnects(self):
        """Load disconnects from file, populate the disconnects field for any
        worker_id that has disconnects in the list. Any disconnect that
        occurred longer ago than the disconnect persist length is ignored
        """
        self.disconnects = []
        # Load disconnects from file
        file_path = os.path.join(parent_dir, DISCONNECT_FILE_NAME)
        compare_time = time.time()
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                old_disconnects = pickle.load(f)
                self.disconnects = [
                    d
                    for d in old_disconnects
                    if (compare_time - d['time']) < DISCONNECT_PERSIST_LENGTH
                ]
        # Initialize worker states with proper number of disconnects
        for disconnect in self.disconnects:
            worker_id = disconnect['id']
            if worker_id not in self.mturk_workers:
                # add this worker to the worker state
                self.mturk_workers[worker_id] = WorkerState(worker_id)
            self.mturk_workers[worker_id].disconnects += 1

    def save_disconnects(self):
        """Saves the local list of disconnects to file"""
        file_path = os.path.join(parent_dir, DISCONNECT_FILE_NAME)
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self.disconnects, f, pickle.HIGHEST_PROTOCOL)

    def handle_agent_disconnect(self, worker_id, assignment_id,
                                partner_callback):
        """Handles a disconnect by the given worker, calls partner_callback
        on all of the conversation partners of that worker
        """
        agent = self._get_agent(worker_id, assignment_id)
        if agent is not None:
            # Disconnect in conversation is not workable
            agent.set_status(AssignState.STATUS_DISCONNECT)
            # in conversation, inform others about disconnect
            conversation_id = agent.conversation_id
            if conversation_id in self.conv_to_agent:
                conv_participants = self.conv_to_agent[conversation_id]
                if agent in conv_participants:
                    for other_agent in conv_participants:
                        if agent.assignment_id != other_agent.assignment_id:
                            partner_callback(other_agent)
                if len(conv_participants) > 1:
                    # The user disconnected from inside a conversation with
                    # another turker, record this as bad behavior
                    self.handle_bad_disconnect(worker_id)

    def handle_bad_disconnect(self, worker_id):
        """Update the number of bad disconnects for the given worker, block
        them if they've exceeded the disconnect limit
        """
        if not self.is_sandbox:
            self.mturk_workers[worker_id].disconnects += 1
            self.disconnects.append({'time': time.time(), 'id': worker_id})
            if self.mturk_workers[worker_id].disconnects > MAX_DISCONNECTS:
                if self.opt['hard_block']:
                    text = (
                        'This worker has repeatedly disconnected from these '
                        'tasks, which require constant connection to complete '
                        'properly as they involve interaction with other '
                        'Turkers. They have been blocked after being warned '
                        'and failing to adhere. This was done in order to '
                        'ensure a better experience for other '
                        'workers who don\'t disconnect.'
                    )
                    self.mturk_manager.block_worker(worker_id, text)
                    shared_utils.print_and_log(
                        logging.INFO,
                        'Worker {} blocked - too many disconnects'.format(
                            worker_id
                        ),
                        True
                    )
                elif self.opt['disconnect_qualification'] is not None:
                    self.mturk_manager.soft_block_worker(
                        worker_id, 'disconnect_qualification')
                    shared_utils.print_and_log(
                        logging.INFO,
                        'Worker {} soft blocked - too many disconnects'.format(
                            worker_id
                        ),
                        True
                    )

    def worker_alive(self, worker_id):
        """Creates a new worker record if it doesn't exist, returns state"""
        if worker_id not in self.mturk_workers:
            self.mturk_workers[worker_id] = WorkerState(worker_id)
        return self.mturk_workers[worker_id]

    def assign_task_to_worker(self, hit_id, assign_id, worker_id):
        self.assignment_to_worker_id[assign_id] = worker_id
        agent = self._create_agent(hit_id, assign_id, worker_id)
        self.hit_id_to_agent[hit_id] = agent
        curr_worker_state = self.mturk_workers[worker_id]
        curr_worker_state.add_agent(agent)

    def get_complete_hits(self):
        """Returns the list of all currently completed HITs"""
        hit_ids = []
        for hit_id, agent in self.hit_id_to_agent.items():
            if agent.hit_is_complete:
                hit_ids.append(hit_id)
        return hit_ids

    def get_agent_work_status(self, assignment_id):
        """Get the current status of an assignment's work"""
        client = mturk_utils.get_mturk_client(self.is_sandbox)
        try:
            response = client.get_assignment(AssignmentId=assignment_id)
            status = response['Assignment']['AssignmentStatus']
            worker_id = self.assignment_to_worker_id[assignment_id]
            agent = self._get_agent(worker_id, assignment_id)
            if agent is not None and status == MTurkAgent.ASSIGNMENT_DONE:
                agent.hit_is_complete = True
            return status
        except ClientError as e:
            # If the assignment isn't done, asking for the assignment will fail
            not_done_message = ('This operation can be called with a status '
                                'of: Reviewable,Approved,Rejected')
            if not_done_message in e.response['Error']['Message']:
                return MTurkAgent.ASSIGNMENT_NOT_DONE
            else:
                shared_utils.print_and_log(
                    logging.WARN,
                    'Unanticipated error in `get_agent_work_status`: ' +
                    e.response['Error']['Message'],
                    should_print=True
                )
                # Assume not done if status check seems to be faulty.
                return MTurkAgent.ASSIGNMENT_NOT_DONE

    def _log_missing_agent(self, worker_id, assignment_id):
        """Logs when an agent was expected to exist, yet for some reason it
        didn't. If these happen often there is a problem"""
        shared_utils.print_and_log(
            logging.WARN,
            'Expected to have an agent for {}_{}, yet none was found'.format(
                worker_id,
                assignment_id
            )
        )

    def _get_agent_from_pkt(self, pkt):
        """Get the agent object corresponding to this packet's sender"""
        worker_id = pkt.sender_id
        assignment_id = pkt.assignment_id
        agent = self._get_agent(worker_id, assignment_id)
        if agent is None:
            self._log_missing_agent(worker_id, assignment_id)
        return agent

    def _change_agent_to_conv(self, pkt):
        """Update an agent to a new conversation given a packet from the
        conversation to be switched to
        """
        agent = self._get_agent_from_pkt(pkt)
        if agent is not None:
            self._assign_agent_to_conversation(agent, agent.conversation_id)

    def _assign_agent_to_conversation(self, agent, conv_id):
        """Register an agent object with a conversation id, update status"""
        agent.conversation_id = conv_id
        if conv_id not in self.conv_to_agent:
            self.conv_to_agent[conv_id] = []
        self.conv_to_agent[conv_id].append(agent)

    def change_agent_conversation(self, agent, conversation_id, new_agent_id):
        """Handle changing a conversation for an agent, takes a callback for
        when the command is acknowledged
        """
        agent.id = new_agent_id
        agent.conversation_id = conversation_id
        data = {
            'text': data_model.COMMAND_CHANGE_CONVERSATION,
            'conversation_id': conversation_id,
            'agent_id': new_agent_id
        }
        agent.flush_msg_queue()
        self.mturk_manager.send_command(
            agent.worker_id,
            agent.assignment_id,
            data,
            ack_func=self._change_agent_to_conv
        )

    def shutdown(self):
        """Handles cleaning up and storing state related to workers"""
        self.un_time_block_workers()
        self.save_disconnects()
