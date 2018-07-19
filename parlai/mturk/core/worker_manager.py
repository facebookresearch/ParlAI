# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.mturk.core.agents import MTurkAgent
import parlai.mturk.core.data_model as data_model
import os

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

    def add_agent(self, assign_id, mturk_agent):
        """Add an assignment to this worker state with the given assign_it"""
        self.agents[assign_id] = mturk_agent

    def get_agent_for_assignment(self, assignment_id):
        return self.agents.get(assignment_id, None)

    def has_assignment(self, assign_id):
        """Returns true if this worker has an assignment for the given id"""
        return assign_id in agents

    def completed_assignments(self):
        """Returns the number of assignments this worker has completed"""
        complete_count = 0
        for agent in self.agents.values():
            if agent.get_status() == AssignState.STATUS_DONE:
                complete_count += 1
        return complete_count

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
        self._load_disconnects()

    def _create_agent(self, hit_id, assignment_id, worker_id):
        """Initialize an agent and return it"""
        return MTurkAgent(
            self.opt, self.mturk_manager, hit_id, assignment_id, worker_id)

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
                    self.block_worker(worker_id, text)
                    shared_utils.print_and_log(
                        logging.INFO,
                        'Worker {} blocked - too many disconnects'.format(
                            worker_id
                        ),
                        True
                    )
                elif self.opt['disconnect_qualification'] is not None:
                    self.soft_block_worker(worker_id,
                                           'disconnect_qualification')
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
        curr_worker_state.add_agent(assign_id, agent)

    def get_complete_hits(self):
        """Returns the list of all currently completed HITs"""
        hit_ids = []
        for hit_id, agent in self.hit_id_to_agent.items():
            if agent.hit_is_complete:
                hit_ids.append(hit_id)
        return hit_ids

    def shutdown(self):
        """Handles cleaning up and storing state related to workers"""
        self.un_time_block_workers()
        self.save_disconnects()
