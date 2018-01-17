# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import math
import os
import pickle
import threading
import time
import uuid

from botocore.exceptions import ClientError

from parlai.messenger.core.agents import MessengerAgent
from parlai.messenger.core.socket_manager import Packet, SocketManager
from parlai.messenger.core.worker_state import WorkerState, AssignState
import parlai.messenger.core.data_model as data_model
import parlai.messenger.core.messenger_utils as messenger_utils
import parlai.messenger.core.server_utils as server_utils
import parlai.messenger.core.shared_utils as shared_utils

parent_dir = os.path.dirname(os.path.abspath(__file__))


class MessengerManager():
    """Manages interactions between agents on messenger as well as direct
    interactions between agents and the messenger overworld
    """

    def __init__(self, opt):
        """Create an MessengerManager using the given setup options
        """
        self.opt = opt
        self.server_url = None
        self.port = 443
        self.run_id = None
        self.agent_pool_change_condition = threading.Condition()
        self.overworld = None
        self.world_options = {}
        self.active_worlds = {}
        self.socket_manager = None
        self._init_logs()

    # Helpers and internal manager methods #

    def _init_state(self):
        """Initialize everything in the agent, task, and thread states"""
        self.agent_pool = []
        self.messenger_agents = {}
        self.messenger_instances = {}
        self.conv_to_agent = {}
        self.assignment_to_agent_ids = {}

    def _init_logs(self):
        """Initialize logging settings from the opt"""
        shared_utils.set_is_debug(self.opt['is_debug'])
        shared_utils.set_log_level(self.opt['log_level'])

    def _get_agent_from_pkt(self, pkt):
        """Get sender and assignment from a packet"""
        agent_id = pkt.sender_id
        assignment_id = pkt.assignment_id
        agent = self._get_agent(agent_id, assignment_id)
        if agent is None:
            self._log_missing_agent(agent_id, assignment_id)
        return agent

    def add_agent_to_pool(self, agent):
        """Add the agent to pool"""
        with self.agent_pool_change_condition:
            shared_utils.print_and_log(
                logging.DEBUG,
                "Adding agent {} to pool...".format(agent.id)
            )
            self.agent_pool.append(agent)

    def _expire_all_conversations(self):
        """iterate through all sub-worlds and shut them down"""
        # TODO implement this
        pass

    def _get_unique_pool(self, eligibility_function):
        """Return a filtered version of the agent pool where each agent is
        only listed a maximum of one time.
        """
        # TODO filter by psid -> agent id mappings for multi-page setup
        return self.agent_pool

    def _setup_socket(self):
        """Set up a socket_manager with defined callbacks"""
        self.socket_manager = SocketManager(
            self.server_url,
            self.port,
            self._on_alive,
            self._on_new_message,
            self._on_socket_dead,
            self.task_group_id
        )

    def _on_first_message(self, pkt):
        """Handle a new incoming message from a psid that has not yet been
        registered to any assignment.
        """
        # TODO this should register the user and enter them into the overworld,
        # spawning a thread for that world which will put them in a task
        # world when it is completed
        pass

    def _on_new_message(self, pkt):
        """Put an incoming message onto the correct agent's message queue.
        """
        # TODO find the correct agent to put this message into based on the
        # messenger_id to agent thing, then put it into that agent's message
        # queue on the spot. If no such agent, use the _on_first_message func
        pass

    def _create_agent(self, assignment_id, agent_id):
        """Initialize an agent and return it"""
        return MessengerAgent(self.opt, self, assignment_id, agent_id)

    def _onboard_new_worker(self, mturk_agent):
        """Handle creating an onboarding thread and moving an agent through
        the onboarding process
        """
        # TODO implement this. Involves creating the right onboarding and
        # task worlds for a particular agent, and setting up a thread that will
        # move the agent to a waiting world when onboarding is completed
        # # get state variable in question
        # worker_id = mturk_agent.worker_id
        # assignment_id = mturk_agent.assignment_id
        #
        # def _onboard_function(mturk_agent):
        #     """Onboarding wrapper to set state to onboarding properly"""
        #     if self.onboard_function:
        #         conversation_id = 'o_'+str(uuid.uuid4())
        #         mturk_agent.change_conversation(
        #             conversation_id=conversation_id,
        #             agent_id='onboarding',
        #             change_callback=self._set_worker_status_to_onboard
        #         )
        #         # Wait for turker to be in onboarding status
        #         mturk_agent.wait_for_status(AssignState.STATUS_ONBOARDING)
        #         # call onboarding function
        #         self.onboard_function(mturk_agent)
        #
        #     # once onboarding is done, move into a waiting world
        #     self._move_workers_to_waiting([mturk_agent])
        #
        # if assignment_id not in self.assignment_to_onboard_thread:
        #     # Start the onboarding thread and run it
        #     onboard_thread = threading.Thread(
        #         target=_onboard_function,
        #         args=(mturk_agent,),
        #         name='onboard-{}-{}'.format(worker_id, assignment_id)
        #     )
        #     onboard_thread.daemon = True
        #     onboard_thread.start()
        #
        #     self.assignment_to_onboard_thread[assignment_id] = onboard_thread
        pass

    def _get_agent_state(self, agent_id):
        """A safe way to get a worker by agent_id"""
        if agent_id in self.messenger_agents:
            return self.messenger_agents[agent_id]
        return None

    def _get_agent(self, agent_id, assignment_id):
        """A safe way to get an agent by agent_id and assignment_id"""
        agent_state = self._get_agent_state(agent_id)
        if agent_state is not None:
            if agent_state.has_assignment(assignment_id):
                return agent_state.get_agent_for_assignment(assignment_id)
        return None

    def _log_missing_agent(self, agent_id, assignment_id):
        """Logs when an agent was expected to exist, yet for some reason it
        didn't. If these happen often there is a problem"""
        shared_utils.print_and_log(
            logging.WARN,
            'Expected to have an agent for {}_{}, yet none was found'.format(
                agent_id,
                assignment_id
            )
        )

    # Manager Lifecycle Functions #

    def setup_server(self, task_directory_path=None):
        """Prepare the MTurk server for the new HIT we would like to submit"""
        fin_word = 'start'
        if self.opt['count_complete']:
            fin_word = 'finish'
        shared_utils.print_and_log(
            logging.INFO,
            '\nYou are going to allow workers from Amazon Mechanical Turk to '
            'be an agent in ParlAI.\nDuring this process, Internet connection '
            'is required, and you should turn off your computer\'s auto-sleep '
            'feature.\nEnough HITs will be created to fulfill {} times the '
            'number of conversations requested, extra HITs will be expired '
            'once the desired conversations {}.'.format(HIT_MULT, fin_word),
            should_print=True
        )
        input('Please press Enter to continue... ')
        shared_utils.print_and_log(logging.NOTSET, '', True)

        mturk_utils.setup_aws_credentials()

        # See if there's enough money in the account to fund the HITs requested
        num_assignments = self.required_hits
        payment_opt = {
            'type': 'reward',
            'num_total_assignments': num_assignments,
            'reward': self.opt['reward'],  # in dollars
            'unique': self.opt['unique_worker']
        }
        total_cost = mturk_utils.calculate_mturk_cost(payment_opt=payment_opt)
        if not mturk_utils.check_mturk_balance(
                balance_needed=total_cost,
                is_sandbox=self.opt['is_sandbox']):
            raise SystemExit('Insufficient funds')

        if ((not self.opt['is_sandbox']) and
                (total_cost > 100 or self.opt['reward'] > 1)):
            confirm_string = '$%.2f' % total_cost
            expected_cost = total_cost / HIT_MULT
            expected_string = '$%.2f' % expected_cost
            shared_utils.print_and_log(
                logging.INFO,
                'You are going to create {} HITs at {} per assignment, for a '
                'total cost up to {} after MTurk fees. Please enter "{}" to '
                'confirm and continue, and anything else to cancel.\nNote that'
                ' of the {}, the target amount to spend is {}.'.format(
                    self.required_hits,
                    '$%.2f' % self.opt['reward'],
                    confirm_string,
                    confirm_string,
                    confirm_string,
                    expected_string
                ),
                should_print=True
            )
            check = input('Enter here: ')
            if (check != confirm_string and ('$' + check) != confirm_string):
                raise SystemExit('Cancelling')

        shared_utils.print_and_log(logging.INFO, 'Setting up MTurk server...',
                                   should_print=True)
        mturk_utils.create_hit_config(
            task_description=self.opt['task_description'],
            unique_worker=self.opt['unique_worker'],
            is_sandbox=self.opt['is_sandbox']
        )
        # Poplulate files to copy over to the server
        if not self.task_files_to_copy:
            self.task_files_to_copy = []
        if not task_directory_path:
            task_directory_path = os.path.join(
                self.opt['parlai_home'],
                'parlai',
                'mturk',
                'tasks',
                self.opt['task']
            )
        self.task_files_to_copy.append(
            os.path.join(task_directory_path, 'html', 'cover_page.html'))
        try:
            for file_name in os.listdir(os.path.join(task_directory_path, 'html')):
                self.task_files_to_copy.append(os.path.join(
                    task_directory_path, 'html', file_name
                ))
        except FileNotFoundError:  # noqa F821 we don't support python2
            # No html dir exists
            pass
        for mturk_agent_id in self.mturk_agent_ids + ['onboarding']:
            self.task_files_to_copy.append(os.path.join(
                task_directory_path,
                'html',
                '{}_index.html'.format(mturk_agent_id)
            ))

        # Setup the server with a likely-unique app-name
        task_name = '{}-{}'.format(str(uuid.uuid4())[:8], self.opt['task'])
        self.server_task_name = \
            ''.join(e for e in task_name.lower() if e.isalnum() or e == '-')
        self.server_url = server_utils.setup_server(self.server_task_name,
                                                    self.task_files_to_copy)
        shared_utils.print_and_log(logging.INFO, self.server_url)

        shared_utils.print_and_log(logging.INFO, "MTurk server setup done.\n",
                                   should_print=True)

    def ready_to_accept_workers(self):
        """Set up socket to start communicating to workers"""
        shared_utils.print_and_log(logging.INFO,
                                   'Local: Setting up SocketIO...',
                                   not self.is_test)
        self._setup_socket()

    def start_new_run(self):
        """Clear state to prepare for a new run"""
        self.run_id = str(int(time.time()))
        self.task_group_id = '{}_{}'.format(self.opt['task'], self.run_id)
        self._init_state()
        self.topic_arn = mturk_utils.setup_sns_topic(
            self.opt['task'],
            self.server_url,
            self.task_group_id
        )

    def set_onboard_function(self, onboard_function):
        self.onboard_function = onboard_function

    def start_task(self, eligibility_function, assign_role_function,
                   task_function):
        """Handle running a task by checking to see when enough agents are
        in the pool to start an instance of the task. Continue doing this
        until the desired number of conversations is had.
        """

        def _task_function(opt, workers, conversation_id):
            """Wait for workers to join the world, then run task function"""
            shared_utils.print_and_log(
                logging.INFO,
                'Starting task {}...'.format(conversation_id)
            )
            shared_utils.print_and_log(
                logging.DEBUG,
                'Waiting for all workers to join the conversation...'
            )
            start_time = time.time()
            while True:
                all_joined = True
                for worker in workers:
                    # check the status of an individual worker assignment
                    if worker.state.status != AssignState.STATUS_IN_TASK:
                        all_joined = False
                if all_joined:
                    break
                if time.time() - start_time > WORLD_START_TIMEOUT:
                    # We waited but not all workers rejoined, throw workers
                    # back into the waiting pool. Stragglers will disconnect
                    # from there
                    shared_utils.print_and_log(
                        logging.INFO,
                        'Timeout waiting for {}, move back to waiting'.format(
                            conversation_id
                        )
                    )
                    self._move_workers_to_waiting(workers)
                    return
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)

            shared_utils.print_and_log(
                logging.INFO,
                'All workers joined the conversation {}!'.format(
                    conversation_id
                )
            )
            self.started_conversations += 1
            task_function(mturk_manager=self, opt=opt, workers=workers)
            # Delete extra state data that is now unneeded
            for worker in workers:
                worker.state.clear_messages()

            # Count if it's a completed conversation
            if self._no_workers_incomplete(workers):
                self.completed_conversations += 1

        while True:
            # Loop forever starting task worlds until desired convos are had
            with self.worker_pool_change_condition:
                valid_workers = self._get_unique_pool(eligibility_function)
                needed_workers = len(self.mturk_agent_ids)
                if len(valid_workers) >= needed_workers:
                    # enough workers in pool to start new conversation
                    self.conversation_index += 1
                    new_conversation_id = \
                        't_{}'.format(self.conversation_index)

                    # Add the required number of valid workers to the conv
                    workers = [w for w in valid_workers[:needed_workers]]
                    assign_role_function(workers)
                    # Allow task creator to filter out workers and run
                    # versions of the task that require fewer agents
                    workers = [w for w in workers if w.id is not None]
                    for w in workers:
                        w.change_conversation(
                            conversation_id=new_conversation_id,
                            agent_id=w.id,
                            change_callback=self._change_worker_to_conv
                        )
                        # Remove selected workers from the pool
                        self.worker_pool.remove(w)

                    # Start a new thread for this task world
                    task_thread = threading.Thread(
                        target=_task_function,
                        args=(self.opt, workers, new_conversation_id),
                        name='task-{}'.format(new_conversation_id)
                    )
                    task_thread.daemon = True
                    task_thread.start()
                    self.task_threads.append(task_thread)

            # Once we've had enough conversations, finish and break
            compare_count = self.started_conversations
            if (self.opt['count_complete']):
                compare_count = self.completed_conversations
            if compare_count == self.num_conversations:
                self.accepting_workers = False
                self.expire_all_unassigned_hits()
                self._expire_onboarding_pool()
                self._expire_worker_pool()
                # Wait for all conversations to finish, then break from
                # the while loop
                for thread in self.task_threads:
                    thread.join()
                break
            time.sleep(shared_utils.THREAD_MEDIUM_SLEEP)

    def shutdown(self):
        """Handle any mturk client shutdown cleanup."""
        # Ensure all threads are cleaned and state and HITs are handled
        try:
            self.expire_all_unassigned_hits()
            self._expire_onboarding_pool()
            self._expire_worker_pool()
            self.socket_manager.close_all_channels()
            for assignment_id in self.assignment_to_onboard_thread:
                self.assignment_to_onboard_thread[assignment_id].join()
        except BaseException:
            pass
        finally:
            server_utils.delete_server(self.server_task_name)
            mturk_utils.delete_sns_topic(self.topic_arn)
            self._save_disconnects()

    # MTurk Agent Interaction Functions #

    def force_expire_hit(self, worker_id, assign_id, text=None, ack_func=None):
        """Send a command to expire a hit to the provided agent, update State
        to reflect that the HIT is now expired
        """
        # Expire in the state
        agent = self._get_agent(worker_id, assign_id)
        if agent is not None:
            if not agent.state.is_final():
                agent.state.status = AssignState.STATUS_EXPIRED
                agent.hit_is_expired = True

        # Send the expiration command
        if text is None:
            text = ('This HIT is expired, please return and take a new '
                    'one if you\'d want to work on this task.')
        data = {'text': data_model.COMMAND_EXPIRE_HIT, 'inactive_text': text}
        self.send_command(worker_id, assign_id, data, ack_func=ack_func)

    def handle_turker_timeout(self, worker_id, assign_id):
        """To be used by the MTurk agent when the worker doesn't send a message
        within the expected window.
        """
        # Expire the hit for the disconnected user
        text = ('You haven\'t entered a message in too long, leaving the other'
                ' participant unable to complete the HIT. Thus this hit has '
                'been expired and you have been considered disconnected. '
                'Disconnect too frequently and you will be blocked from '
                'working on these HITs in the future.')
        self.force_expire_hit(worker_id, assign_id, text)

        # Send the disconnect event to all workers in the convo
        self._handle_worker_disconnect(worker_id, assign_id)

    def send_message(self, receiver_id, assignment_id, data,
                     blocking=True, ack_func=None):
        """Send a message through the socket manager,
        update conversation state
        """
        data['type'] = data_model.MESSAGE_TYPE_MESSAGE
        # Force messages to have a unique ID
        if 'message_id' not in data:
            data['message_id'] = str(uuid.uuid4())
        event_id = shared_utils.generate_event_id(receiver_id)
        packet = Packet(
            event_id,
            Packet.TYPE_MESSAGE,
            self.socket_manager.get_my_sender_id(),
            receiver_id,
            assignment_id,
            data,
            blocking=blocking,
            ack_func=ack_func
        )

        shared_utils.print_and_log(
            logging.INFO,
            'Manager sending: {}'.format(packet),
            should_print=self.opt['verbose']
        )
        # Push outgoing message to the message thread to be able to resend
        # on a reconnect event
        agent = self._get_agent(receiver_id, assignment_id)
        if agent is not None:
            agent.state.messages.append(packet.data)
        self.socket_manager.queue_packet(packet)

    def send_command(self, receiver_id, assignment_id, data, blocking=True,
                     ack_func=None):
        """Sends a command through the socket manager,
        update conversation state
        """
        data['type'] = data_model.MESSAGE_TYPE_COMMAND
        event_id = shared_utils.generate_event_id(receiver_id)
        packet = Packet(
            event_id,
            Packet.TYPE_MESSAGE,
            self.socket_manager.get_my_sender_id(),
            receiver_id,
            assignment_id,
            data,
            blocking=blocking,
            ack_func=ack_func
        )

        agent = self._get_agent(receiver_id, assignment_id)
        if (data['text'] != data_model.COMMAND_CHANGE_CONVERSATION and
                data['text'] != data_model.COMMAND_RESTORE_STATE and
                agent is not None):
            # Append last command, as it might be necessary to restore state
            agent.state.last_command = packet.data

        self.socket_manager.queue_packet(packet)

    def mark_workers_done(self, workers):
        """Mark a group of workers as done to keep state consistent"""
        for worker in workers:
            if not worker.state.is_final():
                worker.state.status = AssignState.STATUS_DONE

    def free_workers(self, workers):
        """End completed worker threads"""
        for worker in workers:
            self.socket_manager.close_channel(worker.get_connection_id())

    # Amazon MTurk Server Functions #

    def get_agent_work_status(self, assignment_id):
        """Get the current status of an assignment's work"""
        client = mturk_utils.get_mturk_client(self.is_sandbox)
        try:
            response = client.get_assignment(AssignmentId=assignment_id)
            return response['Assignment']['AssignmentStatus']
        except ClientError as e:
            # If the assignment isn't done, asking for the assignment will fail
            not_done_message = ('This operation can be called with a status '
                                'of: Reviewable,Approved,Rejected')
            if not_done_message in e.response['Error']['Message']:
                return MTurkAgent.ASSIGNMENT_NOT_DONE

    def create_additional_hits(self, num_hits, qualifications=None):
        """Handle creation for a specific number of hits/assignments
        Put created HIT ids into the hit_id_list
        """
        shared_utils.print_and_log(logging.INFO,
                                   'Creating {} hits...'.format(num_hits))
        if qualifications is None:
            qualifications = []

        # Add the soft block qualification if it has been specified
        if self.opt['block_qualification'] != '':
            block_qual_id = mturk_utils.find_or_create_qualification(
                self.opt['block_qualification'],
                'A soft ban from using a ParlAI-created HIT due to frequent '
                'disconnects from conversations, leading to negative '
                'experiences for other Turkers and for the requester.'
            )
            assert block_qual_id is not None, (
                'Hits could not be created as block qualification could not be'
                ' acquired. Shutting down server.'
            )
            qualifications.append({
                'QualificationTypeId': block_qual_id,
                'Comparator': 'DoesNotExist',
                'RequiredToPreview': True
            })

        hit_type_id = mturk_utils.create_hit_type(
            hit_title=self.opt['hit_title'],
            hit_description='{} (ID: {})'.format(self.opt['hit_description'],
                                                 self.task_group_id),
            hit_keywords=self.opt['hit_keywords'],
            hit_reward=self.opt['reward'],
            # Set to 30 minutes by default
            assignment_duration_in_seconds=self.opt.get(
                'assignment_duration_in_seconds', 30 * 60),
            is_sandbox=self.opt['is_sandbox'],
            qualifications=qualifications,
        )
        mturk_chat_url = '{}/chat_index?task_group_id={}'.format(
            self.server_url,
            self.task_group_id
        )
        shared_utils.print_and_log(logging.INFO, mturk_chat_url)
        mturk_page_url = None

        mturk_utils.subscribe_to_hits(
            hit_type_id,
            self.is_sandbox,
            self.topic_arn
        )

        if self.opt['unique_worker'] is True:
            # Use a single hit with many assignments to allow
            # workers to only work on the task once
            mturk_page_url, hit_id = mturk_utils.create_hit_with_hit_type(
                page_url=mturk_chat_url,
                hit_type_id=hit_type_id,
                num_assignments=num_hits,
                is_sandbox=self.is_sandbox
            )
            self.hit_id_list.append(hit_id)
        else:
            # Create unique hits, allowing one worker to be able to handle many
            # tasks without needing to be unique
            for _i in range(num_hits):
                mturk_page_url, hit_id = mturk_utils.create_hit_with_hit_type(
                    page_url=mturk_chat_url,
                    hit_type_id=hit_type_id,
                    num_assignments=1,
                    is_sandbox=self.is_sandbox
                )
                self.hit_id_list.append(hit_id)
        return mturk_page_url

    def create_hits(self, qualifications=None):
        """Create hits based on the managers current config, return hit url"""
        shared_utils.print_and_log(logging.INFO, 'Creating HITs...', True)

        mturk_page_url = self.create_additional_hits(
            num_hits=self.required_hits,
            qualifications=qualifications,
        )

        shared_utils.print_and_log(logging.INFO,
                                   'Link to HIT: {}\n'.format(mturk_page_url),
                                   should_print=True)
        shared_utils.print_and_log(
            logging.INFO,
            'Waiting for Turkers to respond... (Please don\'t close'
            ' your laptop or put your computer into sleep or standby mode.)\n',
            should_print=True
        )
        return mturk_page_url
