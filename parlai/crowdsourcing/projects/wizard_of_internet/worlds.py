#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Union
from joblib import Parallel, delayed

from parlai.agents.rag.retrieve_api import SearchEngineRetriever
from parlai.crowdsourcing.utils.worlds import CrowdOnboardWorld, CrowdTaskWorld
from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.worlds import validate
import parlai.utils.logging as logging
from parlai.crowdsourcing.projects.wizard_of_internet import constants
from parlai.crowdsourcing.projects.wizard_of_internet.acceptability import (
    WizardOfInternetAcceptabilityChecker,
)

from mephisto.abstractions.blueprint import AgentState
from mephisto.abstractions.databases.local_database import LocalMephistoDB

mephisto_db = LocalMephistoDB()
ROLE_TALLY_CHACHE = {'data': None, 'last_update': None}


def sec_to_min_pretty(time_secs: int) -> str:
    """
    Returns formatted string for converting secs to mins.
    """
    if time_secs % 60 == 0:
        return f'{time_secs // 60}'
    m = time_secs / 60
    return f'{m:.2g}'


def get_worker_from_agent(agent: Agent):
    """
    Returns Mephisto worker for a given ParlAI agent.
    """
    return agent.mephisto_agent.get_worker()


def get_worker_by_name(worker_name: str):
    """
    Returns the Mephisto worker from their worker name.
    """
    workers = mephisto_db.find_workers(worker_name)
    if len(workers) != 1:
        logging.warning(f'Found {len(workers)} for worker {worker_name}')
        if not workers:
            return
    return workers[0]


def _is_wiz(agent: Agent):
    """
    Returns true if the `agent` is wizard.
    """
    return agent.agent_id == 'Wizard'


def _is_query(act: Message):
    """
    Checks if an agent action is a search query (only wizard can do this).
    """
    k = 'is_search_query'
    return k in act and act[k]


def _has_selected_sentence_from_search_results(action: Union[Dict, Message]):
    """
    Whether there is any knowledges selected with this message.
    """
    k_task = 'task_data'
    k_selected = 'selected_text_candidates'
    if (k_task in action) and (k_selected in action[k_task]):
        # Boolean value that user has not selected any option
        return not action[k_task][k_selected][0][0]
    return False


def create_search_agent(opt):
    """
    Creates and instance of SearchEngineRetriever object.
    """
    logging.info('Initializing the search engine API.')
    search_api_opt = deepcopy(opt)
    search_api_opt['skip_retrieval_token'] = None
    return SearchEngineRetriever(search_api_opt)


def run_search_query(query: str, search_client: SearchEngineRetriever):
    """
    Conducts search through the SearchEngineRetriever client, and sorts the retrieved
    docs.

    This function runs two searches for each query:
    1- <query> + " news"
    2- <query>

    The
    """

    def _search(q: str, n: int):
        """
        Sends the search query to the search API.
        """
        return search_client.retrieve([q], n)[0]

    def _dedupl_docs(docs_list):
        uniq_docs = []
        seen_urls = set()
        for d in docs_list:
            url = d['url']
            if url in seen_urls:
                continue
            uniq_docs.append(d)
            if len(uniq_docs) == constants.NUM_RETRIEVED_SEARCH_DOCS:
                return uniq_docs
            seen_urls.add(url)
        logging.warning(
            f'Only retrieved {len(uniq_docs)}, not {constants.NUM_RETRIEVED_SEARCH_DOCS}'
        )
        return uniq_docs

    def _wiki_sort_key(doc):
        """
        Helper function to put the Wikipedia pages last in ranking retrieved doc
        results.
        """
        url = doc['url']
        return 1 if url.startswith('https://en.wikipedia') else -1

    if not search_client:
        logging.error('No search client; can not run search request.')
        return
    logging.info(f'Running search for query "{query}"')

    # getting query with news
    query_had_news = 'news' in query
    if not query_had_news:
        search_results = _search(f'{query} news', constants.NUM_RETRIEVED_SEARCH_NEWS)
    else:
        search_results = []

    # getting web documents for the main search query
    search_results.extend(_search(query, constants.NUM_RETRIEVED_SEARCH_DOCS))

    # Remove a doc that was fetched by both news and regular search
    # and reduce the number of dosc to NUM_RETRIEVED_SEARCH_DOCS
    if not query_had_news:
        # We did not have two separate queries if query_had_news was True.
        search_results = _dedupl_docs(search_results)

    # Sorting retrieved docs based on their URL: Wikipedia pages go last.
    search_results.sort(key=_wiki_sort_key)

    return Message(
        {
            'id': constants.SEARCH_AGENT,
            'text': '*** SEARCH AGENT RESULTS (CHECK ACCOMPANIED DATA FOR RETRIEVED DOCS) ***',
            'task_data': {'search_results': search_results},
        }
    )


def _coordinator_send_message(
    agent, message: str = '', task_data: Dict = None, episode_done: bool = False
):
    """
    Sends a message to 'agent' from the coordinator.

    We use this to send a message to only one of the agents. It usually contains
    specific instructions, alerts, or warnings for certain situations during the task.
    """
    if not task_data:
        task_data = dict()
    agent.observe(
        {
            'id': constants.COORDINATOR_AGENT,
            'text': message,
            'episode_done': episode_done,
            'task_data': task_data,
        }
    )


def persona_from_template_values(topic: str, topic_item: str, extra_details: str = ''):
    """
    Generates a sentence stating the persona of the apprentice, given their selection.
    """
    pers = f'My favorite {topic} is {topic_item}.'
    if extra_details:
        pers += f'\n{extra_details}'
    return pers


def _form_response_get_field(form_response: Dict[str, Any], filed_num: int):
    """
    Extracts the value of a certain field from the Mephisto response.
    """
    frd = form_response['task_data']
    k = 'form_responses'
    if k in frd and len(frd[k]) and (filed_num < len(frd[k])):
        return frd[k][filed_num]['response']


def _form_response_main_persona(form_response: Dict[str, Any]):
    """
    Extracts the main selected persona from persona selection form response.
    """
    topic = _form_response_get_field(form_response, 0)
    entity = _form_response_get_field(form_response, 1)
    return persona_from_template_values(topic, entity)


def _form_response_persona_expantion(form_response: Dict[str, Any]):
    """
    Extracts the expanded details of persona from persona selection form response.
    """
    return _form_response_get_field(form_response, -1)


def _send_persona_too_short_warning(agent: Agent, persona_expantion: str):
    """
    Sends a warning to agent if persona details it too short.
    """
    _coordinator_send_message(
        agent,
        message=f'Your expansion on persona ("{persona_expantion}") was too short. '
        'Please rewrite to make a more elaborate and refined persona.',
    )


def _send_persona_overuse_warning(agent: Agent, main_persona: str):
    """
    Ask agent to choose another persona, if the selected one looks repeated.

    For example, we don't want 200 pesonas that reads "My favorite book is Harry
    Potter".
    """
    _coordinator_send_message(
        agent,
        message=f'The character you chose for the persona ("{main_persona}")'
        ' has already been used by others. Please choose some other character.',
    )


class SharedOnboardWorld(CrowdOnboardWorld):
    """
    The parent (base) onboarding class for both agents.
    """

    def __init__(self, opt: Opt, mturk_agent: Agent):
        super().__init__(opt, mturk_agent)
        self.agent.agent_id = 'Participant'
        self.role_training_qname = opt[constants.ROLE_QUALIFICATION_NAME_KEY]
        self._world_name = self._get_world_name()
        self._num_rounds = 0
        self.messages = []

    def _get_world_name(self):
        """
        Assigns a name to this world.
        """
        dt = datetime.now()
        return f'onboarding_world_{dt.strftime("%H-%M-%S")}'

    def wait_for_response(self, message: str = None, delay_time: int = 0):
        """
        Starts waiting for a response from the agent, after `delay_time` many seconds.
        """
        self._num_rounds += 1
        logging.info(
            f'{self._world_name} waiting for response at round {self._num_rounds}'
        )
        if delay_time > 0:
            time.sleep(delay_time)
        self.agent.observe(
            {'id': constants.ONBOARDING_AGENT, 'text': message, 'episode_done': False}
        )
        self.messages.append(self.agent.act(timeout=self.turn_timeout))

    def send_message(
        self,
        message: str,
        onboarding_step: int = None,
        done: bool = False,
        delay_time: int = 0,
    ):
        """
        Sends the next onboarding instruction to the agent.
        """
        task_data = dict()
        if onboarding_step:
            task_data['on_boarding_step'] = onboarding_step
        act = {
            'id': constants.ONBOARDING_AGENT,
            'text': message,
            'episode_done': done,
            'task_data': task_data,
        }
        if delay_time > 0:
            time.sleep(delay_time)
        self.agent.observe(act)

    def introduce_chat_interface(self):
        """
        Showing the welcome onboard message to the agent, the first step during the
        onboarding.
        """
        self.send_message(
            message=constants.ONBOARDING_WELCOME,
            onboarding_step=constants.ONBOARDING_STEPS["CHAT_INTERFACE"],
            done=True,
        )

    def go_for_start(self):
        """
        The onboarding graduation message.
        """
        self.send_message(message=constants.FINISHED_ONBOARDING, done=False)
        # waiting for agent to read the final message
        # then ending the onboarding (purging the onboarding world).
        time.sleep(5)
        self.send_message(
            message="", onboarding_step=constants.ONBOARDING_STEPS["WAITING"], done=True
        )

    def parley(self):
        """
        Provides a step by step scripted interactive onboarding for the agents.

        In each step, we introduce the agent to one part of their experience and
        expectations in this task (eg, persona, chat interface etc.). Then, after a
        short delay (parametrized by TUTORIAL_WAIT_TIMES), we as ask them to send a
        response to move to the next step. This method needs to be implemented for
        Wizard and Apprentice separately, as they have different onboarding experiences.
        """
        error_message = "Implement parley for each role individually."
        raise NotImplementedError(error_message)

    def get_worker(self):
        return get_worker_from_agent(self.agent)

    def get_worker_name(self):
        return self.get_worker().worker_name

    def grant_agent_training_qualification(self, role_id: int):
        """
        Granting the onboarding qualification to the agent, based on their assigned
        role.
        """
        role = constants.ROLE_QUALIFICATION_NAME_KEY[role_id]
        logging.info(f'Granting worker qualification for {role} role.')
        worker = self.get_worker()
        worker.grant_qualification(self.role_training_qname, role_id)

    def reason_to_reject(self):
        """
        Check for bad behavior for poor quality of work from agent.
        """
        if not self.episodeDone:
            return 'left/diconnected before the task was over.'

        # messages were too short
        messages_len = []
        for msg in self.messages:
            if self.agent.agent_id != msg['id']:
                # Not from this agent
                continue
            messages_len.append(len(msg['text']))
        msg_char_length_avg = sum(messages_len) / len(messages_len)
        if msg_char_length_avg < constants.MIN_AVG_CHAR_LENGTH_UTTERANCES:
            return (
                'messages were too short for meaningfull conversations '
                f'(average message length: {msg_char_length_avg:.2f} chars).'
            )

        # how many times talked abut persona
        n_persona_keyword_mentions = 0
        for msg in self.messages:
            if self.agent.agent_id != msg['id']:
                continue
            for keyword in constants.ONBOARDING_PERSONA_KEYWORDS:
                if keyword in msg['text'].lower():
                    n_persona_keyword_mentions += 1

        if n_persona_keyword_mentions < 1:
            return (
                'Did not talk enough about the persona. '
                f'Number of keyword overlaps: {n_persona_keyword_mentions}.'
            )

        # returning None means no reason to reject
        return None

    def shutdown(self):
        logging.info(f'Shutting down {self._world_name}')
        super().shutdown()
        logging.info('Shutdown completed successfully.')


class WizardOnboardingWorld(SharedOnboardWorld):
    """
    The onboarding world for the wizard agent.
    """

    def __init__(self, opt: Opt, mturk_agent: Agent):
        self.turn_timeout = opt['wizard_time_out']
        self._search_client = create_search_agent(opt)
        self.num_searches = 0
        super().__init__(opt, mturk_agent)

    def _get_world_name(self):
        return f'wizard-{super()._get_world_name()}'

    def introduce_knowledgeable_entity(self):
        self.send_message(constants.WIZARD_INTRODUCE_KNOWLEDGE)

    def introduce_search(self):
        self.send_message(message=constants.WIZARD_INTRODUCE_SEARCH)

    def try_search(self):
        self.send_message(
            message=constants.WIZARD_TRY_SEARCH,
            onboarding_step=constants.ONBOARDING_STEPS['TRY_SEARCH'],
        )

    def introduce_persona(self):
        self.send_message(
            message=constants.WIZARD_INTRODUCE_APPRENTICE_PERSONA,
            onboarding_step=constants.ONBOARDING_STEPS['PERSONA_WIZARD'],
        )

    def wait_for_response_with_search(self, message: str = None, delay_time: int = 0):
        """
        Send a message to Wizard and waits for a search or response action.
        """
        if message:
            self.send_message(message=message, delay_time=delay_time)

        time_out = self.turn_timeout
        agent = self.agent
        while time_out > 0:
            start_time = time.time()
            act = agent.act(timeout=time_out)
            if _is_query(act):
                self.num_searches += 1
                search_query = act['text']
                search_res = run_search_query(search_query, self._search_client)
                n = len(search_res['task_data']['search_results'])
                logging.info(
                    f'Retrieved {n} documents for search query "{search_query}".'
                )
                agent.observe(search_res)
            else:
                self.messages.append(act)
                return
            # subtracting the wait time from what was spent during search
            spent_time = time.time() - start_time
            time_out -= spent_time

    def parley(self):
        """
        The interactive onboarding for the Wizard.
        """
        wait_times = constants.TUTORIAL_WAIT_TIMES
        self.introduce_chat_interface()
        self.wait_for_response(
            message='Please type a greeting message to continue.',
            delay_time=wait_times['chat-interface'],
        )
        self.introduce_knowledgeable_entity()
        self.wait_for_response(
            message=constants.ONBOARDING_ACKNOWLEDGE_UNDERSTOOD,
            delay_time=wait_times['chat-interface'],
        )
        self.introduce_search()
        self.wait_for_response(
            message=constants.ONBOARDING_ACKNOWLEDGE_UNDERSTOOD,
            delay_time=wait_times['knowledge'],
        )
        self.try_search()
        self.wait_for_response_with_search()
        self.introduce_persona()
        self.wait_for_response_with_search()
        self.go_for_start()
        self.episodeDone = True

    def reason_to_reject(self):
        """
        Check for bad behavior for poor quality of work from wizard agent.
        """
        # Has used search enough
        if self.num_searches < constants.MIN_NUM_SEARCH_ONBOARDING:
            return f'did not use search enough (number of use {self.num_searches}).'

        # Has selected enough sentenes
        num_selections = 0
        for msg in self.messages:
            task_data = msg.get('task_data')
            if not (task_data and isinstance(task_data, dict)):
                continue
            sel_options = task_data.get('selected_text_candidates')
            if not sel_options or len(sel_options) == 1:  # No choices
                continue
            if not sel_options[0][0]:
                # sel_options[0][0] is "Did no use ..." option
                num_selections += 1

        if num_selections < constants.MIN_NUM_SELECTED_SENTENCES_ONBOARDING:
            return (
                'did not use or select search results enough times '
                f'(number of times used: {num_selections})'
            )
        return super().reason_to_reject()

    def prep_save_data(self, agent: Agent):
        """
        Saving session data after the world is closed.
        """
        rejection_reason = self.reason_to_reject()
        qualified_role = constants.WIZARD if self.episodeDone else constants.NO_ROLE
        return {
            constants.SAVED_DATA_IS_WIZARD_KEY: True,
            constants.SAVED_DATA_WORKER_KEY: self.get_worker_name(),
            constants.SAVED_DATA_ROLE_QUALIFICATION_DATA_KEY: (
                self.role_training_qname,
                qualified_role,
            ),
            constants.WORKER_REJECT_REASON: rejection_reason,
        }


class ApprenticeOnboardingWorld(SharedOnboardWorld):
    def __init__(self, opt, mturk_agent):
        self.turn_timeout = opt['apprentice_time_out']
        super().__init__(opt, mturk_agent)

    def _get_world_name(self):
        return f'apprentice-{super()._get_world_name()}'

    def introduce_persona(self):
        self.send_message(
            message=constants.APPRENTICE_INTRODUCE_PERSONA,
            onboarding_step=constants.ONBOARDING_STEPS['PERSONA_APPRENTICE'],
        )

    def introduce_partner_entity(self):
        self.send_message(message=constants.APPRENTICE_INTRODUCE_WIZARD)

    def introduce_partner_knowledge(self):
        self.send_message(message=constants.APPRENTICE_INTRODUCE_WIZARD_KNOWLEDGE)

    def parley(self):
        """
        The interactive onboarding for the Apprentice.
        """
        wait_times = constants.TUTORIAL_WAIT_TIMES
        self.introduce_chat_interface()
        self.wait_for_response(
            message='Please type a greeting message to continue.',
            delay_time=wait_times['chat-interface'],
        )
        self.introduce_persona()
        self.wait_for_response(
            message=constants.APPRENTICE_PERSONA_ROLE_INSTRUCTION,
            delay_time=wait_times['persona'],
        )
        self.introduce_partner_entity()
        self.wait_for_response(
            message=constants.APPRENTICE_CHITCHAT_INSTRUCTION,
            delay_time=wait_times['persona'],
        )
        self.introduce_partner_knowledge()
        self.wait_for_response(
            message=constants.APPRENTICE_PERSONA_MSG_INSTRUCTION,
            delay_time=wait_times['knowledge'],
        )
        self.go_for_start()
        self.episodeDone = True

    def prep_save_data(self, agent: Agent):
        """
        Saving session data after the world is closed.
        """
        rejection_reason = self.reason_to_reject()
        qualified_role = constants.APPRENTICE if self.episodeDone else constants.NO_ROLE
        return {
            constants.SAVED_DATA_IS_WIZARD_KEY: False,
            constants.SAVED_DATA_WORKER_KEY: self.get_worker_name(),
            constants.SAVED_DATA_ROLE_QUALIFICATION_DATA_KEY: (
                self.role_training_qname,
                qualified_role,
            ),
            constants.WORKER_REJECT_REASON: rejection_reason,
        }


class MTurkMultiAgentDialogWorld(CrowdTaskWorld):
    """
    The ParlAI world to run conversation, search, and flow.

    Two agents (wizard, apprentice) chat. One agent (wizard) has access to a search bar
    that they may use for seraching our knowledge source (common crawl here).
    """

    def __init__(self, opt: Opt, agents: List[Agent] = None):
        # Init world state
        self.agents = agents
        self._change_agents_order = False
        self.messages = []
        self.episodeDone = False
        self.turn_idx = 0
        self.num_search_queries = 0
        self.num_times_search_resutls_selected = 0
        self.world_tag = self._get_world_name()

        # Get world parameters from opt
        self.min_num_turns = opt['min_turns']
        self.wizard_time_out = opt['wizard_time_out']
        self.apprentice_time_out = opt['apprentice_time_out']
        self.search_warning_turn = opt['search_warning_turn']
        self.search_warning_threshold = opt['search_warning_threshold']
        self.select_warning_turn = opt['select_warning_turn']
        self.select_warning_threshold = opt['select_warning_threshold']
        self.soft_block_qname = opt['soft_block_qname']
        self.send_task_data = opt['send_task_data']
        self.role_training_qname = opt[constants.ROLE_QUALIFICATION_NAME_KEY]

        # The agent that checks the acceptability of the messages (quality and safety).
        self.acceptability_checker = self._get_acceptability_checker()

        # Number of pages to request for each wizard search
        self.num_passages_to_retrieve = opt['num_passages_retrieved']
        self._search_client = create_search_agent(opt)

        # Information about personas and their availability.
        self.personas_list = opt['personas']
        self.prev_persona_count = opt['prev_persona_count']
        self.max_times_persona_use = opt['max_times_persona_use']
        self.locations_list = opt['locations']
        self.persona_replacement = opt['pick_persona_with_replacement']
        self.selected_persona = None

        # Get worker names
        self.worker_names = dict()
        for a in self.agents:
            self.worker_names[a] = get_worker_from_agent(a).worker_name

    def _get_acceptability_checker(self):
        """
        Instantiate an instance of WizardOfInternetAcceptabilityChecker to monitor the
        world.
        """
        acr = WizardOfInternetAcceptabilityChecker()
        acr.min_words_violation_threshold = constants.MIN_AVG_WORD_LENGTH_UTTERANCES
        return acr

    def _get_world_name(self):
        dt = datetime.now()
        return f'cc_world_{dt.strftime("%H-%M-%S")}'

    def get_agent_order_mask(self, agent_index: int):
        """
        A mask for simulating rotation/reordering of agents.

        Use this method for accessing agents by a certaint order. Do not use
        self.agents[i] directly!
        """
        assert agent_index in (0, 1), 'Invalid index for accessing agents.'
        if self._change_agents_order:
            # 0->1  and   1->0
            agent_index = 1 - agent_index
        return self.agents[agent_index]

    def get_wizard_action(self, agent: Agent):
        """
        Handles wizard message or search action.
        """
        time_out = self.wizard_time_out
        while time_out > 0:
            start_time = time.time()
            act = agent.act(timeout=time_out)
            if _is_query(act):
                self.num_search_queries += 1
                search_res = run_search_query(act['text'], self._search_client)
                n = len(search_res['task_data']['search_results'])
                logging.info(f'{n} search results were retrieved.')
                agent.observe(search_res)
            else:
                if _has_selected_sentence_from_search_results(act):
                    self.num_times_search_resutls_selected += 1
                break

            # subtracting the wait time from what was spent during search
            spent_time = time.time() - start_time
            time_out -= spent_time

        return act

    def _send_task_objective_reminders(self, agent: Agent):
        """
        Monitors the stats for target activies. If needed, sends goal reminders to
        agent.

        This is mostly for checking if wizard does enough search and knowledge
        selection.
        """
        agent_id = agent.agent_id
        if agent_id == constants.ROLE_NAMES[constants.WIZARD]:
            # Checks if wizard has used search enough so far
            if (self.turn_idx >= self.search_warning_turn) and (
                self.num_search_queries < self.search_warning_threshold
            ):
                _coordinator_send_message(
                    agent, message=constants.USE_SEARCH_WARNING_MESSAGE
                )
            # Checks if wizard has selected search results enough times so far
            elif (self.turn_idx >= self.select_warning_turn) and (
                self.num_times_search_resutls_selected < self.select_warning_threshold
            ):
                _coordinator_send_message(
                    agent, message=constants.USE_SEARCH_RESULTS_WARNING_MESSAGE
                )

    def next_utterance(self, agent: Agent):
        """
        Handles receiving the next message from agent.
        """
        agent_id = agent.agent_id
        if agent_id == constants.ROLE_NAMES[constants.APPRENTICE]:
            return agent.act(timeout=self.apprentice_time_out)
        else:  # It is wizard
            return self.get_wizard_action(agent)

    def end_onboarding_state(self):
        """
        Sends a message to front-end app to announce transition from onboarding.
        """
        onboard_state = constants.ONBOARDING_STEPS['NOT_ONBOARDING']
        for agent in self.agents:
            agent.observe(onboarding_mode_toggle_message(onboard_state))

    def broadcast_apprentice_persona(self, persona: str):
        """
        Sends the selected apprentice persona to the front-end app for display.
        """
        for agent in self.agents:
            persona_msg = {
                'id': constants.PERSONA_AGENT,
                'text': '',
                'episode_done': False,
                'task_data': {'apprentice_persona': persona},
            }
            agent.observe(persona_msg)

    def shuffle_agents(self):
        """
        Changes the starting order: who goes first.
        """
        reorder = random.random() > 0.5
        if reorder:
            logging.info(f'Switching agents orders in {self.world_tag}')
            self._change_agents_order = True

    def sample_personas(self):
        """
        Generates a list of sampled personas, apprentice will choose from this list.
        """
        persona = self.personas_list
        n = constants.CURATED_PERSONA_CHOICES
        logging.info(
            f'Randomly choosing {n} personas from {len(persona)} available ones.'
        )
        if self.persona_replacement:
            return random.sample(persona, k=n)
        else:
            return [persona.pop() for _ in range(n)]

    def random_location(self):
        """
        Chooses a random location (only for personas that need one)
        """
        return random.choice(self.locations_list)

    def assign_roles(self):
        """
        Determines the order and the role of the agents in the world.

        Determines which agent goes first by random assignment. The agent roles are
        based on their onboarding qualification.
        """
        # Roling the dice for the starting order
        self.shuffle_agents()

        # The role and order assignment to the agents.
        starting_role = None
        for agent_index in range(len(self.agents)):
            agent = self.get_agent_order_mask(agent_index)
            worker = get_worker_from_agent(agent)
            qual = worker.get_granted_qualification(self.role_training_qname)
            assert qual
            role_qual = qual.value
            if role_qual == constants.WIZARD:
                agent.agent_id = 'Wizard'
            elif role_qual == constants.APPRENTICE:
                agent.agent_id = 'Apprentice'
            else:
                raise ValueError(f'Unrecognized role qulification {role_qual}.')
            if not starting_role:  # sets it the first time that loop runs
                starting_role = role_qual

        logging.info('Agent roles assigned.')
        logging.info(f'Agent with {self.get_agent_order_mask(0).agent_id} role starts.')
        return starting_role

    def _get_apprentice(self):
        if _is_wiz(self.agents[0]):
            return self.agents[1]
        else:
            return self.agents[0]

    def receive_form_response(self, agent: Agent, check_persona_overuse: bool = False):
        """
        Extracts the selected persona from the response form and validates it.
        """

        def generate_persona_key(persona_desc):
            ret = persona_desc.strip().lower()
            for sym in ('.', ',', ';', '!', '?'):
                ret = ret.replace(sym, ' ')
            return ' '.join([s for s in ret.split(' ') if s])

        # Repeat asking for persona until having a valid one.
        acceptable_response = False
        while not acceptable_response:
            agent_resp = agent.act(timeout=self.wizard_time_out)

            pers_exp = _form_response_persona_expantion(agent_resp)

            # Too short
            if not pers_exp or len(pers_exp) < constants.PERSONA_EXPANSION_MIN_LEN_CHAR:
                _send_persona_too_short_warning(agent, pers_exp)
                continue

            # Persona was selected before
            if check_persona_overuse:
                persona_key = generate_persona_key(
                    _form_response_main_persona(agent_resp)
                )
                if self.prev_persona_count[persona_key] >= self.max_times_persona_use:
                    _send_persona_overuse_warning(agent, persona_key)
                    continue
                self.prev_persona_count[persona_key] += 1

            acceptable_response = True
        return agent_resp

    def _update_curated_personas_use(self, persona: str):
        """
        Updates the persona use count.

        Increases the count for the number of times that the selected `persona` was
        used, and removes it from available list of personas if it was selected too many
        times.
        """
        lower_persona = persona.lower()
        self.prev_persona_count[lower_persona] += 1
        if self.prev_persona_count[lower_persona] < self.max_times_persona_use:
            return

        logging.info(f'Trying to remove "{persona}" from list of personas.')
        if len(persona) < constants.CURATED_PERSONA_CHOICES:
            logging.warning(
                'Not enough personas may remain after removing, canceling removal.'
            )
            return

        self.personas_list.remove(persona)
        logging.info(
            f'New number of available personas is "{len(self.personas_list)}".'
        )

    def _choose_curated_persona(self):
        """
        Asks apprentice to choose a persona from the curated list of personas.
        """
        persona_opts = self.sample_personas()
        apprentice_agent = self._get_apprentice()

        # Removing PERSONA_NEEDS_LOCATION_TOKEN from what agents will see
        persona_opts_views = [
            p.replace(constants.PERSONA_NEEDS_LOCATION_TOKEN, '') for p in persona_opts
        ]
        persona_selection_form = [
            {
                'type': 'choices',
                'question': 'Choose one of these personas to start:',
                'choices': persona_opts_views,
            },
            {'type': 'text', 'question': 'Add something imaginative to refine it:'},
        ]
        _coordinator_send_message(
            apprentice_agent,
            message=constants.APPRENTICE_CHOOSE_CURATED_PERSONA_REQUEST,
            task_data={'respond_with_form': persona_selection_form},
        )
        agent_response = self.receive_form_response(apprentice_agent)

        rs = [r['response'] for r in agent_response['task_data']['form_responses']]
        assert len(rs) == 2, 'Persona response form length is not 2.'
        selected_persona, added_persona = rs
        apprentice_persona = f'{selected_persona}\n{added_persona}'
        worker_name = self.worker_names[apprentice_agent]
        logging.info(f'Agent ({worker_name}) selected a persona: {apprentice_persona}')

        selected_persona_ind = persona_opts_views.index(selected_persona)
        # Checking if persona needs location
        if constants.PERSONA_NEEDS_LOCATION_TOKEN in persona_opts[selected_persona_ind]:
            apprentice_location = self.random_location()
            logging.info(f'Persona needs a location. {apprentice_location} selected.')
            apprentice_persona = (
                f'I live in {apprentice_location}.\n{apprentice_persona}'
            )

        # Checking if the persona was used too often and needs to be removed.
        self._update_curated_personas_use(persona_opts[selected_persona_ind])

        return apprentice_persona

    def _choose_templated_topics_persona(self):
        """
        Asks apprentice to choose a persona using the provided template.
        """

        topic_bundles = random.sample(
            constants.TEMPLATE_PERSONAS_TOPICS, k=constants.TEMPLATE_PERSONAS_CHOICES
        )  # Each topic bundle is string of comma-seperated related topics, eg. "book,author"
        topics = []
        for tb in topic_bundles:
            topics.extend(tb.split(','))

        apprentice_agent = self._get_apprentice()
        persona_selection_form = [
            {
                'type': 'choices',
                'question': 'My character\'s favorite ',
                'choices': topics,
            },
            {'type': 'text', 'question': 'is '},
            {'type': 'text', 'question': 'Add something imaginative to refine it:'},
        ]
        _coordinator_send_message(
            apprentice_agent,
            message=constants.APPRENTICE_CHOOSE_PERSONA_TEMPLATE_REQUEST,
            task_data={'respond_with_form': persona_selection_form},
        )
        agent_response = self.receive_form_response(
            apprentice_agent, check_persona_overuse=True
        )

        rs = [r['response'] for r in agent_response['task_data']['form_responses']]
        assert len(rs) == 3, 'Template persona response form length is not 3.'
        topic, topic_item, extra_details = rs
        apprentice_persona = persona_from_template_values(
            topic, topic_item, extra_details
        )
        worker_name = self.worker_names[apprentice_agent]
        logging.info(f'Agent ({worker_name}) selected a persona: {apprentice_persona}')

        return apprentice_persona

    def _reset_to_text_response(self, agent):
        """
        Returns Mephisto response from form to text.
        """
        _coordinator_send_message(agent=agent, task_data={'respond_with_form': False})

    def apprentice_choose_persona(self):
        """
        Randomly selects a persona selection type (template, curated) and asks agent.
        """
        logging.info('Randomly choosing persona selection type.')
        choose_from_templates = (
            random.random() < constants.PROBABILITY_CHOOSING_TEMPLATE_PERSONA
        )
        if choose_from_templates:
            logging.info('Choosing persona persona from template.')
            resp = self._choose_templated_topics_persona()
        else:
            logging.info('Choosing persona persona from curated cases.')
            resp = self._choose_curated_persona()
        self._reset_to_text_response(self._get_apprentice())
        return resp

    def send_time_length_info(self):
        """
        Sends a message to agents informing them about the length of the task (turns,
        and timeout).
        """
        min_rounds = self.min_num_turns
        wiz_time = sec_to_min_pretty(self.wizard_time_out)
        app_time = sec_to_min_pretty(self.apprentice_time_out)
        for agent in self.agents:
            message = f'This conversation continues for at least {min_rounds} rounds.\n'
            t = wiz_time if _is_wiz(agent) else app_time
            message += (
                f'In your turn, please send your message within {t} minutes. '
                'Otherwise you may be disqualified. '
            )
            if not _is_wiz(agent):
                message += (
                    f'Note that you might have to wait up to {wiz_time} '
                    'mintes to receive a response from the other person.'
                )
            agent.observe(
                {
                    'id': constants.COORDINATOR_AGENT,
                    'text': message,
                    'episode_done': False,
                }
            )

    def send_starter_instruction(self, role: int):
        """
        Sends a reminder about the role and goals in the beginning of chat.
        """
        message_text = None
        if role == constants.WIZARD:
            message_text = constants.WIZARD_STARTING_INSTRUCTION
        else:
            assert role == constants.APPRENTICE
            message_text = constants.APPRENTICE_STARTING_INSTRUCTION
        start_instruction_message = {
            'id': constants.COORDINATOR_AGENT,
            'text': message_text,
            'episode_done': False,
        }
        self.get_agent_order_mask(0).observe(start_instruction_message)

    def send_wizard_persona_emphasize_message(self):
        """
        Sends a message to wizard emphasizing on main goal here (apprentice persona).
        """
        for agent in self.agents:
            if not _is_wiz(agent):
                continue
            agent.observe(
                {
                    'id': constants.COORDINATOR_AGENT,
                    'text': constants.WIZARD_PERSONA_EMPHASIZE,
                    'episode_done': False,
                }
            )

    def setup_roles_and_persona(self):
        """
        Prepares the chat environment and states before starting agent interactions.
        """
        logging.info('Setting up roles, orders, persona.')
        self.end_onboarding_state()
        self.broadcast_apprentice_persona('')  # clear onboarding persona
        starting_role = self.assign_roles()
        self.send_wizard_persona_emphasize_message()
        self.selected_persona = self.apprentice_choose_persona()
        self.broadcast_apprentice_persona(self.selected_persona)
        self.send_time_length_info()
        self.send_starter_instruction(starting_role)

    def parley(self):
        """
        parley process for the agents: running the chat world.
        """
        if self.turn_idx == 0:
            self.setup_roles_and_persona()

        self.turn_idx += 1
        logging.info(
            f'{self.world_tag} is at turn {self.turn_idx}...\n'
            f'Wizard has searched {self.num_search_queries} times and '
            f'selected results {self.num_times_search_resutls_selected} times.'
        )

        for idx in range(len(self.agents)):
            agent = self.get_agent_order_mask(idx)
            act = self.next_utterance(agent)
            self.messages.append(deepcopy(act))
            if self.send_task_data:
                act.force_set(
                    'task_data',
                    {
                        'last_acting_agent': agent.agent_id,
                        'current_dialogue_turn': self.turn_idx,
                        'utterance_count': self.turn_idx + idx,
                    },
                )

            if 'requested_finish' in act and act['requested_finish']:
                # One of the agents has requested for end of the chat.
                self.episodeDone = True
                break

            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(act))

            # Reminds wizard about searching and selecting knowledge if needed.
            self._send_task_objective_reminders(agent)

    def _reason_to_disqualify(self, agent: Agent):
        """
        Determining if agents had low quality work or had unsafe behaviour.
        """
        # Disconncet or timeout
        mephisto_agent = agent.mephisto_agent
        if mephisto_agent.get_status() in (
            AgentState.STATUS_EXPIRED,
            AgentState.STATUS_TIMEOUT,
        ):
            return 'agent was disconnected.'

        # Wizard not using search enough
        if agent.agent_id == 'Wizard' and (
            (self.num_search_queries < self.search_warning_threshold)
            or (self.num_times_search_resutls_selected < self.select_warning_threshold)
        ):
            return (
                'blocked for not enough search activity '
                f'({self.num_search_queries} searches; '
                f'{self.num_times_search_resutls_selected} selected sentecnes).'
            )

        acceptability_checker_results = self.acceptability_checker.check_messages(
            agent.agent_id,
            self.selected_persona,
            messages=self.messages,
            is_worker_0=False,
            violation_types=constants.ACCEPTABILITY_VIOLATIONS,
        )
        if acceptability_checker_results:
            return f'ParlAI acceptability checker found violations: "{acceptability_checker_results}"'

    def _soft_block_agent(self, agent):
        """
        Softblocking the agent: they can not participate in this task anymore.
        """
        worker = get_worker_from_agent(agent)
        logging.warning(f'Soft blocking {worker.worker_name}')
        worker.grant_qualification(self.soft_block_qname)

    def prep_save_data(self, agent_as_list):
        """
        Saving the chat data, after checking its quality and safety.
        """
        agent = agent_as_list[0]
        agent_id = agent.agent_id

        logging.info(f'Preparing saved data for {agent_id}')
        ret = {'agent_id': agent_id, 'message_history_copy': self.messages}
        disqualify_reason = self._reason_to_disqualify(agent)
        if disqualify_reason:
            logging.info(f'Disqualified submission detecetd: "{disqualify_reason}"')
            ret['disqualify_reason'] = disqualify_reason
            self._soft_block_agent(agent)

        return ret

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent

        Parallel(n_jobs=len(self.agents), backend='threading')(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )


def onboarding_mode_toggle_message(onboarding_step):
    """
    Formats a message to be sent to front-end to detemine the state of onboarding.
    """
    return {
        'id': constants.ONBOARDING_AGENT,
        'text': '',
        'episode_done': False,
        'task_data': {'on_boarding_step': onboarding_step},
    }


def _get_cached_roll_tally():
    """
    Returns the role tally counts from cache, if the cache is not expired.
    """
    utime = ROLE_TALLY_CHACHE['last_update']
    if not utime:
        logging.info('Initiated rolls tally cache.')
        return None

    dt = time.time() - utime
    logging.info(f'The last rolls tally cached {dt:.2f} seconds ago.')
    if dt > constants.TALLY_CACHE_TIMEOUT:
        logging.info(
            'Rolls tally cache is outdated '
            f'(is greater than {constants.TALLY_CACHE_TIMEOUT} s).'
        )
        return None

    logging.info(
        'Rolls tally is fresh enough to use '
        f'(is less than {constants.TALLY_CACHE_TIMEOUT} s).'
    )
    return ROLE_TALLY_CHACHE['data']


def _cache_roll_tally(rolls_tally: Dict[int, int]):
    """
    Updates the content of the roles tally cache.
    """
    logging.info('Setting rolls tally cache.')
    ROLE_TALLY_CHACHE['last_update'] = time.time()
    ROLE_TALLY_CHACHE['data'] = rolls_tally


def find_needed_role(agent, rqname: str):
    """
    Determines the role that the agent starting the onboarding needs to go through.

    Checks the number of agents who passed the onboarding and are waiting to be matched,
    and the agents who are currently in the onboarding. Based the number of roles in the
    pool decides what role a newcoming agent needs to be trained on.

    It caches the recent values of tally to avoid heavy DB queries.
    The cache value is handled by ROLE_TALLY_CHACHE global variable.
    To control the cache freshness and time out set TALLY_CACHE_TIMEOUT (in seconds).
    """
    role_tally = _get_cached_roll_tally()
    if not role_tally:
        role_tally = {constants.WIZARD: 0, constants.APPRENTICE: 0}
        db = agent.mephisto_agent.db
        task_run_id = agent.mephisto_agent.task_run_id
        agents_need_paring = db.find_onboarding_agents(
            status=AgentState.STATUS_ONBOARDING, task_run_id=task_run_id
        )
        agents_need_paring.extend(
            db.find_agents(status=AgentState.STATUS_WAITING, task_run_id=task_run_id)
        )

        no_qual = 0
        unk_qual = 0
        this_agent_id = agent.mephisto_agent.get_agent_id()
        for ag in agents_need_paring:
            if ag.get_agent_id() == this_agent_id:
                continue
            worker = ag.get_worker()
            worker_qualification = worker.get_granted_qualification(rqname)
            if not worker_qualification:
                no_qual += 1
                continue
            qstatus = worker_qualification.value
            if qstatus in (constants.WIZARD, constants.WIZARD_IN_TRAINING):
                role_tally[constants.WIZARD] += 1
            elif qstatus in (constants.APPRENTICE, constants.APPRENTICE_IN_TRAINING):
                role_tally[constants.APPRENTICE] += 1
            else:
                unk_qual += 1
        if no_qual or unk_qual:
            logging.warning(
                f'\tNo qualifications: {no_qual}\tUnknown qualifications: {unk_qual}'
            )
        _cache_roll_tally(role_tally)

    logging.info(
        f'Wizard: {role_tally[constants.WIZARD]}\tApprentices: {role_tally[constants.APPRENTICE]}'
    )
    if role_tally[constants.WIZARD] > role_tally[constants.APPRENTICE]:
        logging.info('Onboarding a new Apprentice.')
        role_tally[constants.APPRENTICE] += 1
        return constants.APPRENTICE
    else:
        logging.info('Onboarding a new Wizard.')
        role_tally[constants.WIZARD] += 1
        return constants.WIZARD


def make_onboarding_world(opt, agent: Agent):
    """
    Assigns agents to apporopraite onboarding worlds to balance the roles.
    """
    role_qual_name = opt[constants.ROLE_QUALIFICATION_NAME_KEY]

    def assign_role_based_on_ques(agent):
        worker = get_worker_from_agent(agent)
        needed_worker = find_needed_role(agent, role_qual_name)
        if needed_worker == constants.WIZARD:
            worker.grant_qualification(role_qual_name, constants.WIZARD_IN_TRAINING)
            return WizardOnboardingWorld(opt, agent)
        else:
            worker.grant_qualification(role_qual_name, constants.APPRENTICE_IN_TRAINING)
            return ApprenticeOnboardingWorld(opt, agent)

    # sends a message to UI to set the onboarding step.
    agent.observe(
        onboarding_mode_toggle_message(constants.ONBOARDING_STEPS['CHAT_INTERFACE'])
    )
    worker_qualification = get_worker_from_agent(agent).get_granted_qualification(
        role_qual_name
    )

    if not worker_qualification:  # Has not started onboarding before
        return assign_role_based_on_ques(agent)
    else:  # Had been in onboarding but didn't finish
        qstatus = worker_qualification.value
        if qstatus == constants.WIZARD_IN_TRAINING:
            return WizardOnboardingWorld(opt, agent)
        elif qstatus == constants.APPRENTICE_IN_TRAINING:
            return ApprenticeOnboardingWorld(opt, agent)
        else:
            logging.warning(
                f'Unknown qualification status "{qstatus}" during creating onboarding workds'
                + 'Assigning the roles based on waiting and onboarding agents queue size.'
            )
            return assign_role_based_on_ques(agent)


def assign_role_training_qualification(
    worker, role_qulification_name: str, role_qulification_value: int
):
    """
    Syncs the training qualification of the agent (worker) with the DB.
    """
    if not role_qulification_value or role_qulification_value == constants.NO_ROLE:
        logging.warning('Agent did not qualify for a role.')
        return False
    role_name = constants.ROLE_NAMES[role_qulification_value]
    logging.info(f'Agent qulified for {role_name} role. Granting worker qualification.')
    worker.grant_qualification(role_qulification_name, role_qulification_value)
    return True


def validate_onboarding(data: Dict):
    """
    Check the contents of the data to ensure they are valid and safe.
    """
    try:
        saved_data = data['outputs']['messages'][-1]['data']['WORLD_DATA']
        role = (
            'Wizard' if saved_data[constants.SAVED_DATA_IS_WIZARD_KEY] else 'Apprentice'
        )
        logging.info(f'Validating {role} onboarding.')
    except (IndexError, KeyError) as e:
        logging.warning(
            'Incomplete data to validate agent onboarding.'
            f'Onboarding saved_data error: {e}'
        )
        return False

    rejection_reason = saved_data[constants.WORKER_REJECT_REASON]
    if rejection_reason:
        logging.warning(f'Rejected: {rejection_reason}')
        return False

    # Role qualification
    worker = get_worker_by_name(saved_data[constants.SAVED_DATA_WORKER_KEY])
    qual_name, qual_val = saved_data[constants.SAVED_DATA_ROLE_QUALIFICATION_DATA_KEY]
    if not assign_role_training_qualification(worker, qual_name, qual_val):
        return False

    logging.info('Onboarding work accepted.')
    return True


def make_world(opt, agents: Agent):
    return MTurkMultiAgentDialogWorld(opt, agents)


def get_world_params():
    return {'agent_count': 2}
