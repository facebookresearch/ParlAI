#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Modular Agent Mixin.

Provides shared functionality amongst modular agents.
"""
from abc import ABC
from typing import Dict, Any, List

from parlai.core.agents import Agent
from parlai.core.build_data import modelzoo_path
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.loader import load_agent_module
import parlai.utils.logging as logging


class ModularAgentMixin(Agent, ABC):
    """
    Modular agent mixin.

    Provides shared functionality across modular agents (seeker, BB3)
    """

    def init_shared_model(self, opt: Opt, top_agent: Agent):
        """
        Initialize a shared version of a top-level agent.

        This just makes sure that each "agent" has the same params, but different
        history objects.

        :param opt_key:
            which sub agent to create with the shared model.
        """
        opt.update(opt['override'])
        if 'model' in opt['override']:
            model_class = load_agent_module(opt['override']['model'])
        else:
            model_class = type(top_agent)
        shared = top_agent.share()
        shared['opt'] = opt
        return model_class(opt, shared)

    def get_subagent_opt(
        self,
        datapath: str,
        filename: str,
        specific_override_args: Dict[str, Any],
        general_override_args: Dict[str, Any],
    ) -> Opt:
        """
        Given an agent opt, construct the new opt for the agent.

        :param filename:
            opt path
        :param specific_override_args:
            args for the specific agent
        :param general_override_args:
            args specified for all agents
        """
        if not filename.endswith('.opt'):
            filename += '.opt'
        opt = Opt.load(modelzoo_path(datapath, filename))
        opt['override'] = {}
        blocklist_general = ['model', 'model_file', 'init_model']
        general_override_args['skip_generation'] = False

        # Remove the prefix for the model for the specific override args.
        specific_override_args = {
            '_'.join(k.split('_')[1:]): v for k, v in specific_override_args.items()
        }

        override_args = {**general_override_args, **specific_override_args}

        for k, v in override_args.items():
            if k not in blocklist_general and k in opt:
                logging.warning(f'Overriding {k} to {v} (old val: {opt[k]})')
                opt['override'][k] = v
            elif k in specific_override_args:
                logging.warning(f'Key {k} not originally in opt, setting to {v}')
                opt['override'][k] = v

        return opt

    def batch_act_search_query_generation(
        self,
        observations: Any,
        search_query_generation_observations: List[Message],
        search_indices: List[int],
        search_query_agent: Agent,
        search_knowledge_agent: Agent,
        inject_query_string: str,
    ) -> List[Message]:
        """
        Search Query Generator batch act.

        :param observations:
            list of observations
        :param search_indices:
            list of batch indices for which search is required.

        :return batch_reply:
            return the batch reply from the search query agent
        """
        batch_reply = [Message({}) for _ in range(len(observations))]
        search_queries = []
        if search_indices:
            batch_replies_with_search = search_query_agent.batch_act(
                [
                    o
                    for i, o in enumerate(search_query_generation_observations)
                    if i in search_indices
                ]
            )
            for i, reply in zip(search_indices, batch_replies_with_search):
                batch_reply[i] = reply
            search_queries = [o.get('text', '') for o in batch_reply]
            if inject_query_string:
                for i in range(len(search_queries)):
                    if search_queries[i]:
                        new_query = ' '.join([search_queries[i], inject_query_string])
                        search_queries[i] = new_query
                        batch_reply[i].force_set('text', new_query)
            logging.debug(f"Search Queries: {search_queries}")
        try:
            search_knowledge_agent.model_api.set_search_queries(search_queries)
        except AttributeError:
            # Gold Documents, most likely
            pass

        return batch_reply
