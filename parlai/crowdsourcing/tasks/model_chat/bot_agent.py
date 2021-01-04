#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import Dict, List, Optional

from omegaconf import DictConfig
import parlai.utils.logging as logging
from parlai.core.agents import create_agent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.crowdsourcing.tasks.model_chat.constants import AGENT_1
from parlai.utils.strings import normalize_reply


class TurkLikeAgent:
    """
    Will act like a Turker but actually contains a bot agent.
    """

    def __init__(self, opt, model_name, model_agent, num_turns, semaphore=None):
        self.opt = opt
        self.model_agent = model_agent
        self.id = AGENT_1
        self.num_turns = num_turns
        self.turn_idx = 0
        self.semaphore = semaphore
        self.worker_id = model_name
        self.hit_id = 'none'
        self.assignment_id = 'none'
        self.some_agent_disconnected = False
        self.hit_is_abandoned = False
        self.hit_is_returned = False
        self.disconnected = False
        self.hit_is_expired = False

    def act(self, timeout=None):
        _ = timeout  # The model doesn't care about the timeout
        if self.semaphore:
            with self.semaphore:
                act_out = self.model_agent.act()
        else:
            act_out = self.model_agent.act()
        act_out = Message(act_out)
        # Wrap as a Message for compatibility with older ParlAI models

        if 'dict_lower' in self.opt and not self.opt['dict_lower']:
            # model is cased so we don't want to normalize the reply like below
            final_message_text = act_out['text']
        else:
            final_message_text = normalize_reply(act_out['text'])

        act_out.force_set('text', final_message_text)
        assert ('episode_done' not in act_out) or (not act_out['episode_done'])
        self.turn_idx += 1
        return {**act_out, 'episode_done': False}

    def observe(self, observation, increment_turn: bool = True):
        """
        Need to protect the observe also with a semaphore for composed models where an
        act() may be called within an observe()
        """
        logging.info(
            f'{self.__class__.__name__}: In observe() before semaphore, self.turn_idx is {self.turn_idx} and observation is {observation}'
        )
        new_ob = copy.deepcopy(observation)
        if self.semaphore:
            with self.semaphore:
                self.model_agent.observe(new_ob)
        else:
            self.model_agent.observe(new_ob)
        logging.info(
            f'{self.__class__.__name__}: In observe() AFTER semaphore, self.turn_idx: {self.turn_idx}, observation["text"]: {new_ob["text"]}'
        )

        if increment_turn:
            self.turn_idx += 1

    def shutdown(self):
        pass

    def reset(self):
        self.model_agent.reset()

    @staticmethod
    def get_bot_agents(
        args: DictConfig,
        active_models: Optional[List[str]] = None,
        model_opts: Optional[Dict[str, str]] = None,
        no_cuda=False,
    ) -> Dict[str, dict]:
        """
        Return shared bot agents.

        Pass in model opts in one of two ways: (1) With the `model_opts` arg, where
        `model_opts` is a dictionary whose keys are   model names and whose values are
        strings that specify model params (i.e.   `--model image_seq2seq`). (2) With the
        `active_models` arg, a list of model names: those models' opts will   be read
        from args.blueprint.base_model_folder.
        """
        # NOTE: in the future we may want to deprecate the `active_models` arg, to move
        #  away from the paradigm of having all models in one folder

        model_overrides = {'model_parallel': args.blueprint.task_model_parallel}
        if no_cuda:
            # If we load many models at once, we have to keep it on CPU
            model_overrides['no_cuda'] = no_cuda
        else:
            logging.warn(
                'WARNING: MTurk task has no_cuda FALSE. Models will run on GPU. Will not work if loading many models at once.'
            )

        if active_models is not None:

            model_overrides.update(
                {
                    'datatype': 'valid',  # So we don't have to load the optimizer
                    'encode_candidate_vecs': True,  # For pulling from fixed list cands
                    'interactive_mode': True,
                    'skip_generation': False,
                }
            )
            # Add overrides that were historically used when reading models from a
            # static folder

            # Get the model nicknames from common folder and use them to load opts
            # from file
            base_model_folder = os.path.expanduser(args.blueprint.base_model_folder)
            models_available = []
            for obj in os.listdir(base_model_folder):
                if os.path.isdir(os.path.join(base_model_folder, obj)):
                    models_available.append(obj)
            logging.info(
                f'Found {len(models_available)} models available for Mturk task in {base_model_folder}: {models_available}'
            )

            all_model_opts = {}
            logging.info(f'Active models to use are: {active_models}')
            for model_nickname in active_models:
                model_overrides_copy = copy.deepcopy(model_overrides)
                model_path = os.path.join(base_model_folder, model_nickname, 'model')
                if os.path.isfile(model_path):
                    model_opt = {
                        'model_file': model_path,
                        'override': model_overrides_copy,
                    }
                else:
                    # Sometimes the model file is downloaded, like
                    # `-m hugging_face/dialogpt`
                    model_opt_path = model_path + '.opt'
                    logging.info(
                        f'Model file for model {model_nickname} does not exist! Instead, '
                        f'loading opt from {model_opt_path}.'
                    )
                    model_opt = Opt.load(model_opt_path)
                    if 'override' not in model_opt:
                        model_opt['override'] = {}
                    model_opt['override'].update(model_overrides_copy)
                all_model_opts[model_nickname] = model_opt

            final_model_opts = {m: all_model_opts[m] for m in active_models}

        elif model_opts is not None:

            parser = ParlaiParser(True, True)
            parser.set_params(**model_overrides)

            final_model_opts = {}
            for name, opt in model_opts.items():
                final_model_opts[name] = parser.parse_args(opt.split())

        else:

            raise ValueError('Either active_models or model_opts must be supplied!')

        logging.info(
            f'Got {len(list(final_model_opts.keys()))} active models with keys: {final_model_opts.keys()}.'
        )
        shared_bot_agents = {}
        for model_name, model_opt in final_model_opts.items():
            logging.info('\n\n--------------------------------')
            logging.info(f'model_name: {model_name}, opt_dict: {model_opt}')
            copied_opt_dict = copy.deepcopy(model_opt)
            model_agent = create_agent(model_opt, requireModelExists=True)

            if active_models is not None:
                # have to check that the options are set properly
                for k, v in copied_opt_dict.items():
                    if k != 'override':
                        assert model_agent.opt[k] == v

            shared_bot_agents[model_name] = model_agent.share()
        return shared_bot_agents
