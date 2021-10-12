#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Default agents for TODWorld.
"""

from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.distributed import is_distributed, get_rank, num_workers

import parlai.core.tod.tod_core as tod
from parlai.core.tod.tod_core import SerializationHelpers
from parlai.core.tod.impl.teacher_metrics import SlotMetrics, NlgMetrics

from typing import Optional, List
import json
import pickle
import difflib
import random
from math import ceil

######### Agents that dump information from a dataset; base classes


class TodStructuredDataParser(Agent):
    """
    Base class that specifies intermediate generation representations.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        if hasattr(super(), "add_cmdline_args"):
            parser = super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("TOD StructuredData agent")
        group.add_argument(
            "--episodes-randomization-seed",
            type=int,
            default=-1,
            help="Randomize episodes in a predictable way (eg, for few shot). Set to -1 for no randomization. ",
        )
        parser.add_argument(
            "--n-shot",
            default=-1,
            type=int,
            help="Number of dialogues to keep for each of train/valid/test. -1 means all. Dialogues of lower numbers are strict subsets of larger numbers. Do not use in conjunction with `--percent-shot`. Use `--episodes-randomization-seed` to change seed. NOTE: Beware of using this flag when multitasking as this will apply to *all* datasets unless the ':' syntax for specifying per-dataset flags is used.",
        )
        parser.add_argument(
            "--percent-shot",
            default=-1,
            type=float,
            help="Percentage of dialogues to keep for each of train/valid/test. -1 means all. Dialogues of lower numbers are strict subsets of larger numbers. Do not use in conjunction with `--n-shot`. Use `--episodes-randomization-seed` to change seed. NOTE: Beware of using this flag when multitasking as this will apply to *all* datasets unless the ':' syntax for specifying per-dataset flags is used.",
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.id = self.get_id_task_prefix() + "_" + self.get_agent_type_suffix()
        if shared is None:
            self.episodes = self.generate_episodes()
        else:
            self.episodes = shared["episodes"]

    def share(self):
        share = super().share()
        share["episodes"] = self.episodes
        return share

    def setup_episodes(self, fold: str) -> List[tod.TodStructuredEpisode]:
        """
        Fold here is a data fold.
        """
        raise NotImplementedError("Must have method for generating an episode")

    def generate_episodes(self) -> List[tod.TodStructuredEpisode]:
        if self.opt.get("n_shot", -1) >= 0 and self.opt.get("percent_shot", -1) >= 0:
            # Validate before spending a while to load eeverything
            raise RuntimeError("Both `--n-shot` and `--percent-shot` in use!")
        episodes = list(self.setup_episodes(self.fold))
        if self.opt.get("episodes_randomization_seed", -1) != -1:
            random.Random(self.opt["episodes_randomization_seed"]).shuffle(episodes)
        if self.opt.get("n_shot", -1) != -1:
            episodes = episodes[: self.opt["n_shot"]]
        elif self.opt.get("percent_shot", -1) >= 0:
            episodes = episodes[: int(len(episodes) * self.opt["percent_shot"])]
        return episodes

    def get_id_task_prefix(self) -> str:
        """
        Convenience for setting IDs.
        """
        raise NotImplementedError("Must set ID prefix in downstream task agent")

    def get_agent_type_suffix(self) -> str:
        """
        Convenience for setting IDs.
        """
        raise NotImplementedError("Must set in downstream agent")


######### Agents that dump information from a dataset as gold (explicitly should *not* be used with teachers)
class _TodDataDumpAgent(TodStructuredDataParser):
    """
    For agents which dump data from some dataset, without training/other modifications.

    Implements an "epoch done"

    Member variables assumed to be set in init downstream:
        self.fold
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.epochDone = False
        self.batchsize = opt.get("batchsize", 1)
        self.max_episodes = len(self.episodes)
        if opt.get("num_episodes", 0) > 0:
            self.max_episodes = min(self.max_episodes, opt.get("num_episodes"))
        self.episode_idx = opt.get("batchindex", 0)
        self._setup_next_episode()
        self.round_idx = 0  # for some downstream utt + sysUttAndApiCallAgents.
        if is_distributed():  # cause gotta manually handle
            rank = get_rank()
            chunk_size = ceil(self.max_episodes / num_workers())
            self.episode_idx += rank * chunk_size
            self.max_episodes = min(self.max_episodes, (rank + 1) * chunk_size)

    def _setup_next_episode(self):
        self.epochDone = not self.episode_idx < self.max_episodes
        self.episode = None
        if not self.epochDone:
            self.episode = self.episodes[self.episode_idx]
        self.round_idx = (
            0  # so downstream agents know which round they are in. Update in `act()`
        )

    def epoch_done(self) -> bool:
        return self.epochDone

    def episode_done(self) -> bool:
        """
        This is not actually "episode_done" so much as "we want to signify to the world
        that we have gone past the batch".

        This class should not control whether or not the episode is actually done since
        the TodWorld expects that to come from the User agent.
        """
        return self.epochDone

    def num_episodes(self) -> int:
        return len(self.episodes)

    def reset(self):
        self.episode_idx += self.batchsize
        self._setup_next_episode()


class TodGoalAgent(_TodDataDumpAgent):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.
    """

    def act(self):
        return {
            "text": f"{tod.STANDARD_GOAL}{self.episode.goal_calls_utt}",
            "id": self.id,
            "domain": self.episode.domain,
            "episode_done": False,
        }

    def get_agent_type_suffix(self):
        return "Goal"


class TodApiSchemaAgent(_TodDataDumpAgent):
    def act(self):
        return {
            "text": f"{tod.STANDARD_API_SCHEMAS}{self.episode.api_schemas_utt}",
            "id": self.id,
            "domain": self.episode.domain,
            "episode_done": False,
        }

    def get_agent_type_suffix(self):
        return "ApiSchema"


############# Single Goal + Api Schema Agent
class _EpisodeToSingleGoalProcessor(_TodDataDumpAgent):
    """
    Iterate through all of the goals of a dataset, one by one.

    Slightly different logic than the dump agent since how we count + setup examples for
    an episode are different

    Used as a mixin in the SingleGoal and SingleApiSchema agents below.
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.epochDone = False
        if shared is None:
            self.episodes = self._setup_single_goal_episodes()
        else:
            # Handled fine in _TodDataDumpAgent
            pass

        self.max_episodes = len(self.episodes)
        if opt.get("num_episodes", 0) > 0:
            self.max_episodes = min(self.max_episodes, opt.get("num_episodes"))
        if is_distributed():  # cause gotta manually handle
            rank = get_rank()
            chunk_size = ceil(self.max_episodes / num_workers())
            self.max_episodes = min(self.max_episodes, (rank + 1) * chunk_size)

        self._setup_next_episode()

    def _setup_single_goal_episodes(self) -> List[tod.TodStructuredEpisode]:
        """
        This function assumes that `self.setup_episodes()` has already been called
        prior.

        Based on the `__init__` order of this class, it should be done in
        `TodStructuredDataParser` by this point.
        """
        raw_episodes = self.episodes
        result = []
        for raw in raw_episodes:
            for call in self.filter_goals(raw.goal_calls_machine):
                schema = {}
                for cand in raw.api_schemas_machine:
                    if (
                        cand[tod.STANDARD_API_NAME_SLOT]
                        == call[tod.STANDARD_API_NAME_SLOT]
                    ):
                        schema = cand

                result.append(
                    tod.TodStructuredEpisode(
                        domain=raw.domain,
                        all_domains=raw.all_domains,
                        api_schemas_machine=[schema],
                        goal_calls_machine=[call],
                        rounds=[],
                    )
                )
        return result

    def filter_goals(self, goals):
        """
        Some downstream agents may want to filter the goals.
        """
        return goals


class TodSingleGoalAgent(_EpisodeToSingleGoalProcessor, TodGoalAgent):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    NOTE: If an API schema agent is used, this *must* be used with `TodSingleApiSchemaAgent` since it will be nonsensicle otherwise. Additionally, this agent will not function properly with UserUtt + SystemUttAndApiCall agent, since episodes will not align.
    """

    def get_agent_type_suffix(self):
        return "SingleGoal"


class TodSingleApiSchemaAgent(_EpisodeToSingleGoalProcessor, TodApiSchemaAgent):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    NOTE: Must be used with TodSingleGoalAgent since nonsensicle otherwise. Additionally, this agent will not function properly with UserUtt + SystemUttAndApiCall agent, since episodes will not align.
    """

    def get_agent_type_suffix(self):
        return "SingleApiSchema"


###### User + System  agents


class TodUserUttAgent(_TodDataDumpAgent):
    """
    Hack for getting TOD Metrics in a model-teacher chat (or to get TOD Metrics on a
    ground-truth TOD dataset)

    This class should only ever be used with the model-model chat world which will stop
    upon seeing the '[DONE]' utterance.

    May go out of bounds elsewhere.
    """

    def act(self):
        result = {
            "text": f"{tod.STANDARD_USER_UTTERANCE}{self.episode.rounds[self.round_idx].user_utt}",
            "id": self.id,
            "domain": self.episode.domain,
            "episode_done": False,
        }
        self.round_idx += 1
        return result

    def reset(self):
        super().reset()  # setup next episode
        self.round_idx = 0

    def get_agent_type_suffix(self):
        return "User"


class TodApiCallAndSysUttAgent(_TodDataDumpAgent):
    """
    Hack for getting TOD Metrics in a model-teacher chat (or to get TOD Metrics on a
    ground-truth TOD dataset)

    This class should only ever be used with the model-model chat world which will stop
    upon seeing the '[DONE]' utterance from a user agent.

    May go out of bounds elsewhere.
    """

    def __init__(self, opt: Opt, shared=None):
        # This class represents two "agents" so need to make sure we don't increment episode number (reset) twice
        self.already_reset = False
        self.api_call_turn = True
        super().__init__(opt, shared)

    def act(self):
        self.already_reset = False
        if tod.STANDARD_API_SCHEMAS in self.observation.get("text", ""):
            return {
                "text": tod.STANDARD_API_SCHEMAS,
                "id": self.id,
                "domain": self.episode.domain,
                "episode_down": False,
            }

        if self.api_call_turn:  # comes first, don't iterate round #
            result = {
                "text": f"{tod.STANDARD_CALL}{self.episode.rounds[self.round_idx].api_call_utt}",
                "id": self.id,
                "domain": self.episode.domain,
                "episode_done": False,
            }
        else:
            result = {
                "text": f"{tod.STANDARD_SYSTEM_UTTERANCE}{self.episode.rounds[self.round_idx].sys_utt}",
                "id": self.id,
                "domain": self.episode.domain,
                "episode_done": False,
            }
            self.round_idx += 1

        self.api_call_turn ^= True
        return result

    def reset(self):
        if not self.already_reset:
            super().reset()  # setup next episode
            self.api_call_turn = True
            self.already_reset = True

    def get_agent_type_suffix(self):
        return "System"


class TodApiResponseAgent(_TodDataDumpAgent):
    """
    Hack for getting TOD Metrics in a model-teacher chat (or to get TOD Metrics on a
    ground-truth TOD dataset)

    This class should only ever be used with the model-model chat world which will stop
    upon seeing the '[DONE]' utterance.

    May go out of bounds elsewhere.
    """

    def act(self):
        result = {
            "text": f"{tod.STANDARD_RESP}{self.episode.rounds[self.round_idx].api_resp_utt}",
            "id": self.id,
            "domain": self.episode.domain,
            "episode_done": False,
        }
        self.round_idx += 1
        return result

    def reset(self):
        super().reset()  # setup next episode
        self.round_idx = 0

    def get_agent_type_suffix(self):
        return "ApiResponse"


###### Standalone API agent
class TodStandaloneApiAgent(Agent):
    """
    Trainable agent that saves API calls and responses.

    Use `TodStandaloneApiTeacher` to train this class.
    """

    EMPTY_RESP = {
        "text": tod.STANDARD_RESP,
        "id": "TodStandaloneApiAgent",
        "episode_done": False,
    }

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        group = parser.add_argument_group("TOD Standalone API args")
        group.add_argument(
            "--exact-api-call",
            type=bool,
            default=True,
            help="Validation-time flag. If true, will return '' if exact api call values not found. If false, will pick response from the same intent with similar api parameters (assuming intent is the same when available)",
        )

        group.add_argument(
            "--fail-hard",
            type=bool,
            default=False,
            help="Aids in deugging. Will fail hard if API call not found and '--exact-api-call' is set.",
        )

        group.add_argument(
            "--standalone-api-file",
            type=str,
            default=None,
            help="Path to file holding .pickle of standalone api for validation (will intelligently strip if suffix included, but do not need)",
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = "StandaloneApiAgent"
        file_key = "model_file"
        if self.opt["standalone_api_file"] is not None:
            file_key = "standalone_api_file"
        self.path_base = self.opt[file_key].replace(".pickle", "")
        self.db_path = self.path_base + ".pickle"
        self.exact_api_call = self.opt["exact_api_call"]
        try:
            with (open(self.db_path, "rb")) as openfile:
                self.data = pickle.load(openfile)
                print("Loaded Standalone API data successfully")
        except Exception:
            print(f"No file at {self.db_path}; ASSUMING WE ARE TRAINING")
            self.data = {}
        self.training = False

    def _maybe_filter_prefix(self, text, prefix):
        if prefix in text:
            return text[len(prefix) :].strip()
        return text.strip()

    def act(self):
        if not self.observation["text"].startswith(tod.STANDARD_CALL):
            return self.EMPTY_RESP
        call_text_raw = self.observation["text"]
        # decode then reencode the API call so that we get the API calls in a consistent order
        call_text = SerializationHelpers.api_dict_to_str(
            SerializationHelpers.str_to_api_dict(
                call_text_raw[len(tod.STANDARD_CALL) :]
            )
        )
        if "labels" in self.observation:
            return self._do_train(call_text)
        return self._do_fetch(call_text)

    def _do_train(self, call_text):
        self.training = True
        self.data[call_text] = self.observation["labels"][0]
        return self.EMPTY_RESP

    def _do_fetch(self, call_text):
        if self.exact_api_call:
            if self.opt.get("fail_hard", False):
                resp = self.data[call_text]
            else:
                resp = self.data.get(call_text, tod.STANDARD_RESP)
            return {
                "text": resp,
                "id": self.id,
                "episode_done": False,
            }

        # Not exact case
        best_key = difflib.get_close_matches(call_text, self.data.keys(), 1)
        if len(best_key) == 0:
            return self.EMPTY_RESP
        return {
            "text": self.data.get(best_key[0], tod.STANDARD_RESP),
            "id": self.id,
            "episode_done": False,
        }

    def shutdown(self):
        if self.training:
            with (open(self.db_path, "wb")) as openfile:
                pickle.dump(self.data, openfile)
                print(f"Dumped output to {self.db_path}")
            with open(self.path_base + ".opt", "w") as f:
                json.dump(self.opt, f)


######### Default dummy agents


class TodEmptyApiSchemaAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = "TodEmptyApiSchemaAgent"

    def act(self):
        msg = {
            "id": self.getID(),
            "text": tod.STANDARD_API_SCHEMAS,
            "episode_done": False,
        }
        return Message(msg)


class TodEmptyGoalAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = "TodEmptyGoalAgent"

    def act(self):
        msg = {"id": self.getID(), "text": tod.STANDARD_GOAL, "episode_done": False}
        return Message(msg)


############# Teachers
class SystemTeacher(TodStructuredDataParser, DialogTeacher):
    """
    TOD agent teacher which produces both API calls and NLG responses.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            "--api-schemas",
            type="bool",
            default=False,
            help="Preempt first turn with intents + required/optional parameters as key/value for given domain",
        )
        parser.add_argument(
            "--standalone-api",
            type="bool",
            default=True,
            help="Noop for this agent. Included to make sweeps easier.",
        )
        parser.add_argument(
            "--api-jga-record",
            type=bool,
            default=True,
            help="Should we save jga information per api schema?",
        )
        parser.add_argument(
            "--domain-jga-record",
            type=bool,
            default=False,
            help="Should we save jga information per domain?",
        )
        parser.add_argument(
            "--domain-nlg-record",
            type=bool,
            default=False,
            help="Should we save nlg information per domain?",
        )
        return parser

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        resp = model_response.get("text")
        if not resp:
            return
        if teacher_action["type"] == tod.STANDARD_CALL:
            if resp.startswith(tod.STANDARD_CALL):
                resp = resp[len(tod.STANDARD_CALL) :]
            predicted = SerializationHelpers.str_to_api_dict(resp)
            domains = (
                [teacher_action["domain"]] if self.opt["domain_jga_record"] else []
            )

            metrics = SlotMetrics(
                teacher_slots=teacher_action["slots"],
                predicted_slots=predicted,
                avg_jga_nlg_bleu=True,
                prefixes=domains,
            ).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

            if self.opt["api_jga_record"] and len(teacher_action["slots"]) > 0:
                teacher = teacher_action["slots"]
                slots = list(teacher.keys())
                slots.remove(tod.STANDARD_API_NAME_SLOT)
                api_here = (
                    "api-"
                    + teacher[tod.STANDARD_API_NAME_SLOT]
                    + "--"
                    + "-".join(slots)
                )
                self.metrics.add(f"{api_here}/jga", AverageMetric(teacher == predicted))

        elif teacher_action["type"] == tod.STANDARD_SYSTEM_UTTERANCE:
            domains = (
                [teacher_action["domain"]] if self.opt["domain_nlg_record"] else []
            )
            metrics = NlgMetrics(
                guess=resp,
                labels=labels,
                prefixes=domains,
                avg_jga_nlg_bleu=True,
            ).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

    def setup_data(self, fold):
        for episode in self.generate_episodes():
            if self.opt.get("api_schemas"):
                schemas = episode.api_schemas_utt
            else:
                schemas = ""
            yield {
                "text": f"{tod.STANDARD_API_SCHEMAS}{schemas}",
                "label": f"{tod.STANDARD_API_SCHEMAS}",
                "domain": episode.domain,
                "type": tod.STANDARD_API_SCHEMAS,
                "slots": {},
            }, True
            for r in episode.rounds:
                yield {
                    "text": f"{tod.STANDARD_USER_UTTERANCE}{r.user_utt}",
                    "label": f"{tod.STANDARD_CALL}{r.api_call_utt}",
                    "domain": episode.domain,
                    "type": tod.STANDARD_CALL,
                    "slots": r.api_call_machine,
                }, False
                yield {
                    "text": f"{tod.STANDARD_RESP}{r.api_resp_utt}",
                    "label": f"{tod.STANDARD_SYSTEM_UTTERANCE}{r.sys_utt}",
                    "domain": episode.domain,
                    "slots": r.api_resp_machine,
                    "type": tod.STANDARD_SYSTEM_UTTERANCE,
                }, False

    def get_agent_type_suffix(self):
        return "SystemTeacher"


class UserSimulatorTeacher(TodStructuredDataParser, DialogTeacher):
    """
    Teacher that tries to simulate user actions (ie, switches text/labels between USER
    and SYSTEM)
    """

    def setup_data(self, fold):
        for episode in self.generate_episodes():
            if len(episode.rounds) < 1:
                continue
            yield {
                "text": f"{tod.STANDARD_GOAL}{episode.goal_calls_utt}",
                "label": f"{tod.STANDARD_USER_UTTERANCE}{episode.rounds[0].user_utt}",
                "domain": episode.domain,
                "type": tod.STANDARD_USER_UTTERANCE,
            }, True
            for i, r in enumerate(episode.rounds):
                if i == len(episode.rounds) - 1:
                    continue
                yield {
                    "text": f"{tod.STANDARD_SYSTEM_UTTERANCE}{r.sys_utt}",
                    "label": f"{tod.STANDARD_USER_UTTERANCE}{episode.rounds[i+1].user_utt}",
                    "domain": episode.domain,
                    "type": tod.STANDARD_USER_UTTERANCE,
                    "slots": {},  # slots in agent/user turns are meaningless
                }, False

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        resp = model_response.get("text")
        if not resp:
            return
        if teacher_action["type"] == tod.STANDARD_RESP:
            if resp.startswith(tod.STANDARD_RESP):
                resp = resp[len(tod.STANDARD_RESP) :]
            predicted = SerializationHelpers.str_to_api_dict(resp)

            metrics = SlotMetrics(teacher_action["slots"], predicted).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

        elif teacher_action["type"] == tod.STANDARD_USER_UTTERANCE:
            metrics = NlgMetrics(resp, labels).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

    def get_agent_type_suffix(self):
        return "UserSimulatorTeacher"


class TodStandaloneApiTeacher(TodStructuredDataParser, DialogTeacher):
    """
    Use this to generate a database for `TodStandaloneApiAgent`.

    (Set this as the teacher with `TodStandaloneApiAgent` as the agent.)
    """

    def setup_data(self, fold):
        # As a default, just put everything in
        for fold_overwrite in ["train", "valid", "test"]:
            for episode in self.setup_episodes(fold_overwrite):
                first = True
                for r in episode.rounds:
                    if len(r.api_call_machine) > 0:
                        yield {
                            "text": f"{tod.STANDARD_CALL}{r.api_call_utt}",
                            "label": f"{tod.STANDARD_RESP}{r.api_resp_utt}",
                            "id": self.id,
                            "domain": episode.domain,
                        }, first
                        first = False

    def get_agent_type_suffix(self):
        return "StandaloneApiTeacher"
