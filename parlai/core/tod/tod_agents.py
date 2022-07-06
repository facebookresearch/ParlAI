#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agents (used for dumping data) and Teachers (for training models) related to the TOD
conversation setup.

As a convention, agents and teachers that are inheritable are prefixed with "Tod"
whereas those that can be used as-is are not. Similarly, classes and functions that do
not need to be exposed outside of this file are prefixed with a single underscore ('_')
"""

from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.data import DatatypeHelper
from parlai.utils.distributed import is_distributed, get_rank, num_workers

import parlai.core.tod.tod_core as tod
from parlai.core.tod.tod_core import SerializationHelpers
from parlai.core.tod.teacher_metrics import SlotMetrics, NlgMetrics

from typing import Optional, List
import json
import pickle
import difflib
import random
from math import ceil


######### Agents that dump information from a dataset; base classes
class TodStructuredDataParser(Agent):
    """
    Base class that specifies intermediate representations for Tod conversations.

    Inherit from this class and implement `setup_episodes()` to implement the intermediate representation for a specific dataset. Use multiple inheritence with classes that implement an `act()` below to use.

    For example, if we have a `MyDataset_DataParser(TodStructuredDataParser)` and wanted to make a teacher to train a model to generate User Utterances based on a goal prompt, we would do so by defining `class MyDatasetUserSimulatorTeacher(MyDataset_DataParser, TodUserSimulatorTeacher)`.
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
        self.id = self.get_id_task_prefix() + "_" + self._get_agent_type_suffix()
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
        raise NotImplementedError(
            "Must have method for generating an episode. Must be set in downstream Parser for a given task"
        )

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
        raise NotImplementedError(
            "Must set ID prefix in downstream task agent. Must be set in downsream Parser for a given task"
        )

    def _get_agent_type_suffix(self) -> str:
        """
        Convenience for setting IDs.
        """
        raise NotImplementedError(
            "Must set in downstream agent within `tod_agents`. If you see this error, something is wrong with TOD Infrastructure"
        )


######### Agents that dump information from a dataset as gold (explicitly should *not* be used with teachers)
class _TodDataDumpAgent(TodStructuredDataParser):
    """
    For agents which dump data from some dataset, without training/other modifications.

    Since we have to deal with batching inside of agents (as per ParlAI convention for
    non-generative agents), this does so while also implementing an "epoch done" to
    denote elements in a batch that are past the end of the epoch.
    """

    def __init__(self, opt: Opt, shared=None):
        if not hasattr(self, "fold"):
            self.fold = DatatypeHelper.fold(opt["datatype"])
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
        self.epochDone = self.episode_idx >= self.max_episodes
        self.episode = None
        if not self.epochDone:
            self.episode = self.episodes[self.episode_idx]
        self.round_idx = (
            0  # so downstream agents know which round they are in. Update in `act()`
        )

    def epoch_done(self) -> bool:
        return self.epochDone

    def episode_done(self) -> bool:
        raise RuntimeError("Must be defined in downstream agent")

    def num_episodes(self) -> int:
        return len(self.episodes)

    def reset(self):
        self.episode_idx += self.batchsize
        self._setup_next_episode()


class TodGoalAgent(_TodDataDumpAgent):
    """
    Use as a mixin with a dataset parser class that includes `generate_episodes()` of
    TodStructuredDataParser.

    Dumps out all goal calls from an episode.
    """

    def act(self):
        return {
            "text": f"{tod.STANDARD_GOAL}{self.episode.goal_calls_utt}",
            "id": self.id,
            "domain": self.episode.domain,
            "episode_done": False,
        }

    def _get_agent_type_suffix(self):
        return "Goal"

    def episode_done(self) -> bool:
        # done if end of batch; should never end conversation otherwise
        return self.epoch_done()


class TodApiSchemaAgent(_TodDataDumpAgent):
    """
    Use as a mixin with a dataset parser class that includes `generate_episodes()` of
    TodStructuredDataParser.

    Dumps out api schemas associated with an episode, based on what is manually set in
    the dataset parser.
    """

    def act(self):
        return {
            "text": f"{tod.STANDARD_API_SCHEMAS}{self.episode.api_schemas_utt}",
            "id": self.id,
            "domain": self.episode.domain,
            "episode_done": False,
        }

    def _get_agent_type_suffix(self):
        return "ApiSchema"

    def episode_done(self) -> bool:
        # done if end of batch; should never end conversation otherwise
        return self.epoch_done()


############# Single Goal + Api Schema Agent
class _EpisodeToSingleGoalProcessor(_TodDataDumpAgent):
    """
    Iterate through all of the goals of a dataset, one by one.

    Slightly different logic than the dump agent since how we count + setup examples for
    an episode are different

    Used as a mixin in the SingleGoal and SingleApiSchema agents below.

    This class exposes a `filter_goals()` function that can be overridden by downstream agents.
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
                        api_schemas_machine=[schema],
                        goal_calls_machine=[call],
                        rounds=[],
                    )
                )
        return result

    def filter_goals(self, goals):
        """
        Some downstream agents may want to filter the goals.

        Override this if so.
        """
        return goals


class TodSingleGoalAgent(_EpisodeToSingleGoalProcessor, TodGoalAgent):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    Takes goals of an episode and splits them into single versions. (That is, if an episode has 3 goal API calls, this makes it such that those 3 goal API calls become the grounding for 3 separate episodes.)

    NOTE: If an API schema agent is used, this *must* be used with `TodSingleApiSchemaAgent` since it will be nonsensicle otherwise. Additionally, this agent will not function properly with UserUtt + SystemUttAndApiCall agent, since episodes will not align.
    """

    def _get_agent_type_suffix(self):
        return "SingleGoal"


class TodSingleApiSchemaAgent(_EpisodeToSingleGoalProcessor, TodApiSchemaAgent):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    Takes the schema provided for an episode and filters these to match the single Goal provided by TodSingelGoalAgent.

    NOTE: Must be used with TodSingleGoalAgent since nonsensicle otherwise. Additionally, this agent will not function properly with UserUtt + SystemUttAndApiCall agent, since episodes will not align.
    """

    def _get_agent_type_suffix(self):
        return "SingleApiSchema"


###### Agents used for calculating TOD World Metrics based on a dataset. See `tod_world_script` or `parlai/projects/tod_simulator/` for examples.
class TodUserUttAgent(_TodDataDumpAgent):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    Agent provided as a convenience to run TOD World script code on a dataset without having to write too much code to do so. (Ex. for a quick way to dump data to a `.jsonl` file for generating data for ACUTE or to generate a report file of metrics from TodWorld script.)

    This represents the "User" agent.

    This class should only ever be used with the model-model chat world which will stop
    upon seeing the '[DONE]' utterance; may go out of bounds otherwise.
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

    def _get_agent_type_suffix(self):
        return "User"

    def episode_done(self) -> bool:
        return self.epoch_done() or self.round_idx >= len(self.episode.rounds)


class TodApiCallAndSysUttAgent(_TodDataDumpAgent):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    Agent provided as a convenience to run TOD World script code on a dataset without having to write too much code to do so. (Ex. for a quick way to dump data to a `.jsonl` file for generating data for ACUTE or to generate a report file of metrics from TodWorld script.)

    This class represents the System and will generate both API Calls and System Utterances.

    This class should only ever be used with the model-model chat world which will stop
    upon seeing the '[DONE]' utterance; may go out of bounds otherwise.
    """

    def __init__(self, opt: Opt, shared=None):
        # This class will have `act()` called on it twice per round — once for API call and once for NLG — so need to make sure we don't increment episode number (reset) prematurely; use the `already_reset` flag for this.
        self.already_reset = False
        self.api_call_turn = True
        super().__init__(opt, shared)

    def act(self):
        self.already_reset = False
        if tod.STANDARD_API_SCHEMAS in self.observation.get("text", ""):
            return {
                "text": tod.STANDARD_API_SCHEMAS,  # Default convention for the first turn
                "id": self.id,
                "domain": self.episode.domain,
                "episode_done": False,
            }

        if self.api_call_turn:  # comes first, don't iterate round #
            result = {
                "text": f"{tod.STANDARD_CALL}{self.episode.rounds[self.round_idx].api_call_utt}",
                "id": self.id,
                "domain": self.episode.domain,
                "episode_done": False,
            }
            self.api_call_turn = False
        else:
            result = {
                "text": f"{tod.STANDARD_SYSTEM_UTTERANCE}{self.episode.rounds[self.round_idx].sys_utt}",
                "id": self.id,
                "domain": self.episode.domain,
                "episode_done": False,
            }
            self.round_idx += 1
            self.api_call_turn = True

        return result

    def reset(self):
        if not self.already_reset:
            super().reset()  # setup next episode
            self.api_call_turn = True
            self.already_reset = True

    def _get_agent_type_suffix(self):
        return "System"

    def episode_done(self) -> bool:
        return self.epoch_done() or self.round_idx >= len(self.episode.rounds)


class TodApiResponseAgent(_TodDataDumpAgent):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    Agent provided as a convenience to run TOD World script code on a dataset without having to write too much code to do so. (Ex. for a quick way to dump data to a `.jsonl` file for generating data for ACUTE or to generate a report file of metrics from TodWorld script.)

    This class represents the Api Response mechanism.

    This class should only ever be used with the model-model chat world which will stop
    upon seeing the '[DONE]' utterance; may go out of bounds otherwise.
    """

    def act(self):
        if tod.STANDARD_API_SCHEMAS in self.observation.get("text", ""):
            return {
                "text": tod.STANDARD_API_SCHEMAS,  # Default convention
                "id": self.id,
                "domain": self.episode.domain,
                "episode_done": False,
            }

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

    def _get_agent_type_suffix(self):
        return "ApiResponse"

    def episode_done(self) -> bool:
        return self.epoch_done() or self.round_idx >= len(self.episode.rounds)


###### Standalone API agent
class StandaloneApiAgent(Agent):
    """
    Trainable agent that saves API calls and responses.

    Use `TodStandaloneApiTeacher` to train this class. For example for a MultiWoz V2.2
    standalone API, use ``` parlai train -t multiwoz_v22:StandaloneApiTeacher -m
    parlai.core.tod.tod_agents:StandaloneApiAgent -eps 4 -mf output ``` to generate the
    `.pickle` file to use.
    """

    EMPTY_RESP = {
        "text": tod.STANDARD_RESP,
        "id": "StandaloneApiAgent",
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
            help="Aids in deugging. Will throw exception if API call not found and '--exact-api-call' is set.",
        )

        group.add_argument(
            "--standalone-api-file",
            type=str,
            default=None,
            help="Path to file holding `.pickle` of standalone api for validation (will intelligently strip if suffix included). If not set, assumes the `model_file` argument will contain the `.pickle` file. ",
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
                self.training = True
                print("Loaded Standalone API data successfully")
            if self.exact_api_call != self.data.get("exact_api_call", True):
                raise RuntimeError(
                    f"Standalone API .pickle file generated with `exact_api_call` of {self.data.get('exact_api_call', False)} but StandaloneApiAgent sets it to {self.exact_api_call}"
                )
        except Exception:
            print(f"No file at {self.db_path}; ASSUMING WE ARE TRAINING")
            self.data = {}
            self.data["exact_api_call"] = self.exact_api_call
            self.training = True

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
        assert self.training is True
        self.data[call_text] = self.observation["labels"][0]
        return self.EMPTY_RESP

    def _do_fetch(self, call_text):
        if self.exact_api_call:
            if self.opt.get("fail_hard", False):
                resp = self.data[call_text]
            else:
                resp = self.data.get(call_text, tod.STANDARD_RESP)
            return {"text": resp, "id": self.id, "episode_done": False}

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


######### Empty agents
class EmptyApiSchemaAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = "EmptyApiSchemaAgent"

    def act(self):
        msg = {
            "id": self.getID(),
            "text": tod.STANDARD_API_SCHEMAS,
            "episode_done": False,
        }
        return Message(msg)


class EmptyGoalAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = "EmptyGoalAgent"

    def act(self):
        msg = {"id": self.getID(), "text": tod.STANDARD_GOAL, "episode_done": False}
        return Message(msg)


############# Teachers
class TodSystemTeacher(TodStructuredDataParser, DialogTeacher):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    TOD agent teacher which produces both API calls and NLG responses.

    First turn is API Schema grounding, which may be a an empty schema.
    Subsequent turns alternate between
        1. User utterance -> API Call
        2. API Response -> System Utterance
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
            "--api-jga-record",
            type=bool,
            default=True,
            help="Breaks out jga into individual api schemas",
        )
        parser.add_argument(
            "--domain-jga-record",
            type=bool,
            default=False,
            help="Breaks out jga into individual domains",
        )
        parser.add_argument(
            "--domain-nlg-record",
            type=bool,
            default=False,
            help="Breaks out nlg into individual domains",
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self._num_examples_cache = sum([len(x.rounds) * 2 + 1 for x in self.episodes])
        self._num_episodes_cache = len(self.episodes)

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
            metrics = NlgMetrics(guess=resp, labels=labels, prefixes=domains).report()
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

    def _get_agent_type_suffix(self):
        return "SystemTeacher"


class TodUserSimulatorTeacher(TodStructuredDataParser, DialogTeacher):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    Teacher that has `Goal->User Utterance` for its first turn, then `System
    Utterance->User Utterance` for all subsequent turns.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        # Manually set number of examples + number of episodes
        self._num_examples_cache = sum([len(x.rounds) for x in self.episodes])
        self._num_episodes_cache = len(self.episodes)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            "--api-schemas",
            type="bool",
            default=False,
            help="Preempt first turn with intents + required/optional parameters as key/value for given domain. NOOP for this teacher, but including to make sweeps easier",
        )
        return parser

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

    def _get_agent_type_suffix(self):
        return "UserSimulatorTeacher"


class TodStandaloneApiTeacher(TodStructuredDataParser, DialogTeacher):
    """
    Use as a mixin with classes that also extend + implement TodStructuredDataParser.

    Use this to generate a database for `StandaloneApiAgent`.

    Set this as the teacher with `StandaloneApiAgent` as the agent. Ex for a MultiWoz
    V2.2 standalone API, use ``` parlai train -t multiwoz_v22:StandaloneApiTeacher -m
    parlai.core.tod.tod_agents:StandaloneApiAgent -eps 4 -mf output ```
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

    def _get_agent_type_suffix(self):
        return "StandaloneApiTeacher"
