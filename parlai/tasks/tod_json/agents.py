#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.utils.data import DatatypeHelper

import parlai.core.tod.tod_agents as tod_agents
import parlai.core.tod.tod_core as tod

import json
import os

PREFIXES = [
    tod.STANDARD_USER_UTTERANCE,
    tod.STANDARD_CALL,
    tod.STANDARD_RESP,
    tod.STANDARD_SYSTEM_UTTERANCE,
]

PREFIXES_PREEMPT = [
    tod.STANDARD_API_SCHEMAS,
    tod.STANDARD_API_SCHEMAS,
    tod.STANDARD_API_SCHEMAS,
    tod.STANDARD_GOAL,
]


class JsonTodParser(tod_agents.TodStructuredDataParser):
    """
    This module provides access to data in the TOD conversations format.

    See core/tod.py for more info about the format.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group("Tod Json Task Arguments")
        agent.add_argument(
            "-jfdp",
            "--jsonfile-datapath",
            type=str,
            help="Data file. (Assumed to be in .jsonl)",
        )
        agent.add_argument(
            "-tmdp",
            "--tod-metrics-datapath",
            type=str,
            default=None,
            help="Filter which examples to use from a report including per-turn tod metrics",
        )
        agent.add_argument(
            "-f-agh",
            "--filter-all-goals-hit",
            type=bool,
            default=False,
            help="Filter episodes by all-goals-hit metric being 1. Assumes `tod-metrics-datapath` is set.",
        )
        agent.add_argument(
            "--split-to-folds",
            type=bool,
            default=True,
            help="Use all data or split into 8:1:1 fold",
        )
        agent.add_argument(
            "--split-folds-seed", type=int, default=42, help="Seed for the fold split"
        )
        agent.add_argument(
            "--fail-hard",
            type=bool,
            default=False,
            help="Raise errors if formatting not correct",
        )
        return parser

    def __init__(self, opt, shared=None):
        if not opt.get("jsonfile_datapath"):
            raise RuntimeError("jsonfile_datapath not specified")
        if not hasattr(self, "opt"):
            self.opt = opt
        self.opt["datafile"] = opt["jsonfile_datapath"]
        self.fold = self.opt["datatype"]  # don't care
        # Truncate datafile to just the immediate enclosing folder name and file name
        dirname, basename = os.path.split(self.opt["datafile"])
        self.id = os.path.join(os.path.split(dirname)[1], basename)
        super().__init__(opt, shared)

    def _process_line(self, line):
        blob = json.loads(line)
        if "dialog" not in blob or len(blob["dialog"]) < 1:
            return None
        rounds = []
        for raw_round in blob["dialog"][1:]:
            if "prefix_stripped_text" not in raw_round[0]:
                for i in range(len(raw_round)):
                    if (
                        PREFIXES[i] not in raw_round[i]['text']
                        and self.opt["fail_hard"]
                    ):
                        raise RuntimeError(
                            f"Missing prefix `{PREFIXES[i]}` before turn of text: `{raw_round[i]}`"
                        )
                    raw_round[i]["prefix_stripped_text"] = raw_round[i].get(
                        "text", PREFIXES[i]
                    )[len(PREFIXES[i]) :]
            if len(raw_round) != 4:
                if raw_round[0]["prefix_stripped_text"] != tod.STANDARD_DONE:
                    return None  # misformatted convo, don't learn this.
                break  # TodStructuredEpisode will add in [DONE]
            api_call_machine = tod.SerializationHelpers.str_to_api_dict(
                raw_round[1]["prefix_stripped_text"]
            )
            if (
                len(api_call_machine) > 0
                and tod.STANDARD_API_NAME_SLOT not in api_call_machine
            ):
                raise RuntimeError(
                    f"Trying to make API call without `{tod.STANDARD_API_NAME_SLOT}`. Call is: `{raw_round[1]['text']}`"
                )
            r = tod.TodStructuredRound(
                user_utt=raw_round[0]["prefix_stripped_text"],
                api_call_machine=api_call_machine,
                api_resp_machine=tod.SerializationHelpers.str_to_api_dict(
                    raw_round[2]["prefix_stripped_text"]
                ),
                sys_utt=raw_round[3]["prefix_stripped_text"],
            )
            rounds.append(r)
        preempt_round = blob["dialog"][0]
        if len(preempt_round) != 4:
            return None
        for i in range(len(preempt_round)):
            if "prefix_stripped_text" not in preempt_round[i]:
                preempt_round[i]["prefix_stripped_text"] = preempt_round[i].get(
                    "text", PREFIXES_PREEMPT[i]
                )[len(PREFIXES_PREEMPT[i]) :]

        episode = tod.TodStructuredEpisode(
            domain=preempt_round[0].get("domain", ""),
            api_schemas_machine=tod.SerializationHelpers.str_to_api_schemas(
                preempt_round[0].get("prefix_stripped_text", "")
            ),
            goal_calls_machine=tod.SerializationHelpers.str_to_goals(
                preempt_round[3].get("prefix_stripped_text")
            ),
            rounds=rounds,
        )
        return episode

    def _get_right_file(self, filepath):
        """
        Part of the code for loading worlds or scripts does a regex of "_" to "-" when
        loading agents.

        Not a great hack but suffices
        """
        try:
            with open(filepath) as _:
                pass
            return filepath
        except FileNotFoundError:
            with open(filepath.replace("-", "_")) as _:
                return filepath.replace("-", "_")
        raise FileNotFoundError(
            f"Note: tod_json breaks if path contains mixed '-' and '_' in path, due to opt + agent loading wonkiness. Please try renaming the file. Filename: {filepath}"
        )

    def setup_episodes(self, fold):
        result = []
        if self.opt["tod_metrics_datapath"] is not None:
            with open(self._get_right_file(self.opt["tod_metrics_datapath"])) as f:
                report_data = json.load(f)
                tod_metrics = report_data["report"]["tod_metrics"]

        lines_to_process = []
        with open(self._get_right_file(self.opt["datafile"]), "r") as f:
            result = []
            for i, line in enumerate(f.readlines()):
                if (
                    self.opt["filter_all_goals_hit"]
                    and tod_metrics[i]["all_goals_hit"] < 0.5
                ):
                    continue
                if line:
                    lines_to_process.append(line)

        if self.opt["split_to_folds"]:
            lines_to_process = DatatypeHelper.split_data_by_fold(
                fold, lines_to_process, 0.8, 0.1, 0.1, self.opt["split_folds_seed"]
            )

        for line in lines_to_process:
            processed = self._process_line(line)
            if processed is not None:
                result.append(processed)
        return result

    def get_id_task_prefix(self):
        return (
            "TodJson_#"
            + os.path.basename(self.opt["jsonfile_datapath"]).split(".")[0]
            + "#"
        )


class SystemTeacher(JsonTodParser, tod_agents.TodSystemTeacher):
    pass


class DefaultTeacher(SystemTeacher):
    pass


class UserSimulatorTeacher(JsonTodParser, tod_agents.TodUserSimulatorTeacher):
    pass


## We define the following as a quick way to spot-check a file using TodWorldScript.
# Use
#     python ~/ParlAI/parlai/scripts/tod_world_script.py --system-model parlai.tasks.tod_json.agents:ApiCallAndSysUttAgent --user-model parlai.tasks.tod_json.agents:UserUttAgent --api-resp-model parlai.tasks.tod_json.agents:ApiResponseAgent --goal-grounding-model parlai.tasks.tod_json.agents:GoalAgent --api-schema-grounding-model parlai.tasks.tod_json.agents:ApiSchemaAgent --display-examples true --jsonfile-datapath=<INSERT PATH HERE>
#
# Note: may be relevant to set `split_to_folds`to `False` depending on usage
class GoalAgent(JsonTodParser, tod_agents.TodGoalAgent):
    pass


class ApiSchemaAgent(JsonTodParser, tod_agents.TodApiSchemaAgent):
    pass


class UserUttAgent(JsonTodParser, tod_agents.TodUserUttAgent):
    pass


class ApiCallAndSysUttAgent(JsonTodParser, tod_agents.TodApiCallAndSysUttAgent):
    pass


class ApiResponseAgent(JsonTodParser, tod_agents.TodApiResponseAgent):
    pass
