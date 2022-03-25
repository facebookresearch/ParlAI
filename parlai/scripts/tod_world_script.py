#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base script for running TOD model-model chats.

For example, to extract gold ground truth data from the holdout version of Google SGD, run

```
python -u -m parlai.scripts.tod_world_script --api-schema-grounding-model parlai.tasks.google_sgd_simulation_splits.agents:OutDomainApiSchemaAgent --goal-grounding-model parlai.tasks.google_sgd_simulation_splits.agents:OutDomainGoalAgent --user-model parlai.tasks.google_sgd_simulation_splits.agents:OutDomainUserUttAgent --system-model parlai.tasks.google_sgd_simulation_splits.agents:OutDomainApiCallAndSysUttAgent --api-resp-model parlai.tasks.google_sgd_simulation_splits.agents:OutDomainApiResponseAgent -dt valid --num-episodes -1 --episodes-randomization-seed 42 --world-logs gold-valid
```

This file handles
1. Script param setup, including that used for loading agents which may have their own parameters
2. Running the world (including handling batching, until num episodes or length of epoch has been met).
3. File I/O for both reports (for metrics) and conversation logs + logic for displaying prints
"""

import json
from copy import deepcopy
from shutil import copyfile
import os

import parlai.utils.logging as logging
import parlai.core.tod.tod_world as tod_world
import parlai.core.tod.tod_agents as tod_world_agents
from parlai.core.agents import create_agent
from parlai.core.metrics import dict_report, aggregate_unnamed_reports
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.distributed import (
    is_primary_worker,
    all_gather_list,
    is_distributed,
    get_rank,
    sync_object,
    num_workers,
)
from parlai.utils.io import PathManager
from parlai.utils.misc import TimeLogger, nice_report
from parlai.utils.world_logging import WorldLogger


class TodWorldLogger(WorldLogger):
    """
    WorldLogger has most of what we need.

    We could if-class this logic in it directly, but inheritence + override here is
    neater.
    """

    def _is_batch_world(self, world):
        return True

    def _log_batch(self, world):
        batch_acts = world.get_batch_acts()
        for i, acts in enumerate(batch_acts):
            acts = [
                act for act in acts if act is not None and "id" in act and "text" in act
            ]
            acts = [
                act
                for act in acts
                if act["id"] != "" and (act["text"] != "" or "Human" in act["id"])
            ]
            if len(acts) > 0:
                self._add_msgs(acts, idx=i)
            if world.episode_done():
                self.reset_world(idx=i)


class TodWorldParser(ParlaiParser):
    def add_extra_args(self, args=None):
        super().add_extra_args(args)
        parsed = vars(self.parse_known_args(args, nohelp=True)[0])
        # Also load extra args options if a file is given.
        if parsed.get("init_opt") is not None:
            try:
                self._load_known_opts(parsed.get("init_opt"), parsed)
            except FileNotFoundError:
                # don't die if -o isn't found here. See comment in second call
                # later on.
                pass
        parsed = self._infer_datapath(parsed)

        partial = Opt(parsed)

        for model in [
            "system_model",
            "user_model",
            "api_schema_grounding_model",
            "goal_grounding_model",
            "api_resp_model",
        ]:
            if (
                model in partial
                and partial[model] is not None
                and len(partial[model]) > 0
            ):
                self.add_model_subargs(partial[model], partial)

        for model_file_prefix in ["system", "user"]:
            key = model_file_prefix + "_model_file"
            if key in partial and partial[key] and len(partial[key]) > 0:
                model_name = self._get_model_name_from_model_file(key, partial)
                self.add_model_subargs(model_name, partial)

    def _get_model_name_from_model_file(self, key, opt):
        """
        Get the model name from either `--model` or `--model-file`.
        """
        # try to get model name from model opt file
        model_file = opt.get(key, None)
        optfile = model_file + ".opt"
        new_opt = Opt.load(optfile)
        model = new_opt.get("model", None)
        return model


@register_script("tod_world_script")
class TodWorldScript(ParlaiScript):
    @classmethod
    def setup_tod_args(cls, parser: ParlaiParser):
        tod_args = parser.add_argument_group(
            "TOD World Script Agent arguments. NOTE: If there are issues with invoking downstream opts of agents specified here sometimes you will have more luck with `python -u -m parlai.scripts.tod_world_script` than `parlai tod_world_script`."
        )
        tod_args.add_argument(
            "--system-model-file",
            default="",
            help="Define the system model for the chat. Exactly one of this or system-model must be specified",
        )

        tod_args.add_argument(
            "--system-model",
            default="",
            help="Define the system agent for the chat. Exactly one of this or system-model-file must be specified",
        )

        tod_args.add_argument(
            "--user-model-file",
            default="",
            help="Define the user model for the chat. Exactly one of this user-model must be specified. Currently assumed to be the API Call creation agent as well.",
        )

        tod_args.add_argument(
            "--user-model",
            default="",
            help="Define the user agent for the chat. Exactly one of this or user-model-file must be specified. Currently assumed to be the API Call creation agent as well.",
        )

        tod_args.add_argument(
            "--api-resp-model",
            default="",
            help="Agent used for defining API response values",
        )

        tod_args.add_argument(
            "--api-schema-grounding-model",
            default="",
            help="Agent used in first turn to grounding api call/response agents with api schemas. Will use EmptyApiSchemaAgent if both this and `--api-schemas` not set.",
        )

        tod_args.add_argument(
            "--goal-grounding-model",
            default="",
            help="Agent used in first turn to grounding user agent with goal. Will use EmptyGoalAgent if not set",
        )

        tod_args.add_argument(
            "--api-schemas",
            default=None,
            help="If set and `--api-schema-grounding-model` is empty, will infer `--api-schema-grounding-model` based on this and a regex on `--goal-grounding-model`. If you run into issues with parsing order of opts using this flag, just switch to `--api-schema-grounding-model`.",
        )

    @classmethod
    def setup_args(cls):
        # Use default parlai args for logging + the like, but don't need model args since we specify those manually via command-line
        parser = TodWorldParser(
            True, False, "World for chatting with the TOD conversation structure"
        )
        # Following params are same as the `eval_model` script
        parser.add_argument(
            "--report-filename",
            type=str,
            help="Saves a json file of the evaluation report either as an "
            'extension to the model-file (if begins with a ".") or a whole '
            "file path. Set to the empty string to not save at all.",
        )
        parser.add_argument(
            "--world-logs",
            type=str,
            help="Saves a jsonl file containing all of the task examples and "
            "model replies.",
        )
        parser.add_argument(
            "--save-format",
            type=str,
            default="conversations",
            choices=["conversations", "parlai"],
        )
        parser.add_argument(
            "--num-episodes",
            type=int,
            default=10,
            help="Number of episodes to display. Set to -1 for infinity or the number of examples of the first agent with a non-unlimited number of episodes in the world.",
        )
        parser.add_argument("-d", "--display-examples", type="bool", default=False)
        parser.add_argument("-ltim", "--log-every-n-secs", type=float, default=10)
        TodWorldLogger.add_cmdline_args(parser)

        # Following are specific to TOD World
        parser.add_argument(
            "--max-turns",
            type=int,
            default=30,
            help="The max number of full turns before chat ends, excluding prompting",
        )
        TodWorldScript.setup_tod_args(parser)

        return parser

    def _get_file_or_model_specifiable_agent(self, prefix, opt):
        if len(opt.get(f"{prefix}_model_file", "")) > 0:
            if len(opt.get(f"{prefix}_model", "")) > 0:
                raise KeyError(
                    "Both `--{prefix}-model-file` and `--{prefix}-model` specified. Exactly one should be."
                )
            model = self._make_agent(
                opt,
                f"{prefix}_model_file",
                requireModelExists=True,
                opt_key="model_file",
            )
        elif len(opt.get(f"{prefix}_model", "")) > 0:
            model = self._make_agent(opt, f"{prefix}_model", "")
        else:
            raise KeyError(
                f"Both `--{prefix}-model-file` and `--{prefix}-model` specified. Neither currently set."
            )
        return model

    def _get_model_or_default_agent(self, opt, key, default_class):
        if len(opt.get(key, "")) > 0:
            return self._make_agent(opt, key)
        return default_class(opt)

    def _get_tod_agents(self, opt: Opt):
        agents = [None] * tod_world.AGENT_COUNT

        agents[tod_world.USER_UTT_IDX] = self._get_file_or_model_specifiable_agent(
            "user", opt
        )

        # Get system agent, nothing that api call agent currently same as system agent
        system_model = self._get_file_or_model_specifiable_agent("system", opt)
        agents[tod_world.SYSTEM_UTT_IDX] = system_model
        agents[tod_world.API_CALL_IDX] = system_model

        agents[tod_world.API_RESP_IDX] = self._make_agent(opt, "api_resp_model")
        agents[tod_world.GOAL_GROUNDING_IDX] = self._get_model_or_default_agent(
            opt, "goal_grounding_model", tod_world_agents.EmptyGoalAgent
        )

        if "api_schema_grounding_model" not in opt and "api_schemas" in opt:
            opt["api_schema_grounding_model"] = opt.get(
                "goal_grounding_model", ""
            ).replace("Goal", "ApiSchema")

        agents[tod_world.API_SCHEMA_GROUNDING_IDX] = self._get_model_or_default_agent(
            opt, "api_schema_grounding_model", tod_world_agents.EmptyApiSchemaAgent
        )

        return agents

    def _make_agent(self, opt_raw, name, requireModelExists=False, opt_key="model"):
        """
        Hack.

        `create_agent` expects opt[`model`] to specify the model type and we're
        specifying multiple models from other opt arguments (ex.
        `system_model`/`user_model` etc), so this swaps it in.
        """
        opt = deepcopy(opt_raw)
        opt[opt_key] = opt[name]
        print(opt_key, name)
        return create_agent(opt, requireModelExists)

    def _run_episode(self, opt, world, world_logger):
        while not world.episode_done():
            world.parley()
            world_logger.log(world)

            if opt["display_examples"]:
                logging.info(world.display())

        if opt["display_examples"]:
            logging.info("-- end of episode --")

        world.reset()
        world_logger.reset_world()  # flush this episode
        return zip(world.get_last_batch_goals(), world.get_last_batch_episode_metrics())

    def _save_outputs(self, opt, world, logger, episode_metrics):
        if is_distributed():  # flatten everything intelligently if need be
            world_report = aggregate_unnamed_reports(all_gather_list(world.report()))
            episode_metrics_unflattened = all_gather_list(episode_metrics)
            flattened = []
            for rank_elem in episode_metrics_unflattened:
                for elem in rank_elem:
                    flattened.append(elem)
            episode_metrics = flattened
        else:
            world_report = world.report()
        logging.report("Final report:\n" + nice_report(world_report))

        report = dict_report(world_report)

        def get_episode_report(goal, episode_metric):
            metrics_dict = dict_report(episode_metric.report())
            metrics_dict["goal"] = goal
            return metrics_dict

        report["tod_metrics"] = [get_episode_report(g, e) for g, e in episode_metrics]

        if "report_filename" in opt and opt["report_filename"] is not None:
            if len(world_report) == 0:
                logging.warning("Report is empty; not saving report")

            report_fname = f"{opt['report_filename']}.json"
            # Save report
            if not is_distributed() or is_primary_worker():
                with PathManager.open(report_fname, "w") as f:
                    logging.info(f"Saving model report to {report_fname}")
                    json.dump({"opt": opt, "report": report}, f, indent=4)
                    f.write("\n")  # for jq

        if "world_logs" in opt and opt["world_logs"] is not None:
            if is_distributed():  # Save separately, then aggregate together
                rank = get_rank()
                log_outfile_part = (
                    f"{opt['world_logs']}_{opt['save_format']}_{rank}.jsonl"
                )
                logger.write(log_outfile_part, world, file_format=opt["save_format"])
                sync_object(None)
                if is_primary_worker():
                    log_outfile = f"{opt['world_logs']}_{opt['save_format']}.jsonl"
                    log_outfile_metadata = (
                        f"{opt['world_logs']}_{opt['save_format']}.metadata"
                    )
                    with open(log_outfile, "w+") as outfile:
                        for rank in range(num_workers()):
                            log_outfile_part = (
                                f"{opt['world_logs']}_{opt['save_format']}_{rank}.jsonl"
                            )
                            with open(log_outfile_part) as infile:
                                for line in infile:
                                    json_blob = json.loads(line.strip())
                                    if (
                                        len(json_blob["dialog"]) < 2
                                    ):  # skip when we don't have generation
                                        continue
                                    json_blob["metadata_path"] = log_outfile_metadata
                                    outfile.write(json.dumps(json_blob))
                                    outfile.write("\n")
                            log_output_part_metadata = f"{opt['world_logs']}_{opt['save_format']}_{rank}.metadata"
                            if rank == 0:
                                copyfile(
                                    log_output_part_metadata, log_outfile_metadata
                                ),
                            os.remove(log_outfile_part)
                            os.remove(log_output_part_metadata)
            else:
                log_outfile = f"{opt['world_logs']}_{opt['save_format']}.jsonl"
                logger.write(log_outfile, world, file_format=opt["save_format"])

        return report

    def _setup_world(self):
        # setup world, manually finaggling necessary opt info as needed
        self.opt["task"] = "TodWorld"
        world = tod_world.TodWorld(self.opt, agents=self._get_tod_agents(self.opt))
        return world

    def run(self):
        opt = self.opt

        world = self._setup_world()
        logger = TodWorldLogger(opt)

        # set up logging
        log_every_n_secs = opt.get("log_every_n_secs", -1)
        if log_every_n_secs <= 0:
            log_every_n_secs = float("inf")
        log_time = TimeLogger()

        # episode counter
        max_episodes = opt.get("num_episodes", -1)
        if max_episodes < 0:
            max_episodes = float("inf")
        world_num_episodes = world.num_episodes()
        if world_num_episodes > 0:
            max_episodes = min(max_episodes, world_num_episodes)

        ep_count = 0
        episode_metrics = []
        while not world.epoch_done() and ep_count < max_episodes:
            episode_metrics.extend(self._run_episode(opt, world, logger))
            ep_count += opt.get("batchsize", 1)
            if log_time.time() > log_every_n_secs:
                report = world.report()
                text, report = log_time.log(ep_count, max_episodes, report)
                logging.info(text)

        return self._save_outputs(opt, world, logger, episode_metrics)


if __name__ == "__main__":
    TodWorldScript.main()
