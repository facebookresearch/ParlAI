#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script for training, evaluating, and running model model chats for Task Oriented Dialog.

Note that the code below does assume running in a SLURM environment

Use
```
parlai train -t parlai.tasks.google_sgd.agents:StandaloneApiTeacher --standalone-api-file standalone_api_file -m parlai.core.tod.tod_agents:StandaloneApiAgent -mf <OUTPUT_PATH> -eps 5
```
to define STANDALONE_API_FILE_PATH
"""
from parlai.core.script import ParlaiScript, register_script
from parlai.scripts.train_model import TrainModel
from parlai.scripts.distributed_train import (
    DistributedTrain,
    setup_args as train_setup_args,
)
from parlai.scripts.eval_model import EvalModel
from parlai.scripts.distributed_eval import DistributedEval
from parlai.scripts.tod_world_script import TodWorldScript
from parlai.scripts.distributed_tod_world_script import DistributedTodWorldScript
from projects.tod_simulator.scripts.get_passing_only import GetPassingOnlyScript
from projects.tod_simulator.scripts.get_al_samples_for_gsgd import (
    GetAlSamplesForGsgdScript,
    setup_args as setup_al_sample_script_args,
)

import copy
import os
from shutil import copyfile
import json
import random

STANDALONE_API_FILE_PATH = (
    "/checkpoint/<INSERT_HERE>/projects/user_simulator/standalone_api_data/google_sgd"
)


@register_script("tod_distributed_uber_script")
class TodDistributedUberScript(ParlaiScript):
    def get_mm_opt(self, datatype, port, mm_prefix="", mm_suffix=""):
        model_model_opt = {}
        model_model_opt["exact_api_call"] = True
        model_model_opt["api_schemas"] = self.opt["api_schemas"]
        model_model_opt["display_examples"] = False  # we'll see these later
        model_model_opt["episodes_randomization_seed"] = 42
        if "nucleus" in mm_prefix or "nucleus" in mm_suffix:
            model_model_opt["episodes_randomization_seed"] = random.randint(
                0, 1000000000000
            )
        model_model_opt["skip_generation"] = False
        model_model_opt["batchsize"] = 32
        model_model_opt["num_episodes"] = -1
        model_model_opt["datatype"] = datatype
        model_model_opt["log_keep_fields"] = "all"
        # Do this so that the agents get the ride settings
        model_model_opt["override"] = copy.deepcopy(model_model_opt)

        if self.opt["api_schemas"]:
            grounding = "parlai.tasks.google_sgd_simulation_splits.agents:OutDomainSingleApiSchemaAgent"
        else:
            grounding = "parlai.core.tod.tod_agents_and_teachers:TodEmptyApiSchemaAgent"

        model_model_opt["api_schema_grounding_model"] = grounding
        model_model_opt[
            "goal_grounding_model"
        ] = "parlai.tasks.google_sgd_simulation_splits.agents:OutDomainSingleGoalAgent"
        model_model_opt[
            "api_resp_model"
        ] = "parlai.core.tod.tod_agents_and_teachers:TodStandaloneApiAgent"
        model_model_opt["standalone_api_file"] = STANDALONE_API_FILE_PATH

        model_model_opt["system_model_file"] = self.opt["model_file"]
        model_model_opt["user_model_file"] = self.opt["model_file"]
        model_model_opt["distributed_world_size"] = self.opt["distributed_world_size"]
        model_model_opt["ddp_backend"] = self.opt["ddp_backend"]
        model_model_opt["save_format"] = "conversations"
        model_model_opt["log_every_n_seconds"] = 30

        if self.opt["custom_model_model_name"] is None:
            model_model_name = ""
            if "zoo:bart" in self.opt["init_model"]:
                model_model_name = "BartOnlyNoApi"
        else:
            model_model_name = self.opt["custom_model_model_name"]

        model_model_name += "_ApiSchemas" + str(self.opt["api_schemas"])
        lr = self.opt["learningrate"]
        multitask = "Multitask-" + str(self.opt["multitask_weights"][0])
        model_model_name = f"{model_model_name}_{lr}_{multitask}"

        base_path_name = os.path.dirname(self.opt["model_file"])

        model_model_opt["report_filename"] = os.path.join(
            base_path_name, mm_prefix + model_model_name + mm_suffix
        )
        model_model_opt["world_logs"] = os.path.join(
            base_path_name, mm_prefix + model_model_name + mm_suffix
        )
        model_model_opt["port"] = port

        pretty_print = ""

        for key in model_model_opt:
            if key == "override":
                continue
            pretty_print += "--" + key.replace("_", "-")
            pretty_print += " "
            pretty_print += str(model_model_opt[key])
            pretty_print += " "
        self.dist_print(pretty_print)

        return model_model_opt

    @classmethod
    def setup_args(cls):
        parser = train_setup_args()
        # Convenience
        group = parser.add_argument_group("Tod distributed uber script")
        group.add_argument(
            "--custom-model-model-name",
            default=None,
            type=str,
            help="model model name. Set to empty string to derive",
        )
        group.add_argument(
            "--existing-train-files",
            nargs="*",
            required=True,
            help="Path to previous model-model generated (and processed) train files",
        )
        group.add_argument(
            "--rl-level",
            type=int,
            required=True,
            help="Which level of RL are we running? Base pretrain is 0. Using JSON once is 1. 'existing-train-files' and 'existing-al-files' will be truncated to this length.",
        )
        group.add_argument(
            "--skip-al-generation",
            type=bool,
            default=False,
            help="Skip AL geenration, ie for IN-JSON rains",
        )
        group.add_argument(
            "--skip-train-convo-generation",
            type=bool,
            default=False,
            help="Skip train-convo geenration, ie for ones that don't use json",
        )
        group.add_argument(
            "--nucleus-mm-topp",
            type=float,
            nargs="*",
            default=[],
            help="List of coefficients of nucleus used for train data generation",
        )
        group.add_argument(
            "--api-schemas",
            type=bool,
            help="Is this a yes API Schemas or a no API Schemas model?",
        )
        setup_al_sample_script_args(parser)

        return parser

    def run(self):
        ######
        # This is a gigantic function that sets necessary a priori state (notably, if we are in a distributed setting), then generates a bunch of opts, then runs those opts.
        #####

        # Do this manually since we are not yet in a distributed context yet at this piece of code and cannot use distributed.py
        if "SLURM_PROCID" in os.environ:
            self.rank = int(os.environ["SLURM_PROCID"])
        else:
            self.rank = -1

        self.dist_print(f"Using distributed: {self.is_slurm_distributed()}")

        # Setup all of the args first, then see if any issues.
        model_file = self.opt["model_file"]
        base_path_name = os.path.dirname(model_file)
        api_schemas = self.opt["api_schemas"]

        #################### EVAL OPTS
        # Grab necessary default args and set them
        eval_argparser = DistributedEval.setup_args()
        eval_opt = eval_argparser.parse_args(
            [
                "--task",
                "parlai.tasks.google_sgd_simulation_splits.agents:OutDomainSystemTeacher",
                "--model",
                self.opt["model"],
            ]
        )
        # ...but also make sure to use the right settings passed in via run_grid (ie distributed opts)
        for key in eval_opt:
            if key in self.opt:
                eval_opt[key] = self.opt[key]
        # Now reset opts that we need here
        eval_opt[
            "task"
        ] = "parlai.tasks.google_sgd_simulation_splits.agents:OutDomainSystemTeacher"
        eval_opt["model_file"] = model_file
        eval_opt["api_schemas"] = api_schemas
        eval_opt["batchsize"] = 32
        eval_opt["skip_generation"] = False
        eval_opt["report_filename"] = os.path.join(base_path_name, "eval_stats.json")
        eval_opt["datatype"] = "valid"
        eval_opt["port"] = self.opt["port"] + 1
        eval_opt["distributed_world_size"] = self.opt["distributed_world_size"]
        eval_opt["override"] = copy.deepcopy(eval_opt)

        ### OUT User valid eval
        user_eval_opt = copy.deepcopy(eval_opt)
        user_eval_opt[
            "task"
        ] = "parlai.tasks.google_sgd_simulation_splits.agents:OutDomainUserSimulatorTeacher"
        user_eval_opt["override"] = copy.deepcopy(user_eval_opt)
        user_eval_opt["report_filename"] = os.path.join(
            base_path_name, "user_eval_stats.json"
        )
        user_eval_opt["port"] = self.opt["port"] + 2

        ### OUT Test eval (system + user)
        test_eval_opt = copy.deepcopy(eval_opt)
        test_eval_opt["datatype"] = "test"
        test_eval_opt["report_filename"] = os.path.join(
            base_path_name, "test_eval_stats.json"
        )
        test_eval_opt["port"] = self.opt["port"] + 5

        user_test_eval_opt = copy.deepcopy(eval_opt)
        user_test_eval_opt["datatype"] = "test"
        user_test_eval_opt[
            "task"
        ] = "parlai.tasks.google_sgd_simulation_splits.agents:OutDomainUserSimulatorTeacher"
        user_test_eval_opt["report_filename"] = os.path.join(
            base_path_name, "test_user_eval_stats.json"
        )
        user_test_eval_opt["port"] = self.opt["port"] + 6

        ### IN valid eval (system + user)
        in_eval_opt = copy.deepcopy(eval_opt)
        in_eval_opt["datatype"] = "valid"
        in_eval_opt[
            "task"
        ] = "parlai.tasks.google_sgd_simulation_splits.agents:InDomainSystemTeacher"
        in_eval_opt["report_filename"] = os.path.join(
            base_path_name, "in_eval_stats.json"
        )
        in_eval_opt["port"] = self.opt["port"] + 7

        user_in_eval_opt = copy.deepcopy(eval_opt)
        user_in_eval_opt["datatype"] = "valid"
        user_in_eval_opt[
            "task"
        ] = "parlai.tasks.google_sgd_simulation_splits.agents:InDomainUserSimulatorTeacher"
        user_in_eval_opt["report_filename"] = os.path.join(
            base_path_name, "in_user_eval_stats.json"
        )
        user_in_eval_opt["port"] = self.opt["port"] + 8

        eval_model_model_opt = self.get_mm_opt(
            "valid", self.opt["port"] + 3, "mm_eval_"
        )

        train_model_model_opt = self.get_mm_opt(
            "train", self.opt["port"] + 4, mm_prefix="", mm_suffix="_greedy"
        )

        # At this point, everything above is distributed and everything below is non distributed...
        # ...except for nucleus stuff, so we have some of its own code there.
        NEXT_FREE_PORT = 9

        #### For processing everything after train data
        if (
            self.opt["skip_train_convo_generation"]
            and len(self.opt["nucleus_mm_topp"]) > 0
        ):
            raise RuntimeError(
                "Makes no sense to do nucleus generation if we're not making convos"
            )

        mm_report_filename = (
            train_model_model_opt["report_filename"].replace(".json", "") + ".json"
        )
        mm_convo_filename = f"{train_model_model_opt['world_logs']}_{train_model_model_opt['save_format']}.jsonl"

        ### Passing only script args
        passing_only_parser = GetPassingOnlyScript.setup_args()
        passing_only_opt = passing_only_parser.parse_args(
            [
                "--convo-path",
                mm_convo_filename,
                "--report-path",
                mm_report_filename,
                "--print-to-file",
                str(True),
            ]
        )

        # NOTE: Following line needs to be kept in sync with the get_passing_only script
        infile_base = mm_convo_filename.replace(".jsonl", "")
        passing_only_stats_file = infile_base.replace(
            "_conversations", "_processed_stats"
        )
        passing_only_convo_file = infile_base + "_processed.jsonl"
        passing_only_convo_metadata = infile_base + "_processed.metadata"

        # Args for generating nucleus... bit of a mess.
        nucleus_mm_opts = []
        nucleus_passing_only_opts = []
        nucleus_processed_convo_filenames = []
        if len(self.opt["nucleus_mm_topp"]) > 0:
            if self.opt["skip_train_convo_generation"]:
                raise RuntimeError(
                    "Makes no sense to do nucleus generation if we're not making convos"
                )
            for i, topp in enumerate(self.opt["nucleus_mm_topp"]):
                nucleus_opt = self.get_mm_opt(
                    "train", 0, mm_prefix="", mm_suffix=f"-nucleus-{topp}-{i}"
                )
                nucleus_opt["inference"] = "nucleus"
                nucleus_opt["topp"] = topp
                nucleus_opt["override"]["inference"] = "nucleus"
                nucleus_opt["override"]["topp"] = topp
                nucleus_opt["port"] = self.opt["port"] + NEXT_FREE_PORT + i
                nucleus_mm_opts.append(nucleus_opt)
                nucleus_report_filename = (
                    nucleus_opt["report_filename"].replace(".json", "") + ".json"
                )
                nucleus_convo_filename = (
                    f"{nucleus_opt['world_logs']}_{nucleus_opt['save_format']}.jsonl"
                )
                nucleus_passing_only_opt = passing_only_parser.parse_args(
                    [
                        "--convo-path",
                        nucleus_convo_filename,
                        "--report-path",
                        nucleus_report_filename,
                        "--print-to-file",
                        str(True),
                    ]
                )
                nucleus_passing_only_opts.append(nucleus_passing_only_opt)
                nucleus_processed_convo = (
                    nucleus_convo_filename.replace(".jsonl", "") + "_processed.jsonl"
                )
                nucleus_processed_convo_filenames.append(nucleus_processed_convo)

        # Cumulative converseation args
        cumulative_convo = os.path.join(base_path_name, "processed_cumulative.jsonl")
        cumulative_metadata = os.path.join(
            base_path_name, "processed_cumulative.metadata"
        )

        noncumulative_convo = os.path.join(
            base_path_name, "processed_noncumulative.jsonl"
        )
        noncumulative_metadata = os.path.join(
            base_path_name, "processed_noncumulative.metadata"
        )

        ### Active learning script args
        al_sample_opt = {}
        al_sample_opt["find_random_al_samples"] = self.opt["find_random_al_samples"]
        al_sample_opt["processed_stats_section"] = self.opt["processed_stats_section"]
        al_sample_opt["num_apis_to_get"] = self.opt["num_apis_to_get"]
        al_sample_opt["existing_al_files"] = self.opt["existing_al_files"]
        if "datapath" in self.opt:
            al_sample_opt["datapath"] = self.opt["datapath"]
        elif "parlai_home" in self.opt:
            al_sample_opt["datapath"] = self.opt["parlai_home"] + "/data"

        # Manually override what we need to manually override
        al_sample_opt["input_processed_stats"] = passing_only_stats_file
        # To save on deciding if we're going to do cumulative runs together or separately, just do both
        al_sample_noncumulative_opt = copy.deepcopy(al_sample_opt)
        al_sample_noncumulative_opt["cumulative_al"] = False
        al_sample_noncumulative_opt["al_output_file"] = os.path.join(
            base_path_name, "al_noncumulative.json"
        )

        al_sample_cumulative_opt = copy.deepcopy(al_sample_opt)
        al_sample_cumulative_opt["cumulative_al"] = True
        al_sample_cumulative_opt["al_output_file"] = os.path.join(
            base_path_name, "al_cumulative.json"
        )

        ####### Run everything, skipping if we've already finished it
        if not os.path.isfile(model_file + ".test"):
            self.dist_print("RUNNING TRAIN", self.opt)
            train_result = self.train_class()(self.opt).run()
            self.dist_print("train result: ", train_result)

        # Required necessary things
        if not os.path.isfile(eval_model_model_opt["report_filename"] + ".json"):
            self.dist_print("RUNNING VALID MODEL-MODEL", eval_model_model_opt)
            model_model_result = self.tod_world_class()(eval_model_model_opt).run()
            self.dist_print("eval_model_model_result: ", model_model_result)

        if not os.path.isfile(eval_opt["report_filename"]):
            self.dist_print("RUNNING SYSTEM EVAL", eval_opt)
            eval_result = self.eval_class()(eval_opt).run()
            self.dist_print("eval_result: ", eval_result)

        if not os.path.isfile(user_eval_opt["report_filename"]):
            self.dist_print("RUNNING USER SIM EVAL", user_eval_opt)
            eval_result = self.eval_class()(user_eval_opt).run()
            self.dist_print("user_eval_result: ", eval_result)

        # All the other evals
        if not os.path.isfile(in_eval_opt["report_filename"]):
            self.dist_print("RUNNING SYSTEM _IN_ EVAL", in_eval_opt)
            in_eval_result = self.eval_class()(in_eval_opt).run()
            self.dist_print("in_eval_result: ", in_eval_result)

        if not os.path.isfile(user_in_eval_opt["report_filename"]):
            self.dist_print("RUNNING USER SIM _IN_ EVAL", user_in_eval_opt)
            in_eval_result = self.eval_class()(user_in_eval_opt).run()
            self.dist_print("user_in_eval_result: ", in_eval_result)

        if not os.path.isfile(test_eval_opt["report_filename"]):
            self.dist_print("RUNNING SYSTEM TEST EVAL", test_eval_opt)
            test_eval_result = self.eval_class()(test_eval_opt).run()
            self.dist_print("test_eval_result: ", test_eval_result)

        if not os.path.isfile(user_test_eval_opt["report_filename"]):
            self.dist_print("RUNNING USER SIM TEST EVAL", user_test_eval_opt)
            test_eval_result = self.eval_class()(user_test_eval_opt).run()
            self.dist_print("user_test_eval_result: ", test_eval_result)

        # convo generation
        if api_schemas and not self.opt["skip_train_convo_generation"]:
            if not os.path.isfile(train_model_model_opt["report_filename"] + ".json"):
                self.dist_print("RUNNING TRAIN MODEL-MODEL", train_model_model_opt)
                model_model_result = self.tod_world_class()(train_model_model_opt).run()
                self.dist_print("train_model_model_result: ", model_model_result)

            for nucleus_mm_opt in nucleus_mm_opts:
                if os.path.isfile(
                    nucleus_mm_opt["report_filename"].replace(".json", "") + ".json"
                ):
                    continue
                self.dist_print(
                    "RUNNING NUCLEUS MODEL MODEL FOR TOPP: ", nucleus_mm_opt["topp"]
                )
                model_model_result = self.tod_world_class()(nucleus_mm_opt).run()
                self.dist_print("nucleus_mm_result: ", model_model_result)

            if self.is_main_worker():
                self.dist_print("DOING GETTING PASSING ONLY")
                GetPassingOnlyScript(passing_only_opt).run()
                for nucleus_passing_only_opt in nucleus_passing_only_opts:
                    GetPassingOnlyScript(nucleus_passing_only_opt).run()

                if len(nucleus_processed_convo_filenames) > 0:
                    concatinate_me = [
                        passing_only_convo_file
                    ] + nucleus_processed_convo_filenames
                    concat_conversations(
                        concatinate_me,
                        destination=noncumulative_convo,
                        source_metadata=passing_only_convo_metadata,
                        destination_metadata=noncumulative_metadata,
                    )

                concatinate_me = (
                    self.opt["existing_train_files"] + nucleus_processed_convo_filenames
                )
                if len(concatinate_me) > 0:
                    concatinate_me = [passing_only_convo_file] + concatinate_me
                    concat_conversations(
                        concatinate_me,
                        destination=cumulative_convo,
                        source_metadata=passing_only_convo_metadata,
                        destination_metadata=cumulative_metadata,
                    )

        # Al generation
        if api_schemas and not self.opt["skip_al_generation"] and self.is_main_worker():
            self.dist_print(
                "Getting AL samples (printing only cumulative opts)",
                al_sample_cumulative_opt,
            )
            GetAlSamplesForGsgdScript(al_sample_noncumulative_opt).run()
            GetAlSamplesForGsgdScript(al_sample_cumulative_opt).run()

    def is_slurm_distributed(self):
        return self.rank < 0

    def is_main_worker(self):
        return self.rank <= 0

    def dist_print(self, *args):
        # distributed aware print
        if self.is_main_worker():
            print(*args)

    def train_class(self):
        if self.is_slurm_distributed():
            return TrainModel
        return DistributedTrain

    def eval_class(self):
        if self.is_slurm_distributed():
            return EvalModel
        return DistributedEval

    def tod_world_class(self):
        if self.is_slurm_distributed():
            return TodWorldScript
        return DistributedTodWorldScript


def concat_conversations(
    concatinate_me, destination, source_metadata=None, destination_metadata=None
):
    if source_metadata:
        copyfile(source_metadata, destination_metadata)
    seen_lines = set()
    file_stats = {}
    with open(destination, "w+") as destfile:
        for source in concatinate_me:
            print(source)
            new_in_file = 0
            dup_in_file = 0
            with open(source) as infile:
                for line in infile:
                    convo_raw = json.loads(line)["dialog"]
                    convo_formatted = [x["text"] for round in convo_raw for x in round]
                    convo = json.dumps(convo_formatted)
                    if convo not in seen_lines:
                        destfile.write(line)
                        seen_lines.add(convo)
                        new_in_file += 1
                    else:
                        dup_in_file += 1
            file_stats[source] = {"new": new_in_file, "dup": dup_in_file}
    print("Concat + filter dupe stats")
    print(json.dumps(file_stats, indent=4))
    with open(destination.replace(".jsonl", "_concat_stats.json"), "w+") as f:
        json.dump(file_stats, f, indent=4)


if __name__ == "__main__":
    TodDistributedUberScript.main()
