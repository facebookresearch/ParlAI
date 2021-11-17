#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script for training, evaluating, and running model model chats for Task Oriented Dialog.

Note that the code below does assume running in a SLURM environment

Modifications for MultiWoz

Use
```
parlai train -t parlai.tasks.multiwoz_v22.agents:StandaloneApiTeacher --standalone-api-file standalone_api_file -m parlai.core.tod.tod_agents.agents:TodStandaloneApiAgent -mf <OUTPUT_PATH> -eps 5
```
to define STANDALONE_API_FILE_PATH
"""
from parlai.core.script import register_script
from parlai.scripts.distributed_eval import DistributedEval
from projects.tod_simulator.scripts.get_passing_only import GetPassingOnlyScript
from projects.tod_simulator.scripts.tod_distributed_uber_script import (
    concat_conversations,
    TodDistributedUberScript,
)
import copy
import os


STANDALONE_API_FILE_PATH = (
    "/checkpoint/<INSERT_HERE>/projects/user_simulator/standalone_api_data/multiwoz_v22"
)


@register_script("tod_distributed_uber_multiwoz_script")
class TodDistributedUberMultiwozScript(TodDistributedUberScript):
    """
    Version of Tod Distributed Uber Script, but for Multiwoz.
    """

    def get_mm_opt(self, datatype, port, mm_prefix="", mm_suffix=""):
        """
        Override for Multiwoz.
        """
        model_model_opt = super().get_mm_opt(datatype, port, mm_prefix, mm_suffix)

        if self.opt["api_schemas"]:
            grounding = "parlai.tasks.multiwoz_v22.agents:SingleApiSchemaAgent"
        else:
            grounding = "parlai.core.tod.tod_agents_and_teachers:TodEmptyApiSchemaAgent"

        model_model_opt["api_schema_grounding_model"] = grounding
        model_model_opt[
            "goal_grounding_model"
        ] = "parlai.tasks.multiwoz_v22.agents:SingleGoalAgent"
        model_model_opt[  # Same as original but including for completion
            "api_resp_model"
        ] = "parlai.core.tod.tod_agents_and_teachers:TodStandaloneApiAgent"
        model_model_opt["standalone_api_file"] = STANDALONE_API_FILE_PATH

        model_model_opt["system_model_file"] = self.opt["model_file"]
        model_model_opt["user_model_file"] = self.opt["model_file"]
        model_model_opt["distributed_world_size"] = self.opt["distributed_world_size"]
        model_model_opt["ddp_backend"] = self.opt["ddp_backend"]
        model_model_opt["save_format"] = "conversations"
        model_model_opt["log_every_n_seconds"] = 30

        pretty_print = ""
        for key in model_model_opt:
            if key == "override":
                continue
            pretty_print += "--" + key.replace("_", "-")
            pretty_print += " "
            pretty_print += str(model_model_opt[key])
            pretty_print += " "
        self.dist_print("After updating with MultiWoz v2.2 args:")
        self.dist_print(pretty_print)

        return model_model_opt

    @classmethod
    def setup_args(cls):
        parser = super().setup_arg()
        parser.set_defaults(
            skip_al_generation=True
        )  # We're not doing this for multiwoz yet
        return parser

    def run(self):
        # NOTE: Following is ugly copypasta of `TodDistributedUberScript`, but good for keeping open optionality
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
                "parlai.tasks.multiwoz_v22.agents:SystemTeacher",
                "--model",
                self.opt["model"],
            ]
        )
        # ...but also make sure to use the right settings passed in via run_grid (ie distributed opts)
        for key in eval_opt:
            if key in self.opt:
                eval_opt[key] = self.opt[key]
        # Now reset opts that we need here
        eval_opt["task"] = "parlai.tasks.multiwoz_v22.agents:SystemTeacher"
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
        user_eval_opt["task"] = "parlai.tasks.multiwoz_v22.agents:UserSimulatorTeacher"
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
        ] = "parlai.tasks.multiwoz_v22.agents:UserSimulatorTeacher"
        user_test_eval_opt["report_filename"] = os.path.join(
            base_path_name, "test_user_eval_stats.json"
        )
        user_test_eval_opt["port"] = self.opt["port"] + 6

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
                "--filter-call-attempts",  # Way the API works with multiple answers requires not filtering one shot
                str(True),
            ]
        )

        # NOTE: Following line needs to be kept in sync with the get_passing_only script
        infile_base = mm_convo_filename.replace(".jsonl", "")
        # Unused in this script at the moment
        # passing_only_stats_file = infile_base.replace(
        #    "_conversations", "_processed_stats"
        # )
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


if __name__ == "__main__":
    TodDistributedUberMultiwozScript.main()
