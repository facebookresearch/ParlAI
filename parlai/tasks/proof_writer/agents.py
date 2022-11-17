#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
From https://allenai.org/data/proofwriter ; paper: "ProofWriter: Generating
Implications, Proofs, and Abductive Statements over Natural Language" (Tafjord et al
2021) https://aclanthology.org/2021.findings-acl.317.pdf.

This is a synthentically dataset of initial clauses and rules, with questions about
statements that these initial clauses and rules imply.
"""

import json
import os

import copy
import random
import re
from typing import Optional, List, Tuple

from tasks.reasoning.agents import StepByStepReasoningTeacher

from .build import build
from parlai.core.message import Message
from parlai.core.mutators import (
    ManyEpisodeMutator,
    register_mutator,
)
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.data import DatatypeHelper
from parlai.utils.io import PathManager

QUESTION_PREFIX = {
    "CWA": "Is the following True or False? ",
    "OWA": "Is the following True, False, or Unknown? ",
}


class ProofWriterTeacher(DialogTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("Proof Writer args")
        group.add_argument(
            '--proofwriter-world',
            type=str,
            choices=["OWA", "CWA"],
            default="OWA",
            help='Closed world assumption vs open world assumption',
        )
        group.add_argument(
            '--proofwriter-dataset',
            type=str,
            choices=[
                "depth-0",
                "depth-1",
                "depth-2",
                "depth-3",
                "depth-5",
                "birds-electricity",
                "NatLang",
                "depth-3ext",
                "depth-3ext-NatLang",
            ],
            help="""The different proofwriter datasets. Descriptions:
            * depth-N (N=0, 1, 2, 3, 5): Questions with rulebases in synthetic language reasoning paths up to depth N, as
            defined in the paper.
            * birds-electricity: Questions with "birds" and "electricity" rulebases.
            * NatLang: Questions with rulebases in crowdsourced natural language.
            * depth-3ext: The Depth3 dataset augmented with 10% each of the depth=0, depth=1, and depth-2 datasets.
            * depth-3ext-NatLang: The Depth3Ext dataset augmented with the NatLang one.""",
            default="depth-0",
        )
        group.add_argument(
            '--proofwriter-version',
            type=str,
            choices=["base", "stage", "abduct"],
            default="base",
            help="""Version of the dataset to use
            * base: Main dataset for each split, with theories and associated questions and proofs
            * stage: "Staged" version of each theory, with depth-1 implications derived one at a
            time while adding to the theory.
            * abduct: "Missing fact" data for each theory. Only exists for OWA""",
        )
        group.add_argument(
            '--proofwriter-shuffle-context',
            type=bool,
            default=False,
            help="Use default theory as context or used a shuffled version based off of component facts + rules.",
        )
        group.add_argument(
            '--proofwriter-data-verbose',
            type=bool,
            default=False,
            help="Copy *all* the metadata for facts, rules, and questions into the Message. Useful for enabling some"
            + "mutators. Kept also to keep track of which relevant fields out of ProofWriter that we might want to use"
            + "have or have not been used.",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.fold = DatatypeHelper.fold(opt['datatype'])
        build(opt)
        opt['datafile'] = os.path.join(
            opt['datapath'], 'proof_writer', "proofwriter-dataset-V2020.12.3"
        )
        self.proofwriter_random = random.Random(42)
        super().__init__(opt, shared)

    def setup_data(self, path):
        data_file, version = self.get_datafile()

        process_funcs = {
            "base": self.process_base,
            "stage": self.process_stage,
            "abduct": self.process_abduct,
        }

        with PathManager.open(data_file) as f:
            for x in f.readlines():
                for ex in process_funcs[version](json.loads(x.strip())):
                    yield ex

    def get_datafile(self) -> str:
        base_path = os.path.join(
            self.opt["datafile"],
            self.opt.get("proofwriter-world", "OWA"),
            self.opt.get("proofwriter_dataset", "depth-0"),
        )
        fold_path_string = self.fold
        if self.fold == "valid":
            fold_path_string = "dev"
        version = self.opt.get("proofwriter_version", "base")
        if version == "base":
            return os.path.join(base_path, f"meta-{fold_path_string}.jsonl"), version
        else:
            return (
                os.path.join(base_path, f"meta-{version}-{fold_path_string}.jsonl"),
                version,
            )

    def process_base(self, blob) -> List[Tuple[Message, bool]]:
        result = []
        base = {
            "id": blob["id"],
        }
        facts = [blob["triples"][x]["text"] for x in blob["triples"]]
        rules = [blob["rules"][x]["text"] for x in blob["rules"]]
        if self.opt.get("proofwriter_shuffle_context", False):
            self.proofwriter_random.shuffle(facts)
            self.proofwriter_random.shuffle(rules)
            context = " ".join(facts + rules)
        else:
            context = blob["theory"]

        facts_dict = {x: blob["triples"][x]["text"] for x in blob["triples"]}
        rules_dict = {x: blob["rules"][x]["text"] for x in blob["rules"]}
        if self.opt.get("proofwriter_data_verbose", False):
            base["context"] = context
            base["facts"] = facts_dict
            base["rules"] = rules_dict
            for x in ["maxD", "NFact", "NRule"]:
                base[x] = blob[x]

        for question_blob in blob["questions"].values():
            message = copy.deepcopy(base)
            question_prefix = QUESTION_PREFIX[self.opt.get('proofwriter_world')]
            question = question_blob["question"]
            message["text"] = f"{context}\n{question_prefix}{question}"
            message["label"] = str(question_blob["answer"])
            # note that we are explicitly choosing the first of the intermdiates here.
            if "proofsWithIntermediates" in question_blob:
                proofs_blob = question_blob["proofsWithIntermediates"][0]
                details = {**facts_dict, **rules_dict}
                if len(proofs_blob["intermediates"]) > 0:
                    intermediates = {
                        x: proofs_blob["intermediates"][x]["text"]
                        for x in proofs_blob["intermediates"]
                    }
                    details.update(intermediates)
                proof_string = proofs_blob["representation"]
                matches = re.findall(r"\b[A-Za-z\d]+\b", proof_string)
                message["steps"] = [details[x] for x in matches]

            if "steps" not in message:
                message["steps"] = []

            if self.opt.get("proofwriter_data_verbose", False):
                for x in ["QDep", "QLen", "answer", "question"]:
                    message[x] = question_blob[x]
            result.append((Message(message), True))

        return result

    def process_stage(self, blob):
        raise NotImplementedError()

    def process_abduct(self, blob):
        raise NotImplementedError()


class DefaultTeacher(ProofWriterTeacher):
    pass


@register_mutator("proof_writer_multistep")
class ProofWriterMultistepMutator(ManyEpisodeMutator):
    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        if len(episode) == 0 or len(episode[0]["steps"]) == 0:
            return []
        return [episode]


@register_mutator("proof_writer_expand_step")
class ProofWriterExpandStepMutator(ManyEpisodeMutator):
    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        if len(episode) == 0:
            return []
        result = []
        sample = episode[0]
        for i in range(len(sample["steps"])):
            duplicate = dict(copy.deepcopy(sample))
            duplicate["text"] = "\n<THINK>".join([sample["text"]] + sample["steps"][:i])
            duplicate["labels"] = [f"<THINK>{sample['steps'][i]}"]
            result.append([Message(duplicate)])
        # Handle last (where the label is the correct answer) separately
        duplicate = dict(copy.deepcopy(sample))
        duplicate["text"] = "\n<THINK>".join([sample["text"]] + sample["steps"])
        result.append([Message(duplicate)])
        return result


class ProofWriterStepByStepReasoningTeacher(
    StepByStepReasoningTeacher, ProofWriterTeacher
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.set_defaults(
            task_prompt="Solve the following logical problem. Show your work."
        )
        group = parser.add_argument_group("Proof Writer args")
        group.add_argument(
            '--proofwriter-world',
            type=str,
            choices=["OWA", "CWA"],
            default="OWA",
            help='Closed world assumption vs open world assumption',
        )
        group.add_argument(
            '--proofwriter-dataset',
            type=str,
            choices=[
                "depth-0",
                "depth-1",
                "depth-2",
                "depth-3",
                "depth-5",
                "birds-electricity",
                "NatLang",
                "depth-3ext",
                "depth-3ext-NatLang",
            ],
            help="""The different proofwriter datasets. Descriptions:
            * depth-N (N=0, 1, 2, 3, 5): Questions with rulebases in synthetic language reasoning paths up to depth N, as
            defined in the paper.
            * birds-electricity: Questions with "birds" and "electricity" rulebases.
            * NatLang: Questions with rulebases in crowdsourced natural language.
            * depth-3ext: The Depth3 dataset augmented with 10% each of the depth=0, depth=1, and depth-2 datasets.
            * depth-3ext-NatLang: The Depth3Ext dataset augmented with the NatLang one.""",
            default="depth-0",
        )
        group.add_argument(
            '--proofwriter-version',
            type=str,
            choices=["base", "stage", "abduct"],
            default="base",
            help="""Version of the dataset to use
            * base: Main dataset for each split, with theories and associated questions and proofs
            * stage: "Staged" version of each theory, with depth-1 implications derived one at a
            time while adding to the theory.
            * abduct: "Missing fact" data for each theory. Only exists for OWA""",
        )
        parser.add_argument(
            "--extrinsic-step",
            type=bool,
            default=False,
        )
        return parser

    def __init__(self, opt, shared=None):
        self.proofwriter_random = random.Random(42)
        self.id = 'proof_writer'
        super().__init__(opt, shared)

    def get_data_for_fold(self, fold):
        extrinsic_step = self.opt.get("extrinsic_step", False)
        data_file, version = super().get_datafile()

        process_funcs = {
            "base": self.process_base,
            "stage": self.process_stage,
            "abduct": self.process_abduct,
        }

        messages = []
        with PathManager.open(data_file) as f:
            for x in f.readlines():
                for ex in process_funcs[version](json.loads(x.strip())):
                    # include only datapoint with at least 1 step
                    if len(ex[0]["steps"]) > 0:
                        messages.append(
                            {
                                "question": ex[0]["text"],
                                "answer": ex[0]["label"],
                                "steps": ex[0]["steps"],
                            }
                        )

        for m in messages:
            if extrinsic_step:
                random_step = None
                # make sure new step is from a different context
                # here we aasume that there is at least one step in the set
                # with different context, otherwise it will go in the
                # infinite loop
                while not random_step or random_step in m["question"]:
                    rand_steps = self.proofwriter_random.choice(messages)["steps"]
                    random_step = self.proofwriter_random.choice(rand_steps)
                m["extrinsic_step"] = random_step
                yield m
            else:
                yield m
