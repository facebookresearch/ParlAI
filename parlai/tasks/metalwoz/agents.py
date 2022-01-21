#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from parlai.utils.data import DatatypeHelper
from .build import build
import os
import pandas as pd
from typing import Optional


class MetalWozTeacherBase(DialogTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            "--metalwoz-domains", nargs="+", help="Use only a subset of the domains"
        )
        return parser

    def _path(self, opt):
        fold = DatatypeHelper.fold(opt["datatype"])
        if fold == "train" or fold == "valid":
            folder = os.path.join(opt["datapath"], "metalwoz", "train")
        else:
            folder = os.path.join(opt["datapath"], "metalwoz", "test")
        return folder, fold

    def __init__(self, opt, shared=None):
        if shared is None:
            build(opt)
        folder, fold = self._path(opt)
        self.fold = fold
        opt["datafile"] = os.path.join(folder, fold)
        super().__init__(opt, shared)

    def load_data(self, datapath):
        folder, fold = os.path.split(datapath)
        with PathManager.open(os.path.join(folder, "tasks.txt")) as taskf:
            tasks_table = pd.read_json(taskf, lines=True)

        dfolder = os.path.join(folder, "dialogues")

        data = []

        for filename in PathManager.ls(dfolder):
            domain = filename.replace(".txt", "")
            if (
                self.opt["metalwoz_domains"]
                and domain not in self.opt["metalwoz_domains"]
            ):
                continue
            fullfn = os.path.join(dfolder, filename)
            with PathManager.open(fullfn) as dataf:
                lines = pd.read_json(dataf, lines=True)
                lines = lines.merge(tasks_table, on="task_id")
                data.append(lines.to_dict("records"))

        # Quick check to make sure we didn't fat-finger the spelling of some domain
        if self.opt["metalwoz_domains"]:
            assert len(data) == len(self.opt["metalwoz_domains"])

        if "test" in self.fold:
            flat = []
            for domain in data:
                flat.extend(domain)
            return flat

        return DatatypeHelper.split_subset_data_by_fold(self.fold, data, 0.8, 0.1, 0.1)


class SystemTeacher(MetalWozTeacherBase):
    def setup_data(self, datapath):
        data = self.load_data(datapath)
        for row in data:
            texts = [row["bot_role"]] + list(row["turns"])
            prompts, labels = texts[::2], texts[1::2]
            for i, (prompt, label) in enumerate(zip(prompts, labels)):
                yield {
                    "text": prompt,
                    "label": label,
                    "bot_role": row["bot_role"],
                    "bot_prompt": row["bot_prompt"],
                    "user_role": row["user_role"],
                    "user_prompt": row["user_prompt"],
                    "utterance_id": row["id"],
                    "domain": row["domain_x"],
                    "task_id": row["task_id"],
                }, i == 0


class UserSimulatorTeacher(MetalWozTeacherBase):
    def setup_data(self, datapath):
        data = self.load_data(datapath)
        for row in data:
            texts = list(row["turns"])
            prompts, labels = (
                [f"{row['user_role']}\n{texts[0]}"] + texts[2::2],
                texts[1::2],
            )
            for i, (prompt, label) in enumerate(zip(prompts, labels)):
                yield {
                    "text": prompt,
                    "label": label,
                    "bot_role": row["bot_role"],
                    "bot_prompt": row["bot_prompt"],
                    "user_role": row["user_role"],
                    "user_prompt": row["user_prompt"],
                    "utterance_id": row["id"],
                    "domain": row["domain_x"],
                    "task_id": row["task_id"],
                }, i == 0


class MetalWozTeacher(SystemTeacher):
    pass


class DefaultTeacher(MetalWozTeacher):
    pass
