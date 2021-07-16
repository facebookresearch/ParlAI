#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
import unittest

from parlai.scripts.display_data import setup_args
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.tasks.cmu_dog.agents import CMUDocumentGroundedConversationsTeacher
import parlai.utils.testing as testing_utils


class CMUDoGTest(unittest.TestCase):
    def test_deduped_split_distributions(self):
        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir

            def _split_type_teacher(
                split_type: str,
            ) -> CMUDocumentGroundedConversationsTeacher:
                kwargs = {
                    'task': 'cmu_dog',
                    'datatype': 'valid',
                    'cmu_dog_split_type': split_type,
                    'datapath': data_path,
                }
                parser = setup_args()
                parser.set_defaults(**kwargs)
                opt = parser.parse_args([])
                agents = create_task_agent_from_taskname(opt)
                assert isinstance(agents, List)
                task = agents[0]
                assert isinstance(task, CMUDocumentGroundedConversationsTeacher)
                return task

            og_teacher = _split_type_teacher('deduped')
            sn_teacher = _split_type_teacher('seen')
            self.assertEqual(
                len(og_teacher.rare_word_f1.freq_dist),
                len(sn_teacher.rare_word_f1.freq_dist),
            )
