# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import unittest


class TestImport(unittest.TestCase):
    """Make sure the package is alive."""

    def test_import_agent(self):
        from parlai.core.agents import Agent
        assert Agent

    def test_import_teacher(self):
        from parlai.core.agents import Teacher
        assert Teacher

    def test_import_world(self):
        from parlai.core.worlds import World
        assert World

    def test_import_threadutils(self):
        from parlai.core.thread_utils import SharedTable
        assert SharedTable

    def test_import_dialog(self):
        from parlai.core.dialog_teacher import DialogTeacher
        assert DialogTeacher

    def test_import_fbdialog(self):
        from parlai.core.fbdialog_teacher import FbDialogTeacher
        assert FbDialogTeacher

if __name__ == '__main__':
    unittest.main()
