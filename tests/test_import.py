#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest


class TestImport(unittest.TestCase):
    """
    Make sure the package is alive.
    """

    def test_import_agent(self):
        from parlai.core.agents import Agent

        assert Agent

    def test_import_teacher(self):
        from parlai.core.teachers import Teacher

        assert Teacher

    def test_import_world(self):
        from parlai.core.worlds import World

        assert World

    def test_import_dialog(self):
        from parlai.core.teachers import DialogTeacher

        assert DialogTeacher

    def test_import_fbdialog(self):
        from parlai.core.teachers import FbDeprecatedDialogTeacher

        assert FbDeprecatedDialogTeacher


if __name__ == '__main__':
    unittest.main()
