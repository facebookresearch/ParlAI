#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import unittest


class TestImport(unittest.TestCase):
    """Make sure the package is alive."""

    def test_import_agent(self):
        from parlai.core.agents import Agent
        assert Agent({}) is not None

    def test_import_teacher(self):
        from parlai.core.agents import Teacher
        assert Teacher({}) is not None

    def test_import_world(self):
        from parlai.core.worlds import World
        assert World({}) is not None

    def test_import_threadutils(self):
        from parlai.core.thread_utils import SharedTable
        assert SharedTable

    def test_import_dialog(self):
        from parlai.core.dialog import DialogTeacher
        assert DialogTeacher

    def test_import_fbdialog(self):
        from parlai.core.fbdialog import FbDialogTeacher
        assert FbDialogTeacher

    def test_import_remoteagent(self):
        # for some reason importing zmq causes an ignored ImportWarning
        # https://github.com/zeromq/pyzmq/issues/1004
        from parlai.agents.remote_agent.agents import RemoteAgent
        assert RemoteAgent


if __name__ == '__main__':
    unittest.main()
