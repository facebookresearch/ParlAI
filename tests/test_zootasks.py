#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import parlai.core.testing_utils as testing_utils


class TestZooAndTasks(unittest.TestCase):
    """Make sure the package is alive."""

    def test_zoolist_types(self):
        from parlai.zoo.model_list import model_list
        self._check_types(model_list, 'Zoo')

    def test_tasklist_types(self):
        from parlai.tasks.task_list import task_list
        self._check_types(task_list, 'Task')

    def test_tasklist(self):
        from parlai.tasks.task_list import task_list
        self._check_directory(
            "task_list", task_list, "parlai/tasks", "task",
            ignore=['fromfile'],
        )

    def test_zoolist(self):
        from parlai.zoo.model_list import model_list
        self._check_directory(
            "model_list", model_list, "parlai/zoo", "id",
            ignore=["fasttext_cc_vectors", "fasttext_vectors", "glove_vectors",
                    "bert"]
        )

    def _check_directory(self, listname, thing_list, thing_dir, thing_key, ignore=None):
        if ignore is None:
            ignore = []
        dirs = testing_utils.git_ls_dirs()
        # get only directories directly in the thing directory
        dirs = [d for d in dirs if os.path.dirname(d) == thing_dir]
        # just the folder names
        dirs = [os.path.basename(d) for d in dirs]
        # skip the whitelist
        dirs = [d for d in dirs if d not in ignore]
        # make it a set
        dirs = set(dirs)

        # and the list of thing names
        thing_names = {thing[thing_key].split(':')[0] for thing in thing_list}

        errors = []
        # items with a directory but not a listing
        for name in dirs - thing_names:
            errors.append(
                "Directory {}/{} exists, but isn't in {}".format(
                    thing_dir, name, listname
                )
            )
        for name in thing_names - dirs:
            errors.append(
                "{} exists in {}, but {}/{} isn't a directory".format(
                    name, listname, thing_dir, name
                )
            )

        if errors:
            self.assertTrue(False, "\n".join(errors))

    def _check_types(self, thing_list, listname):
        for thing in thing_list:
            name = thing['id']
            for key, value in thing.items():
                if key == 'tags':
                    self.assertIsInstance(
                        value, list,
                        "{} {} tags is not a list".format(listname, name)
                    )
                    self.assertIsNot(
                        value, [], "{} {} must have some tags".format(listname, name)
                    )
                else:
                    self.assertIsInstance(
                        value, str,
                        "{} {}:{} must be string".format(listname, name, key)
                    )


if __name__ == '__main__':
    unittest.main()
