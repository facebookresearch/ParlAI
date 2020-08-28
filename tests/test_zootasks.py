#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test common developer mistakes in the model zoo and task list.

Mostly just ensures the docs will output nicely.
"""

import os
import unittest
import pytest
import parlai.utils.testing as testing_utils
from parlai.zoo.model_list import model_list
from parlai.tasks.task_list import task_list

ZOO_EXCEPTIONS = {"fasttext_cc_vectors", "fasttext_vectors", "glove_vectors", "bert"}


class TestZooAndTasks(unittest.TestCase):
    """
    Make sure the package is alive.
    """

    def _assertZooString(self, member, container, animal_name=None):
        msg = f'Missing or empty {member} in parlai.zoo.model_list'
        if animal_name:
            msg += f' [{animal_name}]'
        self.assertIn(member, container, msg=msg)
        self.assertTrue(container[member], msg=msg)

    def test_zoolist_fields(self):
        """
        Ensure zoo entries conform to style standards.
        """

        for animal in model_list:
            self._assertZooString('title', animal)
            name = animal['title']
            # every task must at least contain these
            for key in ['id', 'task', 'description', 'example', 'result']:
                self._assertZooString(key, animal, name)

            # if there's a second example there should be a second result
            if 'example2' in animal:
                self._assertZooString('result2', animal, name)

            # every entry needs a project page or a website
            self.assertTrue(
                ("project" in animal) or ("external_website" in animal),
                f"Zoo entry ({name}) should contain either project or external_website",
            )

    def test_zoolist_types(self):
        """
        Ensure no type errors in the model zoo.
        """
        self._check_types(model_list, 'Zoo')

    def test_tasklist_types(self):
        """
        Ensure no type errors in the task list.
        """
        self._check_types(task_list, 'Task')

    @pytest.mark.nofbcode
    def test_tasklist(self):
        """
        Check the task list for issues.
        """
        self._check_directory(
            "task_list",
            task_list,
            "parlai/tasks",
            "task",
            ignore=['fromfile', 'interactive', 'jsonfile', 'wrapper'],
        )

    @pytest.mark.nofbcode
    def test_zoolist(self):
        """
        Check the zoo list for issues.
        """
        self._check_directory(
            "model_list", model_list, "parlai/zoo", "id", ignore=ZOO_EXCEPTIONS
        )

    def _check_directory(self, listname, thing_list, thing_dir, thing_key, ignore=None):
        if ignore is None:
            ignore = []
        dirs = testing_utils.git_ls_dirs()
        # get only directories directly in the thing directory
        dirs = [d for d in dirs if os.path.dirname(d) == thing_dir]
        # just the folder names
        dirs = [os.path.basename(d) for d in dirs]
        # skip the allowlist
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
                        value, list, "{} {} tags is not a list".format(listname, name)
                    )
                    self.assertIsNot(
                        value, [], "{} {} must have some tags".format(listname, name)
                    )
                elif key == 'links':
                    self.assertIsInstance(value, dict)
                    for k_, v_ in value.items():
                        self.assertIsInstance(k_, str)
                        self.assertIsInstance(v_, str)
                else:
                    self.assertIsInstance(
                        value,
                        str,
                        "{} {}:{} must be string".format(listname, name, key),
                    )


if __name__ == '__main__':
    unittest.main()
