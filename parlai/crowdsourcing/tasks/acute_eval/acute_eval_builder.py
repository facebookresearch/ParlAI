#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.data_model.blueprint import TaskBuilder

import os
import shutil
import subprocess

ACUTE_TASK_DIR = os.path.dirname(__file__)
FRONTEND_SOURCE_DIR = os.path.join(ACUTE_TASK_DIR, "webapp")
FRONTEND_BUILD_DIR = os.path.join(FRONTEND_SOURCE_DIR, "build")


class AcuteEvalBuilder(TaskBuilder):
    """
    Builder for a static task, pulls the appropriate html, builds the frontend (if a
    build doesn't already exist), then puts the file into the server directory.
    """

    BUILT_FILE = "done.built"
    BUILT_MESSAGE = "built!"

    def rebuild_core(self):
        """
        Rebuild the frontend for this task.
        """
        return_dir = os.getcwd()
        os.chdir(FRONTEND_SOURCE_DIR)
        if os.path.exists(FRONTEND_BUILD_DIR):
            shutil.rmtree(FRONTEND_BUILD_DIR)
        packages_installed = subprocess.call(["npm", "install"])
        if packages_installed != 0:
            raise Exception(
                "please make sure npm is installed, otherwise view "
                "the above error for more info."
            )
        webpack_complete = subprocess.call(["npm", "run", "dev"])
        if webpack_complete != 0:
            raise Exception(
                "Webpack appears to have failed to build your "
                "frontend. See the above error for more information."
            )
        os.chdir(return_dir)

    def build_in_dir(self, build_dir: str):
        """
        Build the frontend if it doesn't exist, then copy into the server directory.
        """
        # Only build this task if it hasn't already been built
        if True:  # not os.path.exists(FRONTEND_BUILD_DIR):
            self.rebuild_core()

        # Copy the built core and the given task file to the target path
        bundle_js_file = os.path.join(FRONTEND_BUILD_DIR, "bundle.js")
        target_resource_dir = os.path.join(build_dir, "static")
        target_path = os.path.join(target_resource_dir, "bundle.js")
        shutil.copy2(bundle_js_file, target_path)

        copied_static_file = os.path.join(
            FRONTEND_SOURCE_DIR, "src", "static", "index.html"
        )
        target_path = os.path.join(target_resource_dir, "index.html")
        shutil.copy2(copied_static_file, target_path)

        # Write a built file confirmation
        with open(os.path.join(build_dir, self.BUILT_FILE), "w+") as built_file:
            built_file.write(self.BUILT_MESSAGE)

    # TODO(#97) update test validation
    @staticmethod
    def task_dir_is_valid(task_dir: str) -> bool:
        """
        Acute eval is always valid, we don't have any special resources.
        """
        return True
