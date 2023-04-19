#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities relating to frontend code.
"""

import os
import shutil
import subprocess


def build_task(task_directory: str):
    """
    Build the task with npm.
    """

    # Paths
    frontend_source_dir = os.path.join(task_directory, "webapp")
    frontend_build_dir = os.path.join(frontend_source_dir, "build")
    return_dir = os.getcwd()

    # Build the task
    os.chdir(frontend_source_dir)
    if os.path.exists(frontend_build_dir):
        shutil.rmtree(frontend_build_dir)
    packages_installed = subprocess.call(["npm", "install", "--force"])
    if packages_installed != 0:
        raise Exception(
            "please make sure npm is installed, otherwise view "
            "the above error for more info."
        )
    webpack_complete = subprocess.call(["npm", "run", "dev"])
    if webpack_complete != 0:
        raise RuntimeError(
            "Webpack appears to have failed to build your "
            "frontend. See the above error for more information."
        )
    os.chdir(return_dir)
