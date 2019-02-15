#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import git
import os


_git = git.Git()


def git_ls_files(root=None, skip_nonexisting=True):
    filenames = _git.ls_files(root).split('\n')
    if skip_nonexisting:
        filenames = [fn for fn in filenames if os.path.exists(fn)]
    return filenames


def git_ls_dirs(root=None):
    dirs = set()
    for fn in git_ls_files(root):
        dirs.add(os.path.dirname(fn))
    return list(dirs)


def git_changed_files(skip_nonexisting=True):
    fork_point = _git.merge_base('--fork-point', 'origin/master').strip()
    filenames = _git.diff('--name-only', fork_point).split('\n')
    if skip_nonexisting:
        filenames = [fn for fn in filenames if os.path.exists(fn)]
    return filenames


if __name__ == '__main__':
    from pprint import pprint
    pprint(git_changed_files('parlai'))
