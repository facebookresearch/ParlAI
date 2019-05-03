#!/bin/sh

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

git ls-files '**/build.py' 'parlai/zoo/' 'tests/nightly/gpu' | sort | xargs md5sum
