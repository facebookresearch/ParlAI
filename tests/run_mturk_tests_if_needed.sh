#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
set -e
CHANGES=$(git diff --name-only HEAD~1)
if echo "$CHANGES" | grep -q "parlai/mturk/"; then
  pip install boto3 joblib websocket-client sh websocket_server
  python3 parlai/mturk/core/test/test_mturk_agent.py
  python3 parlai/mturk/core/test/test_worker_manager.py
  python3 parlai/mturk/core/test/test_socket_manager.py
  python3 parlai/mturk/core/test/test_mturk_manager.py
  python3 parlai/mturk/core/test/test_full_system.py
fi
exit 0
