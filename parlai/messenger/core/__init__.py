#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Check 3rd-party dependencies
try:
    import joblib  # noqa: F401
    import websocket  # noqa: F401
    import sh  # noqa: F401
except ImportError:
    raise SystemExit(
        "Please install 3rd-party dependencies by running: "
        "pip install joblib websocket-client sh"
    )
