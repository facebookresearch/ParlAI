# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Check 3rd-party dependencies
try:
    import boto3
    import botocore
    import joblib
    import socketIO_client_nexus
    import sh
except ModuleNotFoundError:
    raise SystemExit("Please install 3rd-party dependencies by running: pip install boto3 joblib socketIO-client-nexus sh")

