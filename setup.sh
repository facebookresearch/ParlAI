#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

echo "Setting up parlai using 'linked' install (see README for details)...
If ParlAI doesn't work after running this, run the commands in setup.sh \
directly from your terminal (if you have multiple versions of python \
installed and use aliases to select between them, this might be necessary)."

if [ -e requirements.txt ]; then
    pip install -r requirements.txt
fi

python setup.py develop

echo "ParlAI setup complete."
