#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.
# if parlai can't be imported after running this, try running this code from
# your terminal directly instead of running this file--sometimes this won't work
# properly if you have python aliased

if [ -e requirements.txt ]; then
    pip install -r requirements.txt
fi

python setup.py develop
