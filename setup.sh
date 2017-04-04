#!/bin/bash

if [ -e requirements.txt ]; then
    pip install -r requirements.txt
fi

python setup.py install
