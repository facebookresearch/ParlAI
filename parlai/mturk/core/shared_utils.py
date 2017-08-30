# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import time
import uuid

# Sleep constants
THREAD_SHORT_SLEEP = 0.1
THREAD_MEDIUM_SLEEP = 0.3
# ThrottlingException might happen if we poll too frequently
THREAD_MTURK_POLLING_SLEEP = 10

logging_enabled = True
logger = None
debug = True

if logging_enabled:
    logging.basicConfig(
        filename=str(time.time())+'.log',
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    logger = logging.getLogger('mturk')


def print_and_log(message, should_print=True):
    if logging_enabled:
        logger.info(message)
    if should_print or debug: # Always print message in debug mode
        print(message)


def generate_event_id(worker_id):
    """Return a unique id to use for identifying a packet for a worker"""
    return '{}_{}'.format(worker_id, uuid.uuid4())
