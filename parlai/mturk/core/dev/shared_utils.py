#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import uuid
import os

# Sleep constants
THREAD_SHORT_SLEEP = 0.1
THREAD_MEDIUM_SLEEP = 0.3
# ThrottlingException might happen if we poll too frequently
THREAD_MTURK_POLLING_SLEEP = 10

logger = None
logging_enabled = True
debug = False
log_level = logging.ERROR

if logging_enabled:
    logging.basicConfig(
        filename=str(time.time()) + '.log',
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    logger = logging.getLogger('mturk')


def set_log_level(new_level):
    global log_level
    log_level = new_level


def set_is_debug(is_debug):
    global debug
    debug = is_debug


def disable_logging():
    global logging_enabled
    logging_enabled = False


def print_and_log(level, message, should_print=False):
    if (logging_enabled and level >= log_level) or debug:
        logger.log(level, message)
    if should_print or debug:  # Always print message in debug mode
        print(message)


def generate_event_id(worker_id):
    """Return a unique id to use for identifying a packet for a worker"""
    return '{}_{}'.format(worker_id, uuid.uuid4())


def get_mturk_dir():
    import parlai.mturk
    return os.path.dirname(os.path.abspath(parlai.mturk.__file__))


def get_core_dir():
    import parlai.mturk.core
    return os.path.dirname(os.path.abspath(parlai.mturk.core.dev.__file__))


def get_tmp_dir():
    """Return the location of the temporary directory in which we store
    things related to a run but that can be safely deleted
    """
    tmp_dir = os.path.join(get_mturk_dir(), 'tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    return tmp_dir
