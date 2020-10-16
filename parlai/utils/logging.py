#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import logging

try:
    import coloredlogs

    COLORED_LOGS = True
except ImportError:
    COLORED_LOGS = False

SPAM = 5
DEBUG = logging.DEBUG
VERBOSE = DEBUG + 5
INFO = logging.INFO
REPORT = INFO + 5
SUCCESS = REPORT + 1
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(SPAM, "SPAM")
logging.addLevelName(REPORT, "REPORT")
logging.addLevelName(SUCCESS, "SUCCESS")

COLORED_FORMAT = '%(asctime)s | %(message)s'
CONSOLE_FORMAT = '%(asctime)s %(levelname).4s | %(message)s'
CONSOLE_DATE_FORMAT = '%H:%M:%S'
LOGFILE_FORMAT = '%(asctime)s %(levelname)-8s | %(message)s'
LOGFILE_DATE_FORMAT = None

COLORED_LEVEL_STYLES = {
    'spam': {'color': 'white', 'faint': True},
    'debug': {'faint': True},
    'verbose': {'color': 'blue'},
    'error': {'color': 'red'},
    'info': {},
    'report': {'bold': True},
    'success': {'bold': True, 'color': 'green'},
    'warning': {'color': 'yellow'},
    'critical': {'bold': True, 'color': 'red'},
}


def _is_interactive():
    if os.environ.get('PARLAI_FORCE_COLOR'):
        return True
    try:
        __IPYTHON__
        return True
    except NameError:
        return sys.stdout.isatty()


# Some functions in this class assume that ':' will be the separator used in
# the logging formats setup for this class
class ParlaiLogger(logging.Logger):
    def __init__(self, name, console_level=INFO):
        """
        Initialize the logger object.

        :param name:
            Name of the logger
        :param console_level:
            minimum level of messages logged to console
        """
        super().__init__(name, console_level)  # can be initialized with any level
        # Logging to stdout
        self.streamHandler = logging.StreamHandler(sys.stdout)
        # Log to stdout levels: console_level and above
        self.prefix = None
        self.interactive = _is_interactive()
        self.streamHandler.setFormatter(self._build_formatter())
        super().addHandler(self.streamHandler)

    def _build_formatter(self):
        prefix_format = f'{self.prefix} ' if self.prefix else ''
        if COLORED_LOGS and self.interactive:
            return coloredlogs.ColoredFormatter(
                prefix_format + COLORED_FORMAT,
                datefmt=CONSOLE_DATE_FORMAT,
                level_styles=COLORED_LEVEL_STYLES,
                field_styles={},
            )
        elif self.interactive:
            return logging.Formatter(
                prefix_format + CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT
            )
        else:
            return logging.Formatter(
                prefix_format + LOGFILE_FORMAT, datefmt=LOGFILE_DATE_FORMAT
            )

    def force_interactive(self):
        self.interactive = True
        self.streamHandler.setFormatter(self._build_formatter())

    def log(self, msg, level=INFO):
        """
        Default Logging function.
        """
        super().log(level, msg)

    def add_format_prefix(self, prefix):
        """
        Include `prefix` in all future logging statements.
        """
        # change both handler formatters to add a prefix
        self.prefix = prefix
        self.streamHandler.setFormatter(self._build_formatter())

    def mute(self):
        """
        Stop logging to stdout.
        """
        self.prev_level = self.streamHandler.level
        self.streamHandler.level = ERROR
        return self.prev_level

    def unmute(self):
        """
        Resume logging to stdout.
        """
        self.streamHandler.level = self.prev_level


# -----------------------------------
# Forming the logger                #
# -----------------------------------
logger = ParlaiLogger(name=__name__)


def set_log_level(level):
    logger.setLevel(level)


def disable():
    logger.mute()


def enable():
    logger.unmute()


def info(msg):
    return logger.info(msg)


def critical(msg):
    return logger.critical(msg)


def report(msg):
    return logger.log(msg, level=REPORT)


def success(msg):
    return logger.log(msg, level=SUCCESS)


def log(*args, **kwargs):
    return logger.log(*args, **kwargs)


def verbose(msg):
    return logger.log(msg, level=VERBOSE)


def debug(*args, **kwargs):
    return logger.debug(*args, **kwargs)


def error(*args, **kwargs):
    return logger.error(*args, **kwargs)


def warn(*args, **kwargs):
    return logger.warn(*args, **kwargs)


def warning(*args, **kwargs):
    return logger.warn(*args, **kwargs)


def get_all_levels():
    levels = set(logging._nameToLevel.keys())
    levels.remove('WARNING')
    return [l.lower() for l in levels]
