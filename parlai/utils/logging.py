#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    'info': {'faint': True},
    'report': {'bold': True},
    'success': {'bold': True, 'color': 'green'},
    'warning': {'color': 'yellow'},
    'critical': {'bold': True, 'color': 'red'},
}


# Some functions in this class assume that ':' will be the separator used in
# the logging formats setup for this class
class ParlaiLogger(logging.Logger):
    def __init__(
        self, name, console_level=INFO,
    ):
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
        warn_colored = False
        if COLORED_LOGS and sys.stdout.isatty():
            self.formatter = coloredlogs.ColoredFormatter(
                COLORED_FORMAT,
                datefmt=CONSOLE_DATE_FORMAT,
                level_styles=COLORED_LEVEL_STYLES,
                field_styles={},
            )
        elif sys.stdout.isatty():
            self.formatter = logging.Formatter(
                CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT
            )
            warn_colored = True
        else:
            self.formatter = logging.Formatter(
                LOGFILE_FORMAT, datefmt=LOGFILE_DATE_FORMAT
            )
        self.streamHandler.setFormatter(self.formatter)
        super().addHandler(self.streamHandler)

        if warn_colored:
            self.warn("Run `pip install coloredlogs` for more friendly output")

        # To be used with testing_utils.capture_output()
        self.altStream = None

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
        new_str = prefix + " " + '%(message)s'

        prevConsoleFormat = self.consoleFormatter._fmt.split(':')[:-1]
        # Check if there was a format before this
        if prevConsoleFormat:
            # If so append prefix neatly after last divider
            prevConsoleFormat += [' ' + new_str]
            updatedConsoleFormat = ':'.join(prevConsoleFormat)
        else:
            updatedConsoleFormat = new_str
        self.streamHandler.setFormatter(logging.Formatter(updatedConsoleFormat))

        if hasattr(self, 'fileHandler'):
            prevFileFormat = self.fileFormatter._fmt.split(':')[:-1]
            # A space before the previous divider because a format always exists
            prevFileFormat += [' ' + new_str]
            updatedFileFormat = ':'.join(prevFileFormat)
            self.fileHandler.setFormatter(logging.Formatter(updatedFileFormat))

    def set_format(self, fmt):
        """
        Set format after instantiation.
        """
        self.streamHandler.setFormatter(logging.Formatter(fmt))
        if hasattr(self, 'fileHandler'):
            self.fileHandler.setFormatter(logging.Formatter(fmt))

    def reset_formatters(self):
        """
        Resort back to initial formatting.
        """
        if hasattr(self, 'fileHandler'):
            self.fileHandler.setFormatter(self.fileFormatter)
        self.streamHandler.setFormatter(self.consoleFormatter)

    def mute(self):
        """
        Stop logging to stdout.
        """
        prev_level = self.streamHandler.level
        self.streamHandler.level = float('inf')
        return prev_level

    def unmute(self, level):
        """
        Resume logging to stdout.
        """
        self.streamHandler.level = level

    def redirect_out(self, stream):
        """
        Redirect all logging output to `stream`.
        """
        self.altStream = stream
        self.altStreamHandler = logging.StreamHandler(self.altStream)
        super().addHandler(self.altStreamHandler)

    def stop_redirect_out(self):
        """
        Stop redirecting output to alternate stream.
        """
        if self.altStream is None:
            raise Exception('No existing redirection.')
        else:
            self.altStreamHandler.flush()
            super().removeHandler(self.altStreamHandler)


# -----------------------------------
# Forming the logger                #
# -----------------------------------
logger = ParlaiLogger(name=__name__)


def set_log_level(level):
    logger.setLevel(level)


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


def debug(*args, **kwargs):
    return logger.debug(*args, **kwargs)


def error(*args, **kwargs):
    return logger.error(*args, **kwargs)


def warn(*args, **kwargs):
    return logger.warn(*args, **kwargs)


def get_all_levels():
    levels = set(logging._nameToLevel.keys())
    levels.remove('WARNING')
    return [l.lower() for l in levels]
