#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import logging
from pathlib import Path
from datetime import date

INFO = logging.INFO
DEBUG = logging.DEBUG
WARN_LEVEL = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


class ParlaiLogger(logging.Logger):
    def __init__(
        self,
        name,
        console_level,
        console_format=None,
        file_format=None,
        file_level=INFO,
        filename=None,
    ):
        """
        Initialize the logger object
        :param name: Name of the logger
        :param console_level: min. Level of messages logged to console
        :param console_format: The format of messages logged to the console. Simple stdout is used if None specified
        :param file_format: The format of messages logged to the file (A default is used if None specified)
        :param file_level: min. Level of messages logged to the file
        :param filename: The file the logs are written to
        """
        super().__init__(name, console_level)  # can be initialized with any level

        # Logging to a file
        if filename:
            # Default format used if no file format provided
            if file_format is None:
                file_format = '%(asctime)s :: %(levelname)s :: %(message)s'
            self.fileHandler = logging.FileHandler(filename)
            self.fileHandler.level = (
                file_level
            )  # Log to file levels: file_level and above
            self.fileFormatter = logging.Formatter(file_format)
            self.fileHandler.setFormatter(self.fileFormatter)
            super().addHandler(self.fileHandler)

        # Logging to stdout
        self.streamHandler = logging.StreamHandler(sys.stdout)
        self.streamHandler.level = (
            console_level
        )  # Log to stdout levels: console_level and above
        if console_format is not None:
            self.consoleFormatter = logging.Formatter(console_format)
            self.streamHandler.setFormatter(self.consoleFormatter)
        super().addHandler(self.streamHandler)

        # To be used with testing_utils.capture_output()
        self.altStream = None

    def log(self, msg, level=INFO):
        """ Default Logging function """
        super().log(level, msg)

    def add_file_handler(self, filename, level=INFO, format=None):
        """
        Add a file handler to the logger object
        Use case: When logging using the logger object instead of instantiating a new ParlaiLogger
                  this function might  be useful to add a filehandler on the go
        Only does so if there is no file handler existing
        """
        if not hasattr(self, 'fileHandler'):
            if format is None:
                file_format = '%(asctime)s :: %(levelname)s :: %(message)s'
            self.fileHandler = logging.FileHandler(filename)
            self.fileHandler.level = level  # Log to file levels: level and above
            self.fileFormatter = logging.Formatter(file_format)
            self.fileHandler.setFormatter(self.fileFormatter)
            super().addHandler(self.fileHandler)
        else:
            self.info("ParlaiLogger: A filehandler already exists")

    def add_format_prefix(self, prefix):
        """Include `prefix` in all future logging statements"""
        # change both handler formatters to add a prefix
        new_str = prefix + " " + '%(message)s'
        self.streamHandler.setFormatter(logging.Formatter(new_str))
        if hasattr(self, 'fileHandler'):
            prevFileFormat = self.fileFormatter._fmt.split('::')[:-1]
            prevFileFormat += [' ' + new_str]
            updatedFileFormat = '::'.join(prevFileFormat)
            self.fileHandler.setFormatter(logging.Formatter(updatedFileFormat))

    def reset_formatters(self):
        """Resort back to initial formatting"""
        if hasattr(self, 'fileHandler'):
            self.fileHandler.setFormatter(self.fileFormatter)
        self.streamHandler.setFormatter(logging.Formatter('%(message)s'))

    def mute(self):
        """Stop logging to stdout"""
        self.streamHandler.level = float('inf')

    def unmute(self):
        """Resume logging to stdout"""
        self.streamHandler.level = logging.DEBUG

    def redirect_out(self, stream):
        """Redirect all logging output to `stream`"""
        self.altStream = stream
        self.altStreamHandler = logging.StreamHandler(self.altStream)
        super().addHandler(self.altStreamHandler)

    def stop_redirect_out(self):
        """Stop redirecting output to alternate stream"""
        if self.altStream is None:
            raise Exception('No existing redirection.')
        else:
            self.altStreamHandler.flush()
            super().removeHandler(self.altStreamHandler)


# -----------------------------------
# Forming the logger                #
# -----------------------------------
logger = ParlaiLogger(name=__name__, console_level=DEBUG)
