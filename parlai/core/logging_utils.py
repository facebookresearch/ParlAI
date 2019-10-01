#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import logging
from pathlib import Path
from datetime import date


class ParlaiLogger(logging.Logger):
    def __init__(
        self,
        name,
        console_level,
        console_format=None,
        file_format=None,
        file_level=logging.INFO,
        filename=None,
    ):
        """
        Initialize the logger object
        :param name: Name of the logger
        :param console_level: min. Level of messages logged to console
        :param console_format: The format of messages logget to the console. Simple stdout is used if None specified
        :param file_format: The format of messages logged to the file (A default is used if None specified)
        :param file_level: min. Level of messages logged to the file
        :param filename: The file the logs are written to

        A default file is initialized if no filename was provided. It can be
        found in ParlAI/Logs/*year*/*month*/*YYYY-MM-DD*.log
        """
        # Setup logging location if none specified
        if filename is None:
            today = date.today().strftime("%d/%m/%Y").split("/")
            day, month, year = today[0], today[1], today[2]
            proj_root = str(Path(__file__).parent.parent.parent)
            log_dir_path = proj_root + "/" + "Logs/" + year + "/" + month
            log_fname = year + "-" + month + "-" + day + ".log"
            if not os.path.exists(log_dir_path):
                os.makedirs(log_dir_path)
            filename = os.path.join(proj_root, log_dir_path, log_fname)
        # Default format used if no file format provided
        if file_format is None:
            file_format = '%(asctime)s :: %(pathname)s :: %(funcName)s :: %(lineno)d :: %(levelname)s :: %(message)s'

        super().__init__(name, console_level)  # can be initialized with any level

        # Logging to a file
        self.fileHandler = logging.FileHandler(filename)
        self.fileHandler.level = file_level  # Log to file levels: file_level and above
        if len(file_format) == 2:  # If a tuple is passed in
            self.fileFormatter = logging.Formatter(file_format[0], file_format[1])
        else:
            self.fileFormatter = logging.Formatter(file_format)
        self.fileHandler.setFormatter(self.fileFormatter)
        super().addHandler(self.fileHandler)

        # Logging to stdout
        self.streamHandler = logging.StreamHandler(sys.stdout)
        self.streamHandler.level = (
            console_level
        )  # Log to stdout levels: console_level and above
        if console_format is not None:
            if len(console_format) == 2:  # If a tuple is passed in
                self.consoleFormatter = logging.Formatter(
                    console_format[0], console_format[1]
                )
            else:
                self.consoleFormatter = logging.Formatter()
            self.streamHandler.setFormatter(self.consoleFormatter)
        super().addHandler(self.streamHandler)

        # To be used with testing_utils.capture_output()
        self.altStream = None

    def add_format_prefix(self, prefix):
        """Include `prefix` in all future logging statements"""
        # change both handler formatters to add a prefix
        new_str = prefix + " " + '%(message)s'
        prevFileFormat = self.fileFormatter._fmt.split('::')[:-1]
        prevFileFormat += [' ' + new_str]
        updatedFileFormat = '::'.join(prevFileFormat)
        self.fileHandler.setFormatter(logging.Formatter(updatedFileFormat))
        self.streamHandler.setFormatter(logging.Formatter(new_str))

    def reset_formatters(self):
        """Resort back to initial formatting"""
        self.fileHandler.setFormatter(self.fileFormatter)
        self.streamHandler.setFormatter(logging.Formatter('%(message)s'))

    def mute_stdout(self):
        """Stop logging to stdout"""
        self.streamHandler.level = float('inf')

    def unmute_stdout(self):
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
logger = ParlaiLogger(
    name=__name__, console_level=logging.DEBUG, file_level=logging.INFO
)
