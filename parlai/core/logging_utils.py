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
    def __init__(self, name, filename, level, format):
        super().__init__(name, level)
        # Setting these here so that never have to import logging everywhere
        self.NOTSET = logging.NOTSET
        self.DEBUG = logging.DEBUG
        self.INFO = logging.INFO
        self.WARNING = logging.WARNING
        self.ERROR = logging.ERROR
        self.CRITICAL = logging.CRITICAL

        # Logging to a file
        self.fileHandler = logging.FileHandler(filename)
        self.fileHandler.level = logging.INFO  # Log to file levels: info and above
        super().addHandler(self.fileHandler)
        self.fileFormatter = logging.Formatter(format)
        self.fileHandler.setFormatter(self.fileFormatter)

        # Logging to stdout
        self.streamHandler = logging.StreamHandler(sys.stdout)
        self.streamHandler.level = (
            logging.DEBUG
        )  # Log to stdout levels: debug and above
        super().addHandler(self.streamHandler)

    def change_formatters(self, prefix):
        # change both handler formatter to add a prefix
        new_str = prefix + " " + '%(message)s'
        prevFileFormat = self.fileFormatter._fmt.split('::')[:-1]
        prevFileFormat += [' ' + new_str]
        updatedFileFormat = '::'.join(prevFileFormat)
        self.fileHandler.setFormatter(logging.Formatter(updatedFileFormat))
        self.streamHandler.setFormatter(logging.Formatter(new_str))

    def reset_formatters(self):
        self.fileHandler.setFormatter(self.fileFormatter)
        self.streamHandler.setFormatter(logging.Formatter('%(message)s'))

    def getStreamHandler(self):
        return self.streamHandler

    def mute_stdout(self):
        self.streamHandler.level = float('inf')

    def unmute_stdout(self):
        self.streamHandler.level = logging.DEBUG

    def redirect_out(self, stream):
        self.altStream = stream
        self.altStreamHandler = logging.StreamHandler(self.altStream)
        super().addHandler(self.altStreamHandler)

    def stop_redirect_out(self):
        self.altStreamHandler.flush()
        super().removeHandler(self.altStreamHandler)


# -----------------------------------
# Forming the logger "The specifics" #
# -----------------------------------

# Background work
today = date.today().strftime("%d/%m/%Y").split("/")
day, month, year = today[0], today[1], today[2]
proj_root = str(Path(__file__).parent.parent.parent)
log_dir_path = proj_root + "/" + "Logs/" + year + "/" + month
log_fname = year + "-" + month + "-" + day + ".log"
if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)
log_loc = os.path.join(proj_root, log_dir_path, log_fname)

# ParlaiLogger obj creation
parlai_format = '%(asctime)s :: %(pathname)s :: %(funcName)s :: %(lineno)d :: %(levelname)s :: %(message)s'
logger = ParlaiLogger(
    name=__name__, filename=log_loc, level=logging.INFO, format=parlai_format
)
