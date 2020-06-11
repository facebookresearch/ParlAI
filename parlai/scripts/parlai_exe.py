#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Global ParlAI command line launcher.
"""
from parlai.utils.strings import colorize, name_to_classname
import importlib
import os
import sys


def display_image():
    if os.environ.get('PARLAI_DISPLAY_LOGO') == 'OFF':
        return
    logo = colorize('ParlAI - Dialogue Research Platform', 'labels')
    print(logo)


def Parlai():
    # List of mappings from a command name to the script to import and class to call.
    maps = {
        'dd': ('parlai.scripts.display_data', 'DisplayData'),
        'train': ('parlai.scripts.train_model', 'TrainModel'),
        'eval': ('parlai.scripts.eval_model', 'EvalModel'),
        'i': ('parlai.scripts.interactive', 'Interactive'),
    }

    if len(sys.argv) > 1 and sys.argv[1] != '--help':
        command = sys.argv[1]
    else:
        print("no command given")
        print("\nMain ParlAI commands:")
        for c in maps:
            command = maps[c][0][maps[c][0].rfind('.') + 1 :]
            print("  " + command + " (" + c + ")")
        exit()

    try:
        # Add user-specified script mappings from parlai_internal, if available.
        module_name = "parlai_internal.scripts.parlai_scripts"
        module = importlib.import_module(module_name)
        module.get_script_mappings(maps)
    except ImportError:
        pass

    if command in maps:
        class_name = name_to_classname(maps[command][1])
        script = maps[command][0]
    else:
        script = "parlai.scripts.%s" % command
        class_name = name_to_classname(command)

    try:
        module = importlib.import_module(script)
        model_class = getattr(module, class_name)
    except ImportError:
        print(command + " not found")
        exit()

    display_image()
    sys.argv = sys.argv[1:]  # remove parlai arg
    model_class.main()


if __name__ == '__main__':
    Parlai()
