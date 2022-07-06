#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.script import superscript_main


def main():
    # indirect call so that console_entry (see setup.py) doesn't use the return
    # value of the final command as the return code. This lets us call
    # superscript_main as a function in other places (test_script.py).
    superscript_main()


if __name__ == '__main__':
    main()
