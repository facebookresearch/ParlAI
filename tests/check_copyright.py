#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import re
import parlai.utils.testing as testing_utils

FILENAME_EXTENSIONS = r'.*\.(rst|py|sh|js|css)$'
WHITELIST_PHRASES = ['Moscow Institute of Physics and Technology.']
COPYRIGHT = [
    "Copyright (c) Facebook, Inc. and its affiliates.",
    "This source code is licensed under the MIT license found in the",
    "LICENSE file in the root directory of this source tree.",
]


def main():
    allgood = True
    for fn in testing_utils.git_ls_files():
        # only check source files
        if not re.match(FILENAME_EXTENSIONS, fn):
            continue

        with open(fn, 'r') as f:
            src = f.read(512)  # only need the beginning

        if not src.strip():
            # skip empty files
            continue

        if any(wl in src for wl in WHITELIST_PHRASES):
            # skip a few things we don't have the copyright on
            continue

        for i, msg in enumerate(COPYRIGHT):
            if "mlb_vqa" in fn and i < 2:
                # very special exception for mlb_vqa
                continue

            if msg not in src:
                print('{} missing copyright "{}"'.format(fn, msg))
                allgood = False

    if not allgood:
        sys.exit(1)


if __name__ == '__main__':
    main()
