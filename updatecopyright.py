#!/usr/bin/env python

import sys

for line in sys.stdin:
    if "updatecopyright" in line:
        continue
    if "parlai_internal" in line:
        continue

    fn = line.strip()
    with open(fn, 'r') as inf:
        src = inf.read()
    origsrc = src
    src = src.replace(
        "Copyright (c) 2017-present, Facebook, Inc.",
        "Copyright (c) Facebook, Inc. and its affiliates."
    )
    src = src.replace(
        "This source code is licensed under the BSD-style license found in the",
        "This source code is licensed under the MIT license found in the"
    )
    src = src.replace(
        "LICENSE file in the root directory of this source tree. An additional grant",
        "LICENSE file in the root directory of this source tree."
    )
    srclines = src.split("\n")
    srclines = [l for l in srclines if "All rights reserved." not in l]
    srclines = [
        l for l in srclines
        if "of patent rights can be found in the PATENTS file in the same directory" not in l
    ]
    src = "\n".join(srclines)
    if src != origsrc:
        with open(fn, 'w') as outf:
            outf.write(src)


