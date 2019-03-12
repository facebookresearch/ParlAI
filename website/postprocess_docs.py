#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

NEEDLE1 = '<a href="#" class="icon icon-home"> ParlAI'
NEEDLE2 = '<a href="index.html" class="icon icon-home"> ParlAI'
REPLACEMENT = """
<a href="/" style="float: left">
    <img style="padding: 0px; background-color: #fff; width: 53px; height: 53px; margin-left: 70px;" src="/static/img/icon.png">
</a>
<a href="/" style="color: #000; float: left; margin-top: 12px; font-size: 20px; font-weight: 600">
    ParlAI
</a>
"""  # noqa: E501


if __name__ == '__main__':
    for root, dirs, files in os.walk("build/docs/"):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                print("Postprocessing ", file_path)
                with open(file_path, 'r') as fin:
                    content = fin.read()
                    content = content.replace(NEEDLE1, REPLACEMENT)
                    content = content.replace(NEEDLE2, REPLACEMENT)
                with open(file_path, 'w') as fout:
                    fout.write(content)
