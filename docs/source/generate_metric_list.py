#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.metrics import METRICS_DISPLAY_DATA


fout = open('metric_list.inc', 'w')

fout.write('| Metric | Explanation |\n')
fout.write('| ------ | ----------- |\n')
for metric, display in sorted(METRICS_DISPLAY_DATA.items()):
    fout.write(f'| `{metric}` | {display.description} |\n')

fout.close()
