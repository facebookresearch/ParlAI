# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import urllib.request
from parlai.core.utils import ProgressLogger


igc_file = 'IGC_crowd_test.csv'
logger = ProgressLogger(should_humanize=False, throttle=0.1)
with open(igc_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        logger.log(i, 2592)
        img_id = row[0]
        url = row[2]
        try:
            urllib.request.urlretrieve(
                url,
                'igc_images/{}.jpg'.format(
                    img_id))
        except Exception:
            print('{} did not work'.format(url))
