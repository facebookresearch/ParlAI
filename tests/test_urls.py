#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import unittest
import requests
import importlib
import os
import warnings
import sys

TO_SKIP = [
    './parlai/tasks/__pycache__',
    './parlai/tasks/interactive',
    './parlai/tasks/fromfile',
    './parlai/tasks/taskntalk',
    './parlai/tasks/decanlp',
    './parlai/tasks/integration_tests',
    './parlai/tasks/dialog_babi_plus'

]

SPECIFIC_BUILDS = {
    './parlai/tasks/opensubtitles': ['build_2009', 'build_2018'],
    './parlai/tasks/coco_caption': ['build_2014', 'build_2017']
}

GOOGLE = [
    './parlai/tasks/dialogue_nli',
    './parlai/tasks/qacnn',
    './parlai/tasks/qangaroo',
    './parlai/tasks/qadailymail',
]

def test_url(url):
    session = requests.Session()
    if '.' not in url:
        # Google Drive URL testing
        URL = 'https://docs.google.com/uc?export=download'
        response = session.head(
            URL, params={'id': url}, stream=True
        )
    else:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'
        }
        response = session.head(
            url, allow_redirects=True, headers=headers
        )
    status = response.status_code
    session.close()

    return status

class TestUtils(unittest.TestCase):
    def test_http_response(self):
        sys.path.insert(0, './lib')
        tasks = [f.path for f in os.scandir('./parlai/tasks') if f.is_dir()]
        for task in tasks:
            if task in TO_SKIP:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                if task in SPECIFIC_BUILDS:
                    for build in SPECIFIC_BUILDS[task]:
                        mod = importlib.import_module((task[2:].replace('/', '.') + '.' + build))
                        for url in mod.URLS:
                            with self.subTest(f"{task}: {url}"):
                                self.assertEqual(test_url(url), 200)
                else:
                    mod = importlib.import_module((task[2:].replace('/', '.') + '.build'))
                    for url in mod.URLS:
                        with self.subTest(f"{task}: {url}"):
                            self.assertEqual(test_url(url), 200)
                    


if __name__ == '__main__':
    unittest.main()
