#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from parlai.core import build_data
import unittest
import unittest.mock
import requests
import parlai.utils.testing as testing_utils
import multiprocessing
from parlai.core.params import ParlaiParser


@testing_utils.skipUnlessGPU
class TestBuildData(unittest.TestCase):
    """
    Basic tests on the build_data.py download_multiprocess.
    """

    dest_filenames = ('mnist0.tar.gz', 'mnist1.tar.gz', 'mnist2.tar.gz')

    def setUp(self):
        self.datapath = ParlaiParser().parse_args([], print_args=False)['datapath']
        self.datapath = os.path.join(self.datapath, 'build_data_pyt_data')
        os.makedirs(self.datapath, exist_ok=True)

        for d in self.dest_filenames:
            # Removing files if they are already there b/c otherwise it won't try to download them again
            try:
                os.remove(os.path.join(self.datapath, d))
            except OSError:
                pass

    def test_download_multiprocess(self):
        urls = [
            'https://parl.ai/downloads/mnist/mnist.tar.gz',
            'https://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
            'https://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
        ]

        download_results = build_data.download_multiprocess(
            urls, self.datapath, dest_filenames=self.dest_filenames
        )

        output_filenames, output_statuses, output_errors = zip(*download_results)
        self.assertEqual(
            output_filenames, self.dest_filenames, 'output filenames not correct',
        )
        self.assertEqual(
            output_statuses, (200, 403, 403), 'output http statuses not correct',
        )

    def test_download_multiprocess_chunks(self):
        # Tests that the three finish downloading but may finish in any order
        urls = [
            'https://parl.ai/downloads/mnist/mnist.tar.gz',
            'https://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
            'https://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
        ]

        download_results = build_data.download_multiprocess(
            urls, self.datapath, dest_filenames=self.dest_filenames, chunk_size=1
        )

        output_filenames, output_statuses, output_errors = zip(*download_results)

        self.assertIn('mnist0.tar.gz', output_filenames)
        self.assertIn('mnist1.tar.gz', output_filenames)
        self.assertIn('mnist2.tar.gz', output_filenames)
        self.assertIn(200, output_statuses, 'unexpected error code')
        self.assertIn(403, output_statuses, 'unexpected error code')

    def test_connectionerror_download(self):
        with unittest.mock.patch('requests.Session.get') as Session:
            Session.side_effect = requests.exceptions.ConnectTimeout
            with testing_utils.tempdir() as tmpdir:
                with self.assertRaises(RuntimeError):
                    build_data.download(
                        'http://test.com/bad', tmpdir, 'foo', num_retries=3
                    )
            assert Session.call_count == 3


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    unittest.main()
