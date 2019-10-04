#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from parlai.core import build_data
import unittest
import parlai.core.testing_utils as testing_utils
import multiprocessing
from parlai.core.params import ParlaiParser


@testing_utils.skipUnlessGPU
class TestBuildData(unittest.TestCase):
    """
    Basic tests on the build_data.py download_multiprocess.
    """

    dest_filenames = ('mnist0.tar.gz', 'mnist1.tar.gz', 'mnist2.tar.gz')

    def setUp(self):
        self.datapath = ParlaiParser().parse_args(print_args=False)['datapath']
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

        with testing_utils.capture_output() as stdout:
            download_results = build_data.download_multiprocess(
                urls, self.datapath, dest_filenames=self.dest_filenames
            )
        stdout = stdout.getvalue()

        output_filenames, output_statuses, output_errors = zip(*download_results)
        self.assertEqual(
            output_filenames,
            self.dest_filenames,
            f'output filenames not correct\n{stdout}',
        )
        self.assertEqual(
            output_statuses,
            (200, 403, 403),
            f'output http statuses not correct\n{stdout}',
        )

    def test_download_multiprocess_chunks(self):
        # Tests that the three finish downloading but may finish in any order
        urls = [
            'https://parl.ai/downloads/mnist/mnist.tar.gz',
            'https://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
            'https://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
        ]

        with testing_utils.capture_output() as stdout:
            download_results = build_data.download_multiprocess(
                urls, self.datapath, dest_filenames=self.dest_filenames, chunk_size=1
            )
        stdout = stdout.getvalue()

        output_filenames, output_statuses, output_errors = zip(*download_results)

        self.assertIn('mnist0.tar.gz', output_filenames, f'missing file:\n{stdout}')
        self.assertIn('mnist1.tar.gz', output_filenames, f'missing file:\n{stdout}')
        self.assertIn('mnist2.tar.gz', output_filenames, f'missing file:\n{stdout}')
        self.assertIn(200, output_statuses, f'unexpected error code:\n{stdout}')
        self.assertIn(403, output_statuses, f'unexpected error code:\n{stdout}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    unittest.main()
