#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import itertools
import parlai.utils.testing as testing_utils


def product_dict(dictionary):
    keys = dictionary.keys()
    vals = dictionary.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


@testing_utils.skipIfCircleCI
class TestIGC(unittest.TestCase):
    """
    Basic tests for IGC Teacher to ensure it works for every given variant.
    """

    def test_display_data(self):

        def _test_display_output(opt, ep_ex_counts):
            output = testing_utils.display_data(opt)
            stats = [o.split('\n')[-2] for o in output]

            for i, ((ep, ex), out) in enumerate(zip(ep_ex_counts, stats)):
                self.assertEqual(
                    out,
                    f'[ loaded {ep} episodes with a total of {ex} examples ]',
                    output[i]
                )

        variant_to_ep_ex_counts = {
            'IGC': [(2902, 4353), (324, 486), (5182, 7773)],
            'ResponseOnly': [(1451, 1451), (162, 162), (2591, 2591)],
            'QuestionOnly': [(1451, 1451), (162, 162), (2591, 2591)]
        }
        igc_multi_ref_opts = [True, False]
        image_modes = ['no_image_model', 'ascii']

        for task, ep_ex_counts in variant_to_ep_ex_counts.items():
            for igc_multi_ref in igc_multi_ref_opts:
                for image_mode in image_modes:
                    opt = {
                        'task': f'igc:{task}',
                        'igc_multi_ref': igc_multi_ref,
                        'image_mode': image_mode
                    }
                    _test_display_output(opt, ep_ex_counts)


if __name__ == '__main__':
    unittest.main()
