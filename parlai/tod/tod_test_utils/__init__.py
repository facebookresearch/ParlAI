#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.utils.logging as logging

DEFAULT_DATAPATH = None


def finalize_registration(path_manager):
    try:
        import torch.manifold.patch  # noqa: F401
        from iopath.fb.manifold import ManifoldPathHandler
        from nltk_data.init import init_nltk_data

        # use packaged nltk_data, will prevent downloads from github.
        init_nltk_data()

        logging.debug("Registering manifold")
        path_manager.register_handler(
            ManifoldPathHandler(max_parallel=4, timeout_sec=240, num_retries=10)
        )
        global DEFAULT_DATAPATH
        DEFAULT_DATAPATH = "manifold://parlai-datapath/tree/"
    except ImportError:
        pass
