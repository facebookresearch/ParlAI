#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from iopath.common.file_io import PathManager as _PathManager
except ImportError:
    try:
        from fvcore.common.file_io import PathManagerBase as _PathManager
    except ImportError:
        raise ImportError(
            "parlai now requires iopath for some I/O operations. Please run "
            "`pip install iopath`"
        )

USE_ATOMIC_TORCH_SAVE = True

PathManager = _PathManager()

try:
    # register any internal file handlers
    import parlai_fb  # noqa: F401

    # internal file handlers can't handle atomic saving. see T71772714
    USE_ATOMIC_TORCH_SAVE = not parlai_fb.finalize_registration(PathManager)
except ModuleNotFoundError:
    USE_ATOMIC_TORCH_SAVE = True
