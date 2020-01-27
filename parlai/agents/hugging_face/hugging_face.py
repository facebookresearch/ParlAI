#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration with Hugging Face Transformers.

Please see <https://huggingface.co/transformers/>.
Currently, the only implementation is GPT2. To use this model,
run with `-m hugging_face/gpt2`.
"""

from parlai.agents.hugging_face.dict import HuggingFaceDictionaryAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent

try:
    import transformers  # noqa: F401
except ImportError as e:
    raise e(
        'Please run `pip install transformers`.'
    )


class HuggingFaceAgent:
    raise RuntimeError(
        '`-m hugging_face` is not a valid choice. Please run with '
        '`-m hugging_face/gpt2`.'
    )
