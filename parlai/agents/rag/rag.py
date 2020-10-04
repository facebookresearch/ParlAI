import os

import torch
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import IdentityLayer, concat_without_padding, padded_tensor

try:
    from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
except ImportError:
    raise ImportError("Please run `pip install transformers`, the version must be >= 3.3.0 .")

