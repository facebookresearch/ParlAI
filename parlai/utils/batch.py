#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility file with ParlAI Batch Implementation.

Separate from TorchAgent as it technically does not require Pytorch.
"""
from parlai.core.message import Message
from parlai.utils.misc import AttrDict
import torch
from typing import Optional, List, Any


class Batch(AttrDict):
    """
    Batch is a namedtuple containing data being sent to an agent.

    This is the input type of the train_step and eval_step functions.
    Agents can override the batchify function to return an extended namedtuple
    with additional fields if they would like, though we recommend calling the
    parent function to set up these fields as a base.

    :param text_vec:
        bsz x seqlen tensor containing the parsed text data.

    :param text_lengths:
        list of length bsz containing the lengths of the text in same order as
        text_vec; necessary for pack_padded_sequence.

    :param full_text_vec:
        bsz x seqlen tensor containing full, untruncated parsed text data.

    :param text_lengths:
        list of length bsz containing the lengths of the full text in same order as
        full_text_vec; necessary for pack_padded_sequence.

    :param label_vec:
        bsz x seqlen tensor containing the parsed label (one per batch row).

    :param label_lengths:
        list of length bsz containing the lengths of the labels in same order as
        label_vec.

    :param labels:
        list of length bsz containing the selected label for each batch row (some
        datasets have multiple labels per input example).

    :param valid_indices:
        list of length bsz containing the original indices of each example in the
        batch. we use these to map predictions back to their proper row, since e.g.
        we may sort examples by their length or some examples may be invalid.

    :param candidates:
        list of lists of text. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param candidate_vecs:
        list of lists of tensors. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param image:
        list of image features in the format specified by the --image-mode arg.

    :param observations:
        the original observations in the batched order

    :param training:
        whether this batch is a train batch
    """

    batchsize: int
    text_vec: Optional[torch.LongTensor]
    text_lengths: Optional[List[int]]
    full_text_vec: Optional[torch.LongTensor]
    full_text_lengths: Optional[List[int]]
    label_vec: Optional[torch.LongTensor]
    label_lengths: Optional[List[int]]
    labels: Optional[List[str]]
    valid_indices: Optional[List[int]]
    candidates: Optional[List[List[str]]]
    candidate_vecs: Optional[List[List[torch.LongTensor]]]
    image: Optional[List[Any]]
    observations: Optional[List[Message]]
    training: Optional[bool]

    def __init__(
        self,
        text_vec=None,
        text_lengths=None,
        full_text_vec=None,
        full_text_lengths=None,
        label_vec=None,
        label_lengths=None,
        labels=None,
        valid_indices=None,
        candidates=None,
        candidate_vecs=None,
        image=None,
        observations=None,
        training=None,
        **kwargs,
    ):
        super().__init__(
            text_vec=text_vec,
            text_lengths=text_lengths,
            full_text_vec=full_text_vec,
            full_text_lengths=full_text_lengths,
            label_vec=label_vec,
            label_lengths=label_lengths,
            labels=labels,
            valid_indices=valid_indices,
            candidates=candidates,
            candidate_vecs=candidate_vecs,
            image=image,
            observations=observations,
            training=training,
            **kwargs,
        )

    def cuda(self):
        """
        Cuda appropriately.
        """
        if self.text_vec is not None:
            self.text_vec = self.text_vec.cuda()
        if self.label_vec is not None:
            self.label_vec = self.label_vec.cuda()
