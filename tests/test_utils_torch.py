#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for parlai.utils.torch.
"""

import torch
import numpy as np
import unittest
from parlai.utils.torch import (
    padded_tensor,
    argsort,
    PipelineHelper,
    neginf,
    IdentityLayer,
    trainable_parameters,
    total_parameters,
)


class TestTorchUtils(unittest.TestCase):
    def test_neginf(self):
        assert neginf(torch.float32) < -1e15
        assert neginf(torch.float16) > -1e15
        assert neginf(torch.float16) < -1e4

    def test_padded_tensor(self):
        # list of lists
        lol = [[1, 2], [3, 4, 5]]
        output, lens = padded_tensor(lol)
        assert np.all(output.numpy() == np.array([[1, 2, 0], [3, 4, 5]]))
        assert lens == [2, 3]
        output, _ = padded_tensor(lol, left_padded=True)
        assert np.all(output.numpy() == np.array([[0, 1, 2], [3, 4, 5]]))
        output, _ = padded_tensor(lol, pad_idx=99)
        assert np.all(output.numpy() == np.array([[1, 2, 99], [3, 4, 5]]))

    def test_argsort(self):
        keys = [5, 4, 3, 2, 1]
        items = ["five", "four", "three", "two", "one"]
        items2 = ["e", "d", "c", "b", "a"]
        torch_keys = torch.LongTensor(keys)
        assert argsort(keys, items, items2) == [
            list(reversed(items)),
            list(reversed(items2)),
        ]
        assert argsort(keys, items, items2, descending=True) == [items, items2]

        assert np.all(argsort(torch_keys, torch_keys)[0].numpy() == np.arange(1, 6))

    def test_trainable_parameters(self):
        ident = IdentityLayer()
        emb = torch.nn.Embedding(32, 8)  # 32 * 8 = 256
        emb2 = torch.nn.Embedding(32, 16)  # 32 * 16 = 128
        emb2.weight.requires_grad = False
        assert trainable_parameters(emb) == 256
        assert trainable_parameters(ident) == 0
        assert trainable_parameters(emb2) == 0
        assert trainable_parameters(torch.nn.ModuleList([ident, emb, emb2])) == 256

    def test_total_parameters(self):
        ident = IdentityLayer()
        emb = torch.nn.Embedding(32, 8)  # 32 * 8 = 256
        emb2 = torch.nn.Embedding(32, 16)  # 32 * 16 = 128
        emb2.weight.requires_grad = False
        assert total_parameters(emb) == 256
        assert total_parameters(ident) == 0
        assert total_parameters(emb2) == 512
        assert total_parameters(torch.nn.ModuleList([ident, emb, emb2])) == 768


class TestPipelineHelper(unittest.TestCase):
    """
    Test the PipelineHelper class.
    """

    def test_guess_split(self):
        t = torch.randn(128, 5)
        assert PipelineHelper.guess_split_size(t, 8) == 8
        assert PipelineHelper.guess_split_size(t, 4) == 16
        assert PipelineHelper.guess_split_size(t, 1) == 128
        t = torch.randn(129, 5)
        assert PipelineHelper.guess_split_size(t, 8) == 8
        assert PipelineHelper.guess_split_size(t, 4) == 16
        assert PipelineHelper.guess_split_size(t, 1) == 129
        t = torch.randn(5, 128)
        assert PipelineHelper.guess_split_size(t, 8, dim=1) == 8
        assert PipelineHelper.guess_split_size(t, 4, dim=1) == 16
        assert PipelineHelper.guess_split_size(t, 1, dim=1) == 128

        # tuple
        t = torch.randn(128, 5)
        assert PipelineHelper.guess_split_size((t, t), 8) == 8
        assert PipelineHelper.guess_split_size((t, t), 1) == 128

        # dict
        t = torch.randn(128, 5)
        assert PipelineHelper.guess_split_size({'x': t, 'y': t}, 8) == 8
        assert PipelineHelper.guess_split_size({'x': t, 'y': t}, 1) == 128

        with self.assertRaises(TypeError):
            t = torch.randn(128, 5)
            PipelineHelper.guess_split_size([t, t], 1)

    def test_split_tensor(self):
        t = torch.randn(32, 5)
        for st in PipelineHelper.split(t, 8):
            assert st.shape == (8, 5)
        a, b = PipelineHelper.split(t, 17)
        assert a.shape == (17, 5)
        assert b.shape == (15, 5)

    def test_split_tuple(self):
        t = torch.randn(32, 5)
        tup = (t, t, t)
        for stup in PipelineHelper.split(tup, 8):
            assert isinstance(stup, tuple)
            assert len(stup) == 3
            for i in range(3):
                assert stup[i].shape == (8, 5)

    def test_split_dict(self):
        t = torch.randn(32, 5)
        d = {'x': t, 'y': t}
        for sd in PipelineHelper.split(d, 8):
            assert isinstance(sd, dict)
            assert 'x' in sd
            assert 'y' in sd
            assert sd['x'].shape == (8, 5)
            assert sd['y'].shape == (8, 5)

    def test_split_complex(self):
        t = torch.randn(32, 5)
        item = (t, {'x': t, 'y': t})
        for sitem in PipelineHelper.split(item, 8):
            assert isinstance(sitem, tuple)
            assert len(sitem) == 2
            left, right = sitem
            assert isinstance(left, torch.Tensor)
            assert left.shape == (8, 5)
            assert isinstance(right, dict)
            assert 'x' in right
            assert 'y' in right
            assert right['x'].shape == (8, 5)
            assert right['y'].shape == (8, 5)

    def test_split_emptydict(self):
        # test a horrible edge case where d is an empty dict, and we need to
        # return a BUNCH of empty dicts
        t = torch.randn(32, 5)
        d = {}
        tup = (t, d)
        items = PipelineHelper.split(tup, 8)
        assert len(items) == 4
        for item in items:
            assert isinstance(item, tuple)
            a, b = item
            assert isinstance(a, torch.Tensor)
            assert a.shape == (8, 5)
            assert isinstance(b, dict)
            assert b == {}

    def test_split_badcases(self):
        # test some cases that cause infinite loops if we don't catch them.
        t = torch.randn(32, 5)
        with self.assertRaises(ValueError):
            PipelineHelper.split((t, {'x': {}}), 8)
        with self.assertRaises(ValueError):
            PipelineHelper.split((t, {'y': ()}), 8)

    def test_join_tensor(self):
        t = torch.randn(8, 5)
        j = PipelineHelper.join([t, t, t, t])
        assert isinstance(j, torch.Tensor)
        assert j.shape == (32, 5)

        j = PipelineHelper.join([t, t], dim=1)
        assert isinstance(j, torch.Tensor)
        assert j.shape == (8, 10)

    def test_join_tuple(self):
        tup = (torch.randn(8, 5), torch.randn(8, 2))
        chunks = [tup, tup]
        j = PipelineHelper.join(chunks)
        assert isinstance(j, tuple)
        assert len(j) == 2
        a, b = j
        assert a.shape == (16, 5)
        assert b.shape == (16, 2)

    def test_join_dict(self):
        chunk = {'x': torch.randn(8, 5), 'y': torch.randn(8, 2)}
        chunks = [chunk, chunk]
        j = PipelineHelper.join(chunks)
        assert isinstance(j, dict)
        assert len(j) == 2
        assert 'x' in j
        assert 'y' in j
        assert isinstance(j['x'], torch.Tensor)
        assert isinstance(j['y'], torch.Tensor)
        assert j['x'].shape == (16, 5)
        assert j['y'].shape == (16, 2)

    def test_join_complex(self):
        d = {'x': torch.randn(8, 5), 'y': torch.randn(8, 2)}
        t = torch.Tensor(8, 3)
        tup = (t, d)
        chunks = [tup, tup]
        j = PipelineHelper.join(chunks)
        assert isinstance(j, tuple)
        assert len(j) == 2
        left, right = j
        assert isinstance(left, torch.Tensor)
        assert left.shape == (16, 3)
        assert isinstance(right, dict)
        assert len(right) == 2
        assert 'x' in right
        assert 'y' in right
        assert right['x'].shape == (16, 5)
        assert right['y'].shape == (16, 2)

    def test_schedule_work_items(self):
        # test that we schedule things correctly
        # pretend we have 8 layers and 4 gpus, and they are unevenly distributed
        model = torch.nn.ModuleList()
        for i in range(8):
            layer = IdentityLayer()
            if i == 0:
                layer._mp_gpu = 'cuda:0'
            elif i in (1, 2, 3):
                layer._mp_gpu = 'cuda:1'
            elif i in (4, 5):
                layer._mp_gpu = 'cuda:2'
            elif i in (6, 7):
                layer._mp_gpu = 'cuda:3'
            model.append(layer)

        # there are 2 chunks, each 16 x 7 in size
        chunks = PipelineHelper.split(torch.randn(32, 7), 16)

        work_items = list(PipelineHelper.schedule_work_items(model, chunks))
        assert len(work_items) == 8
        assert work_items[0].layer_nos == [0] and work_items[0].chunk_idx == 0
        assert work_items[1].layer_nos == [1, 2, 3] and work_items[1].chunk_idx == 0
        assert work_items[2].layer_nos == [0] and work_items[2].chunk_idx == 1
        assert work_items[3].layer_nos == [4, 5] and work_items[3].chunk_idx == 0
        assert work_items[4].layer_nos == [1, 2, 3] and work_items[4].chunk_idx == 1
        assert work_items[5].layer_nos == [6, 7] and work_items[5].chunk_idx == 0
        assert work_items[6].layer_nos == [4, 5] and work_items[6].chunk_idx == 1
        assert work_items[7].layer_nos == [6, 7] and work_items[7].chunk_idx == 1
