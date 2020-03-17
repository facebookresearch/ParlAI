#!/usr/bin/env python3

"""
Unit tests for parlai.utils.torch.
"""

import torch
import numpy as np
import unittest
from parlai.utils.torch import padded_tensor, argsort, PipelineHelper, neginf


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


class TestPipelineHelper(unittest.TestCase):
    """
    Test the PipelineHelper class.
    """

    def test_guess_split(self):
        t = torch.randn(128, 5)
        assert PipelineHelper.guess_split_size(t, 8) == 2
        assert PipelineHelper.guess_split_size(t, 4) == 8
        assert PipelineHelper.guess_split_size(t, 1) == 128
        t = torch.randn(129, 5)
        assert PipelineHelper.guess_split_size(t, 8) == 2
        assert PipelineHelper.guess_split_size(t, 4) == 8
        assert PipelineHelper.guess_split_size(t, 1) == 129
        t = torch.randn(5, 128)
        assert PipelineHelper.guess_split_size(t, 8, dim=1) == 2
        assert PipelineHelper.guess_split_size(t, 4, dim=1) == 8
        assert PipelineHelper.guess_split_size(t, 1, dim=1) == 128

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

    def test_split_emptydict(self):
        # test an even worse edge case, where we have a dict of empty dicts
        t = torch.randn(32, 5)
        d = {'x': {}}
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

    def chunk_layer_iterator(self):
        pass
