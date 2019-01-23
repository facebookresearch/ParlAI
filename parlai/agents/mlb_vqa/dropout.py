#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch import nn


class EmbeddingDropout():

    def __init__(self, p=0.5):
        super(EmbeddingDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.training = True

    def forward(self, input):
        # input must be tensor
        if self.p > 0 and self.training:
            dim = input.dim()
            if dim == 1:
                input = input.view(1, -1)
            batch_size = input.size(0)
            for i in range(batch_size):
                x = np.unique(input[i].numpy())
                x = np.nonzero(x)[0]
                x = torch.from_numpy(x)
                noise = x.new().resize_as_(x)
                noise.bernoulli_(self.p)
                x = x.mul(noise)
                for value in x:
                    if value > 0:
                        mask = input[i].eq(value)
                        input[i].masked_fill_(mask, 0)
            if dim == 1:
                input = input.view(-1)

        return input


class SequentialDropout(nn.Module):

    def __init__(self, p=0.5):
        super(SequentialDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.restart = True

    def _make_noise(self, input):
        return input.new().resize_as_(input)

    def forward(self, input):
        if self.p > 0 and self.training:
            if self.restart:
                self.noise = self._make_noise(input)
                self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
                if self.p == 1:
                    self.noise.fill_(0)
                self.noise = self.noise.expand_as(input)
                self.restart = False
            return input.mul(self.noise)

        return input

    def end_of_sequence(self):
        self.restart = True

    def backward(self, grad_output):
        self.end_of_sequence()
        if self.p > 0 and self.training:
            return grad_output.mul(self.noise)
        else:
            return grad_output

    def __repr__(self):
        return type(self).__name__ + '({:.4f})'.format(self.p)


if __name__ == '__main__':

    dp = SequentialDropout(p=0.5)
    input = torch.ones(1, 10)

    dist_total = torch.zeros(1)
    output_last = dp(input)
    for _ in range(50):
        output_new = dp(input)
        dist_total += torch.dist(output_new, output_last)
        output_last = output_new

    if not torch.equal(dist_total, torch.zeros(1)):
        print('Error')
        print(dist_total)

    dp.end_of_sequence()

    dist_total = torch.zeros(1)
    for _ in range(50):
        dist_total += torch.dist(output_last, dp(input))
        dp.end_of_sequence()

    if torch.equal(dist_total, torch.zeros(1)):
        print('Error')

    ####

    dp = EmbeddingDropout(p=0.15)
    input = torch.Tensor([[1, 2, 3, 0, 0], [5, 3, 2, 2, 0]]).long()
    print(input)
    print(dp.forward(input))
