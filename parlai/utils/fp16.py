#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility methods for mixed precision training.
"""
from parlai.utils.misc import warn_once

from itertools import chain

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise ImportError('Parlai requires pytorch. Go to http://pytorch.org to install.')


###################################################
## General Utilities
###################################################


class FP16SafeCrossEntropy(torch.nn.Module):
    """
    FP16-safe cross entropy loss.

    This avoids overflow in the softmax by doing the operation in FP32.
    """

    def __init__(self, ignore_index, reduction='none'):
        super().__init__()
        self.NULL_IDX = ignore_index
        self.reduction = reduction

    def forward(self, scores, targets):
        return F.nll_loss(
            F.log_softmax(scores, 1, dtype=torch.float32),
            targets,
            ignore_index=self.NULL_IDX,
            reduction=self.reduction,
        )


def clip_grad_norm(params, max_norm):
    """
    Clips grad norm.
    """
    params = list(params)
    if len(params) == 1:
        p = params[0]
        grad_norm = torch.norm(p)
        if grad_norm > max_norm > 0:
            clip_coef = max_norm / (grad_norm + 1e-6)
            p.mul_(clip_coef)
        return grad_norm
    elif max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )


def has_overflow(grad_norm):
    """
    Detect inf and NaN in grad_norm.
    """
    if grad_norm == float('inf') or grad_norm != grad_norm:
        return True
    return False


###################################################
## APEX Wrapper
###################################################


def fp16_optimizer_wrapper(
    optimizer: torch.optim.Optimizer,  # type: ignore
    verbose: bool = False,
    dynamic_loss_scale: bool = True,
    loss_initial_scale: float = 2.0 ** 17,
):
    """
    Wrap the an optimizer with FP16 loss scaling protection.

    Requires apex to be installed. Will throw an ImportError if it is not.

    :param optimizer:
        Any torch optimizer
    :param bool verbose:
        Enables verbose output in the FP16 optimizer. Turning this on can help
        debug when FP16 is underperforming.
    :param bool dynamic_loss_scaling:
        FP16 requires loss scaling to avoid underflows. It is recommended this
        stays on, but advanced users may want it off.
    :param float loss_initial_scale:
        Initial loss scaling. Default chosen empirically, but models with very low
        or high loss values may need this adjusted. Stick with powers of 2.

    :returns:
        An APEX FP16 optimizer. Please note this has different requirements on
        how backward() and step() are called.
    """
    try:
        import apex.fp16_utils
    except ImportError:
        raise ImportError(
            'No fp16 support without apex. Please install it from '
            'https://github.com/NVIDIA/apex'
        )
    return apex.fp16_utils.FP16_Optimizer(
        optimizer,
        dynamic_loss_scale=dynamic_loss_scale,
        verbose=verbose,
        # TODO: We may later want to remove this flag. Right now it
        # empirically improves the first few backward passes, but future APEX
        # improvements may make this unnecessary.
        dynamic_loss_args={'init_scale': loss_initial_scale},
    )


def fp16_apex_available() -> bool:
    # TODO: deprecate this function parlai_fp16_optimizer is available
    try:
        import apex.fp16_utils  # noqa: F401

        return True
    except ImportError:
        warn_once(
            'You set --fp16 true, but fp16 is unavailable. To use fp16, please '
            'install APEX from https://github.com/NVIDIA/apex.'
        )
        return False


###################################################
## ParlAI Wrappers
###################################################


class DynamicLossScaler(object):
    """
    Shamelessly stolen from Fairseq.

    Dynamically adjusts the loss scaling factor. Useful for mixed-precision training.
    Fairseq implementation can be found here:
    <https://github.com/pytorch/fairseq/blob/master/fairseq/optim/fp16_optimizer.py>
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 15,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        tolerance: float = 0.05,
        threshold: float = None,
    ):
        """
        :param init_scale:
            Initial loss scale.
        :param scale_factor:
            Factor by which to increase or decrease loss scale.
        :param scale_window:
            If we do not experience overflow in scale_window iterations,
            loss scale will increase by scale_factor.
        :param tolerance:
            Pct of iterations that have overflowed after which we must
            decrease the loss scale
        :param threshold:
            If not None, loss scale will decrease below this threshold
        """
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold

        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0

    def update_scale(self, overflow: bool):
        """
        Update the loss scale.

        If overflow exceeds our tolerance, we decrease the loss scale. If the number of
        iterations since the last overflow exceeds the scale window, we increase the
        loss scale.
        """
        iter_since_rescale = self._iter - self._last_rescale_iter

        if overflow:
            # calculate how often we overflowed already
            self._last_overflow_iter = self._iter
            self._overflows_since_rescale += 1
            pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                # decrease loss scale by the scale factor
                self._decrease_loss_scale()
                # reset iters
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            # increase the loss scale by scale factor
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter

        self._iter += 1

    def _decrease_loss_scale(self):
        """
        Decrease the loss scale by self.scale_factor.

        NOTE: the loss_scale will not go below self.threshold.
        """
        self.loss_scale /= self.scale_factor
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)


class MemoryEfficientFP16Optimizer(torch.optim.Optimizer):
    """
    Wrap an optimizer to perform memory-efficient mixed precision training.

    This class wraps an optimizer to perform FP16 training.
    This implementation is heavily based on the Fairseq implementation
    of `MemoryEfficientFP16Optimizer`, which can be found here:
    <https://github.com/pytorch/fairseq/blob/master/fairseq/optim/fp16_optimizer.py#L382>

    :param params:
        Model parameters
    :param optimizer:
        Any torch optimizer
    :param float loss_initial_scale:
        Initial loss scaling. Default chosen empirically, but models with very low
        or high loss values may need this adjusted. Stick with powers of 2
    :param float min_loss_scale:
        Throws an error if your loss scale goes below this threshold
    """

    def __init__(
        self,
        init_optimizer: torch.optim.Optimizer,  # type: ignore
        loss_initial_scale: float = 2.0 ** 17,
        min_loss_scale: float = 1e-4,
    ):
        self.optimizer = init_optimizer
        # TODO: we should probably set a bunch of these args with opt?
        self.min_loss_scale = min_loss_scale
        self.scaler = DynamicLossScaler(init_scale=loss_initial_scale)

    @property
    def params(self):
        """
        Return an iterable of the parameters held by the optimizer.
        """
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        self.optimizer.__setstate__(state)

    def __repr__(self):
        self.optimizer.__repr__()

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def _unscale_grads(self, multiply_grads=1.0):
        if self._grads_are_scaled:
            self._grads_are_scaled = False

            # correct for dynamic loss scaler
            self.multiply_grads(multiply_grads / self.scaler.loss_scale)
        else:
            assert multiply_grads == 1.0

    def clip_master_grads(self, gradient_clip):
        """
        Clips gradient norm and updates dynamic loss scaler.

        Returns -1 if the most recently computed gradients overflowed.
        """
        self._unscale_grads()
        grad_norm = clip_grad_norm(self.params, gradient_clip)
        # detect overflow and adjust loss scale
        overflow = has_overflow(grad_norm)
        self.scaler.update_scale(overflow)
        if overflow:
            if self.scaler.loss_scale <= self.min_loss_scale:
                # Use FloatingPointError as an uncommon error that parent
                # functions can safely catch to stop training.
                raise FloatingPointError(
                    (
                        'Minimum loss scale reached ({}). Your loss is probably exploding. '
                        'Try lowering the learning rate, using gradient clipping or '
                        'increasing the batch size.'
                    ).format(self.min_loss_scale)
                )
            warn_once(f'[ Overflow: setting loss scale to {self.scaler.loss_scale} ]')
            # TODO: dso we want to zero grad here?
            self.zero_grad()
            return -1

        return grad_norm

    def update_master_grads(self):
        # No-op
        pass

    def multiply_grads(self, c):
        """
        Multiplies grads by a constant `c`.
        """
        if self._grads_are_scaled:
            self._unscale_grads(c)
        else:
            for p in self.params:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def backward(self, loss, update_master_grads=False):
        """
        Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to a regular backwards call , this function dynamically scales the loss
        to avoid gradient underflow.
        """
        loss = loss * self.scaler.loss_scale
        loss.backward()
        self._grads_are_scaled = True

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        self._unscale_grads()
        self.optimizer.step(closure)

    def state_dict(self):
        """
        Return the optimizer's state dict.
        """
        state_dict = self.optimizer.state_dict()
        state_dict['loss_scale'] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Load an optimizer state dict.

        Override from PyTorch implementation to avoid casting to FP32.
        """
        if 'loss_scale' in state_dict:
            self.scaler.loss_scale = state_dict['loss_scale']

        self.optimizer.load_state_dict(state_dict)

        # Hack: PyTorch automatically casts the optimizer state to match the
        # type of the current parameters. But with memory efficient fp16 the
        # params are FP16 while the optimizer state is FP32 and we don't want
        # to cast. A workaround is to manually copy back the original state
        # after the optimizer has been loaded.
        groups = self.optimizer.param_groups
        saved_groups = state_dict['param_groups']
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain(*(g['params'] for g in saved_groups)),
                chain(*(g['params'] for g in groups)),
            )
        }
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                self.optimizer.state[param] = v

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()
        self._grads_are_scaled = False
