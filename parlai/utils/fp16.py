#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility methods for mixed precision training.
"""
from parlai.utils.misc import warn_once

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


def fp16_available() -> bool:
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
    Directly stolen from fairseq (thanks Myle)!

    TODO: Add a description here
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

    @staticmethod
    def has_overflow(grad_norm):
        """
        Detect inf and NaN in grad_norm.
        """
        if grad_norm == float('inf') or grad_norm != grad_norm:
            return True
        return False


class ParlAIFP16Optimizer(torch.optim.Optimizer):
    """
    Wrap an optimizer to perform mixed precision training.

    Based on the fairseq implementation. This wraps an optimizer to perform
    FP16 training.

    :param optimizer:
        Any torch optimizer
    :param float loss_initial_scale:
        Initial loss scaling. Default chosen empirically, but models with very low
        or high loss values may need this adjusted. Stick with powers of 2.
    """

    def __init__(
        init_optimizer: torch.optim.Optimizer,  # type: ignore
        loss_initial_scale: float = 2.0 ** 17,
    ):
        self.wrapped_optimizer = init_optimizer  # wrapped optimizer is FP32 optimizer
        self.scaler = DynamicLossScaler(init_scale=loss_initial_scale)
        # TODO: finish

    def __getstate__(self):
        return self.wrapped_optimizer.__getstate__()

    def __setstate__(self, state):
        self.wrapped_optimizer.__setstate__(state)

    def __repr__(self):
        self.wrapped_optimizer.__repr__()

    def state_dict(self):
        """
        Return the optimizer's state dict.
        """
        state_dict = self.fp32_optimizer.state_dict()
        state_dict['loss_scale'] = self.scaler.loss_scale
        return state_dict

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        for p in self.fp16_params:
            p.grad = None
        if self.has_flat_params:
            self.fp32_params.grad.zero_()
        else:
            for p32 in self.fp32_params:
                p32.grad.zero_()
        self._needs_sync = False

    def step(self, closure):
        """
        Performs a single optimization step.
        """
        self._sync_fp16_grads_to_fp32()
        self.fp32_optimizer.step(closure)

        # copy FP32 params back into FP16 model
        if self.has_flat_params:
            offset = 0
            for p in self.fp16_params:
                if not p.requires_grad:
                    continue
                numel = p.data.numel()
                p.data.copy_(
                    self.fp32_params.data[offset : offset + numel].view_as(p.data)
                )
                offset += numel
        else:
            for p, p32 in zip(self.fp16_params, self.fp32_params):
                if not p.requires_grad:
                    continue
                p.data.copy_(p32.data)

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """
        Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer instance
        (e.g., learning rate) over that found in the state_dict. This allows us to
        resume training from a checkpoint using a new set of optimizer args.
        """
        if 'loss_scale' in state_dict:
            self.scaler.loss_scale = state_dict['loss_scale']
        self.fp32_optimizer.load_state_dict(state_dict, optimizer_overrides)

    def add_param_group(self, param_group):
        self.wrapped_optimizer.add_param_group(param_group)


class ParlAIFP16MemoryEfficientOptimizer(ParlAIFP16Optimizer):
    """
    Wrap an optimizer to perform memory-efficient mixed precision training.

    Based on the fairseq implementation. This wraps an optimizer to perform
    FP16 training.

    :param optimizer:
        Any torch optimizer
    :param float loss_initial_scale:
        Initial loss scaling. Default chosen empirically, but models with very low
        or high loss values may need this adjusted. Stick with powers of 2.
    """

    def __init__(
        init_optimizer: torch.optim.Optimizer,  # type: ignore
        loss_initial_scale: float = 2.0 ** 17,
    ):
        self.wrapped_optimizer = init_optimizer
        # TODO: finish

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.wrapped_optimizer.zero_grad()
        self._grads_are_scaled = False

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        self._unscale_grads()
        self.wrapped_optimizer.step(closure)

    def state_dict(self):
        """
        Return the optimizer's state dict.
        """
        state_dict = self.wrapped_optimizer.state_dict()
        state_dict['loss_scale'] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Load an optimizer state dict.

        Override from PyTorch implementation to avoid casting to FP32.
        """
        if 'loss_scale' in state_dict:
            self.scaler.loss_scale = state_dict['loss_scale']

        self.wrapped_optimizer.load_state_dict(state_dict)

        # Hack: PyTorch automatically casts the optimizer state to match the
        # type of the current parameters. But with --memory-efficient-fp16 the
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
