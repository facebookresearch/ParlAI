#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility methods for mixed precision training.
"""
from parlai.utils.misc import warn_once

from itertools import chain
import math

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
    try:
        import apex.fp16_utils  # noqa: F401

        return True
    except ImportError:
        warn_once(
            'You set --fp16 true with --fp16-impl apex, but fp16 '
            'with apex is unavailable. To use apex fp16, please '
            'install APEX from https://github.com/NVIDIA/apex.'
        )
        return False


###################################################
## Memory Efficient Wrappers
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

    This allows you to train bigger models on a single GPU, but can be unstable.
    Opt for the APEX implementation if you do not have concerns about memory.

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
        # TODO: set some of these args with opt
        self.min_loss_scale = min_loss_scale
        self.scaler = DynamicLossScaler(init_scale=loss_initial_scale)

    @staticmethod
    def compatible_optimizers():
        """
        List of compatible optimizers.
        """
        return [
            'adam',
            'mem_eff_adam',
            'adafactor',
        ]

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
        state_dict['loss_scaler'] = self.scaler
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Load an optimizer state dict.

        Override from PyTorch implementation to avoid casting to FP32.
        """
        if 'loss_scaler' in state_dict:
            # init from the state dict
            self.scaler = state_dict['loss_scaler']

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

    @property
    def loss_scale(self):
        """
        Convenience function which TorchAgent calls to get current scale value.
        """
        return self.scaler.loss_scale

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()
        self._grads_are_scaled = False


###################################################
## Optimizers for Memory Efficient FP16
###################################################


class MemoryEfficientFP16Adam(torch.optim.Adam):
    """
    Override from Pytorch implementation to ensure aggregations done in FP32.
    """

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()  # NOTE: cast to FP32 here
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()  # NOTE: cast to FP32 here

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p_data_fp32)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss


class Adafactor(torch.optim.Optimizer):
    """
    Implements Adafactor algorithm.

    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Taken from the fairseq implementation, which can be found here:
    <https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py>.

    :param params (iterable):
        iterable of parameters to optimize or dicts defining parameter groups
    :param lr (float, optional):
        external learning rate (default: None)
    :param eps (tuple[float, float]):
        regularization constans for square gradient and parameter scale
        respectively (default: (1e-30, 1e-3))
    :param clip_threshold (float):
        threshold of root mean square of final gradient update
        (default: 1.0)
    :param decay_rate (float):
        coefficient used to compute running averages of square gradient
        (default: -0.8)
    :param beta1 (float):
        coefficient used for computing running averages of gradient
        (default: None)
    :param weight_decay (float, optional):
        weight decay (L2 penalty) (default: 0)
    :param scale_parameter (bool):
        if true, learning rate is scaled by root mean square of parameter
        (default: True)
    :param relative_step (bool):
        if true, time-dependent learning rate is computed instead of external
        learning rate (default: True)
    :param warmup_init (bool):
        time-dependent learning rate computation depends on whether warm-up
        initialization is being used (default: False)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super(Adafactor, self).__init__(params, defaults)

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            min_step = (
                1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'][1], param_state['RMS'])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        """
        Return factored and whether to use first moment (beta1).
        """
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        """
        Root mean square of a tensor.
        """
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col, output):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1).unsqueeze(-1))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()  # NOTE: cast to FP32
                if grad.is_sparse:
                    raise RuntimeError('Adafactor does not support sparse gradients.')

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).type_as(
                            grad
                        )
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).type_as(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0
                else:
                    if use_first_moment:
                        state['exp_avg'] = state['exp_avg'].type_as(grad)
                    if factored:
                        state['exp_avg_sq_row'] = state['exp_avg_sq_row'].type_as(grad)
                        state['exp_avg_sq_col'] = state['exp_avg_sq_col'].type_as(grad)
                    else:
                        state['exp_avg_sq'] = state['exp_avg_sq'].type_as(grad)

                p_data_fp32 = p.data.float()  # NOTE: cast to FP32

                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                group['lr'] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = (grad ** 2) + group['eps'][0]
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-1))
                    exp_avg_sq_col.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-2))

                    # Approximation of exponential moving average of square of gradient
                    self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(1.0 - beta2t, update)
                    torch.rsqrt(exp_avg_sq, out=update).mul_(grad)

                update.div_(max(1.0, self._rms(update) / group['clip_threshold']))
                update.mul_(group['lr'])

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(1 - group['beta1'], update)
                    update = exp_avg

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                p_data_fp32.add_(-update)

                p.data.copy_(p_data_fp32)

        return loss
