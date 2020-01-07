#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Code for LR Schedulers.
See ParlAILRScheduler (super class) and subclasses for detailed documentation
"""

from abc import abstractmethod
from torch import optim
import numpy as np

from parlai.utils.exceptions import StopTrainException


class ParlAILRScheduler(object):
    """ Class for LR Schedulers
    Subclasses must implement abstract methods train_step() and valid_step()
    Schedulers can be initialized with lr_scheduler_factory()
    """

    def __init__(self, optimizer, states, hard_reset, warmup_updates):
        self.warmup_updates = warmup_updates
        updates_so_far = states.get('number_training_updates', 0)
        if self.warmup_updates > 0 and (
            updates_so_far < self.warmup_updates or hard_reset
        ):
            self.warmup_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, self._warmup_lr
            )
        else:
            self.warmup_scheduler = None

    def _is_lr_warming_up(self):
        """Check if we're warming up the learning rate."""
        return (
            self.warmup_scheduler is not None
            and self._number_training_updates <= self.warmup_updates
        )

    def _warmup_lr(self, step):
        start = self.warmup_updates
        end = 1.0
        progress = min(1.0, step / self.warmup_updates)
        lr_mult = start + (end - start) * progress
        return lr_mult

    def load_state(self, states):
        if 'number_training_updates' in states:
            self._number_training_updates = states['number_training_updates']
        if self.scheduler and 'lr_scheduler' in states:
            self.scheduler.load_state_dict(states['lr_scheduler'])
        if states.get('warmup_scheduler') and getattr(self, 'warmup_scheduler', None):
            self.warmup_scheduler.load_state_dict(states['warmup_scheduler'])

    @classmethod
    def lr_scheduler_factory(cls, opt, optimizer, states, hard_reset=False):
        """
        Create the learning rate scheduler, and assign it to self.scheduler.
        This scheduler will be updated upon a call to receive_metrics.
        May also create self.warmup_scheduler, if appropriate.
        :param state_dict states: Possible state_dict provided by model
            checkpoint, for restoring LR state
        :param bool hard_reset: If true, the LR scheduler should ignore the
            state dictionary.
        """

        patience = opt.get('lr_scheduler_patience', 3)
        decay = opt.get('lr_scheduler_decay', 0.5)
        warmup_updates = opt.get('warmup_updates', -1)
        max_lr_steps = opt.get('max_lr_steps', -1)

        if opt.get('lr_scheduler') == 'none':
            scheduler = None
        elif decay == 1.0:
            warn_once(
                "Your LR decay is set to 1.0. Assuming you meant you wanted "
                "to disable learning rate scheduling. Adjust --lr-scheduler-decay "
                "if this is not correct."
            )
            self.scheduler = None
        elif opt.get('lr_scheduler') == 'reduceonplateau':
            scheduler = ReduceOnPlateauLRScheduler(
                optimizer, states, hard_reset, patience, decay, warmup_updates,
            )
        elif opt.get('lr_scheduler') == 'fixed':
            scheduler = FixedLRScheduler(
                optimizer, states, hard_reset, patience, decay, warmup_updates,
            )
        elif opt.get('lr_scheduler') == 'invsqrt':
            scheduler = InvSqrtLRScheduler(
                optimizer,
                states,
                hard_reset,
                patience,
                decay,
                warmup_updates,
                max_lr_steps,
            )
        elif opt.get('lr_scheduler') == 'cosine':
            scheduler = CosineLRScheduler(
                optimizer,
                states,
                hard_reset,
                patience,
                decay,
                warmup_updates,
                max_lr_steps,
            )
        elif opt.get('lr_scheduler') == 'linear':
            scheduler = LinearLRScheduler(
                optimizer,
                states,
                hard_reset,
                patience,
                decay,
                warmup_updates,
                max_lr_steps,
            )
        else:
            raise ValueError(
                "Don't know what to do with lr_scheduler '{}'".format(
                    opt.get('lr_scheduler')
                )
            )

        # time to load LR state from the checkpoint, if possible.
        if (
            # there is already an old LR scheduler saved on disk
            states
            and
            # and the old LR scheduler is different
            states.get('lr_scheduler_type') != opt['lr_scheduler']
            and
            # and we're not already using a fresh scheduler
            not hard_reset
        ):
            # the LR scheduler changed, start things fresh
            warn_once("LR scheduler is different from saved. Starting fresh!")
            hard_reset = True

        if hard_reset:
            # We're not going to use the LR schedule, let's just exit
            return
        else:
            scheduler.load_state(states)
            # do the actual loading (if possible)

        return scheduler

    def step(self, num_steps):
        """
        Use the number of train steps to adjust the warmup scheduler or
        the main scheduler, depending on where in training we are.

        Override this method to override the behavior for training schedulers.
        """
        self._number_training_updates = num_steps
        if hasattr(self, 'warmup_scheduler'):
            if self._is_lr_warming_up():
                self.warmup_scheduler.step(epoch=num_steps)
            else:
                scheduler_steps = num_steps - self.warmup_updates
                self.train_step(scheduler_steps)

    @abstractmethod
    def train_step(self, scheduler_steps):
        """
        Use the number of train steps to decide when to adjust LR schedule.

        Override this method to override the behavior for training schedulers.
        """
        pass

    @abstractmethod
    def valid_step(self, metrics_dict):
        """
        Use the metrics to decide when to adjust LR schedule.

        This uses the loss as the validation metric if present, if not this
        function does nothing. Note that the model must be reporting loss for
        this to work.

        Override this method to override the behavior for validation schedulers.
        """
        pass


class ReduceOnPlateauLRScheduler(ParlAILRScheduler):
    def __init__(self, optimizer, states, hard_reset, patience, decay, warmup_updates):
        super().__init__(optimizer, states, hard_reset, warmup_updates,)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=decay, patience=patience, verbose=True
        )

    def train_step(self, scheduler_steps):
        pass

    def valid_step(self, metrics_dict):
        if self._is_lr_warming_up():
            # we're not done warming up, so don't start using validation
            # metrics to adjust schedule
            return
        if 'loss' not in metrics_dict:
            # nothing to step on, just skip
            warn_once("LR scheduler expected to see loss metric, but didn't.")
            return
        self.scheduler.step(metrics_dict['loss'])


class FixedLRScheduler(ParlAILRScheduler):
    def __init__(self, optimizer, states, hard_reset, patience, decay, warmup_updates):
        super().__init__(optimizer, states, hard_reset, warmup_updates,)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, patience, gamma=decay)

    def train_step(self, scheduler_steps):
        pass

    def valid_step(self, metrics_dict):
        if self._is_lr_warming_up():
            # we're not done warming up, so don't start using validation
            # metrics to adjust schedule
            return
        self.scheduler.step()


class InvSqrtLRScheduler(ParlAILRScheduler):
    def __init__(
        self,
        optimizer,
        states,
        hard_reset,
        patience,
        decay,
        warmup_updates,
        max_lr_steps,
    ):
        super().__init__(optimizer, states, hard_reset, warmup_updates)
        if max_lr_steps <= 0:
            raise ValueError('--lr-scheduler invsqrt requires setting --max_lr_steps')
        self.max_lr_steps = max_lr_steps
        self.decay_factor = np.sqrt(max(1, max_lr_steps))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, self._invsqrt_lr
        )

    def _invsqrt_lr(self, step):
        return self.decay_factor / np.sqrt(max(1, step))

    def train_step(self, scheduler_steps):
        self.scheduler.step(epoch=scheduler_steps)

    def valid_step(self, metrics_dict):
        # this is a training step lr scheduler, nothing to adjust in validation
        pass


class CosineLRScheduler(ParlAILRScheduler):
    """ Scheduler that decays by a cosine function.
    """

    def __init__(
        self,
        optimizer,
        states,
        hard_reset,
        patience,
        decay,
        warmup_updates,
        max_lr_steps,
    ):
        super().__init__(optimizer, states, hard_reset, warmup_updates,)
        if max_lr_steps <= 0:
            raise ValueError('--lr-scheduler cosine requires setting --max_lr_steps')
        self.max_lr_steps = max_lr_steps
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_lr_steps)

    def train_step(self, scheduler_steps):
        if scheduler_steps >= self.max_lr_steps:
            raise StopTrainException('End of Cosine LR Schedule')
        self.scheduler.step(epoch=scheduler_steps)

    def valid_step(self, metrics_dict):
        pass


class LinearLRScheduler(ParlAILRScheduler):
    """ Scheduler that decays linearly.
    """
    def __init__(
        self,
        optimizer,
        states,
        hard_reset,
        patience,
        decay,
        warmup_updates,
        max_lr_steps,
    ):
        super().__init__(optimizer, states, hard_reset, warmup_updates,)
        if max_lr_steps <= 0:
            raise ValueError('--lr-scheduler linear requires setting --max_lr_steps')
        self.max_lr_steps = max_lr_steps
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, self._linear_lr)

    def _linear_lr(self, step):
        # this multiplicative factor ensures linear decay rate
        lr_mult = (self.max_lr_steps - step - 1) / (self.max_lr_steps - step)
        return lr_mult

    def train_step(self, scheduler_steps):
        if scheduler_steps >= self.max_lr_steps:
            raise StopTrainException('End of Linear LR Schedule')
        self.scheduler.step(epoch=scheduler_steps)

    def valid_step(self):
        pass
