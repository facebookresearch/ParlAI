#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Code for LR Schedulers.
See ParlAILRScheduler (super class) and subclasses for detailed documentation
"""

from torch import optim


class ParlAILRScheduler(object):
    """ Class for LR Schedulers
    Subclasses should implement step() - train step - and valid_step()
    Schedulers can be initialized with lr_scheduler_factory()
    """
    def __init__(self, *args, **kwargs):
        if self.fp16:
            # lr schedulers don't work with apex, they expect the "real" optimizer
            optimizer = optimizer.optimizer

        warmup_updates = opt.get('warmup_updates', -1)
        updates_so_far = states.get('number_training_updates', 0)
        if warmup_updates > 0 and (updates_so_far < warmup_updates or hard_reset):
            self.warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.__class__._warmup_lr)
        else:
            self.warmup_scheduler = None

    def _is_lr_warming_up(self):
        """Check if we're warming up the learning rate."""
        return (
            self.warmup_scheduler is not None
            and self._number_training_updates <= self.opt['warmup_updates']
        )

    @classmethod
    def _warmup_lr(cls, step):
        start = opt['warmup_rate']
        end = 1.0
        progress = min(1.0, step / opt['warmup_updates'])
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
                opt, optimizer, states, hard_reset, patience, decay
            )
        elif opt.get('lr_scheduler') == 'fixed':
            scheduler = FixedLRScheduler(
                opt, optimizer, states, hard_reset, patience, decay
            )
        elif opt.get('lr_scheduler') == 'invsqrt':
            scheduler = InvSqrtLRScheduler(
                opt, optimizer, states, hard_reset, patience, decay
            )
        elif opt.get('lr_scheduler') == 'cosine':
            scheduler = CosineLRScheduler(
                opt, optimizer, states, hard_reset, patience, decay
            )
        elif opt.get('lr_scheduler') == 'linear':
            scheduler = LinearLRScheduler(
                opt, optimizer, states, hard_reset, patience, decay
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
        Use the number of train steps to decide when to adjust LR schedule.

        Override this method to override the behavior for training schedulers.
        """

        # compute warmup adjustment if needed
        def warmup_step(epoch):
            if not hasattr(self, 'warmup_scheduler'):
                raise RuntimeError('Looks like you forgot to call build_lr_scheduler')
            if self._is_lr_warming_up():
                self.warmup_scheduler.step(epoch=epoch)

        if self.opt.get('warmup_updates', -1) > 0:
            self.scheduler.warmup_step(epoch=self._number_training_updates)

        step(num_steps)

    @abc.abstractmethod
    def valid_step(self, num_steps):
        """
        Use the number of train steps to decide when to adjust LR schedule.

        Override this method to override the behavior for training schedulers.
        """
        pass

    @abc.abstractmethod
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=decay, patience=patience, verbose=True
        )

    def step(self, num_steps):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, patience, gamma=decay)

    def step(self, num_steps):
        pass

    def valid_step(self, metrics_dict):
        if self._is_lr_warming_up():
            # we're not done warming up, so don't start using validation
            # metrics to adjust schedule
            return
        self.scheduler.step()


class InvSqrtLRScheduler(ParlAILRScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if opt.get('warmup_updates', -1) <= 0:
            raise ValueError('--lr-scheduler invsqrt requires setting --warmup-updates')
        warmup_updates = opt['warmup_updates']
        decay_factor = np.sqrt(max(1, warmup_updates))
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.__class__._invsqrt_lr)

    @classmethod
    def _invsqrt_lr(cls, step):
        return decay_factor / np.sqrt(max(1, step))

    def step(self, num_steps):
        self.scheduler.step()

    def valid_step(self, metrics_dict):
        # this is a training step lr scheduler, nothing to adjust in validation
        pass


class CosineLRScheduler(ParlAILRScheduler):
    """ Scheduler that decays by a cosine function.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = opt['max_steps']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, opt['max_steps']
        )

    def step(self, num_steps):
        if num_steps >= self.max_steps:
            raise StopTrainException('End of Cosine LR Schedule')
        self.scheduler.step()

    def valid_step(self, metrics_dict):
        pass

#TODO @margaretli
# class LinearLRScheduler(ParlAILRScheduler):
    # """ Scheduler that decays linearly.
    # """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # self.max_updates =
#         def _linear_lr(step):
#             #this multiplicative factor ensures linear decay rate
#             lr_mult = (self.max_updates - step - 1) / (self.max_updates - step)
#             return lr_mult
#         self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, _linear_lr)
#
#
    # def step(self, num_steps):
    #     pass
    #
    # def valid_step(self):
#         if self._is_lr_warming_up():
#             # we're not done warming up, so don't start using validation
#             # metrics to adjust schedule
#             return
#         self.scheduler.step()
