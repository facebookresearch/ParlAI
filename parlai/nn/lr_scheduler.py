#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import optim


def init_lr_scheduler(opt, optimizer, states, hard_reset=False):
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


class ParlAILRScheduler(object):
    def __init__(self, *args, **kwargs):
        if self.fp16:
            # lr schedulers don't work with apex, they expect the "real" optimizer
            optimizer = optimizer.optimizer

        warmup_updates = opt.get('warmup_updates', -1)
        updates_so_far = states.get('number_training_updates', 0)
        if warmup_updates > 0 and (updates_so_far < warmup_updates or hard_reset):

            def _warmup_lr(step):
                start = opt['warmup_rate']
                end = 1.0
                progress = min(1.0, step / opt['warmup_updates'])
                lr_mult = start + (end - start) * progress
                return lr_mult

            self.warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, _warmup_lr)
        else:
            self.warmup_scheduler = None

    def load_state(self, states):
        if 'number_training_updates' in states:
            self._number_training_updates = states['number_training_updates']
        if self.scheduler and 'lr_scheduler' in states:
            self.scheduler.load_state_dict(states['lr_scheduler'])
        if states.get('warmup_scheduler') and getattr(self, 'warmup_scheduler', None):
            self.warmup_scheduler.load_state_dict(states['warmup_scheduler'])

    def _is_lr_warming_up(self):
        """Check if we're warming up the learning rate."""
        return (
            self.warmup_scheduler is not None
            and self._number_training_updates <= self.opt['warmup_updates']
        )

    def step():
        pass

    def valid_step(self, metrics_dict):
        """
        Use the metrics to decide when to adjust LR schedule.

        This uses the loss as the validation metric if present, if not this
        function does nothing. Note that the model must be reporting loss for
        this to work.

        Override this to override the behavior.
            """
        pass


class ReduceOnPlateauLRScheduler(ParlAILRScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=decay, patience=patience, verbose=True
        )

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

        def _invsqrt_lr(step):
            return decay_factor / np.sqrt(max(1, step))

    def step(self, epochs):
        self.scheduler.step()

    def valid_step(self, metrics_dict):
        # this is a training step lr scheduler, nothing to adjust in validation
        pass


class CosineLRScheduler(ParlAILRScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, opt['max_num_epochs'] #TODO @margaretli find exact name
            # also investigate whether to use stoptrain instead of relying on max num epochs
        )

    def valid_step(self, metrics_dict):
        if self._is_lr_warming_up():
            # we're not done warming up, so don't start using validation
            # metrics to adjust schedule
            return
        self.scheduler.step()

#TODO @margaretli
# class LinearLRScheduler(ParlAILRScheduler):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # self.max_updates =
#         def _linear_lr(step):
#             #this multiplicative factor ensures linear decay rate
#             lr_mult = (self.max_updates - step - 1) / (self.max_updates - step)
#             return lr_mult
#         self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, _linear_lr)
#
#     def valid_step(self):
#         if self._is_lr_warming_up():
#             # we're not done warming up, so don't start using validation
#             # metrics to adjust schedule
#             return
#         self.scheduler.step()
