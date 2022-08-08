#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from holistic_bias.run_bias_calculation import (
    EvalModelOnHolisticBias,
    HolisticBiasTeacher,
)
from holistic_bias.src.bias_measurements import (
    BiasMeasurementCompiler,
)

from parlai.core.teachers import register_teacher


HOLISTIC_BIAS_OPT_CONTEXT_TASK = 'holistic_bias_opt_context'


@register_teacher(HOLISTIC_BIAS_OPT_CONTEXT_TASK)
class HolisticBiasOptContextTeacher(HolisticBiasTeacher):
    """
    A version of HolisticBiasTeacher that sets a context string that OPT-based models
    will be familiar with.
    """

    def __init__(self, opt, shared=None):
        if opt['use_blenderbot_context'] is True:
            raise ValueError(
                'If you want to use BlenderBot-style context strings, use the base HolisticBiasTeacher.'
            )
        self.id = HOLISTIC_BIAS_OPT_CONTEXT_TASK
        super().__init__(opt, shared)

    def setup_data(self, path):
        """
        Modify each output message to add in an OPT-compatible context string.
        """
        for message, new_episode in super().setup_data(path):
            assert (
                message['text'] == '__SILENCE__'
            ), 'The expected original context string is not found!'
            message['text'] = 'Person 1:'
            yield message, new_episode


class EvalModelOnHolisticBiasOptContext(EvalModelOnHolisticBias):
    @classmethod
    def setup_args(cls):
        parser = super(EvalModelOnHolisticBiasOptContext, cls).setup_args()
        parser.set_params(task=HOLISTIC_BIAS_OPT_CONTEXT_TASK)
        return parser


if __name__ == '__main__':
    EvalModelOnHolisticBiasOptContext.main()
    parser_ = EvalModelOnHolisticBiasOptContext.setup_args()
    opt_ = parser_.parse_args()
    BiasMeasurementCompiler(opt_).compile()
