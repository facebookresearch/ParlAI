#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from holistic_bias.run_bias_calculation import (
    EvalModelOnHolisticBias,
)
from holistic_bias.src.bias_measurements import (
    BiasMeasurementCompiler,
)

from projects.seeker.tasks import mutators

_ = mutators
# Register the mutator


if __name__ == '__main__':
    EvalModelOnHolisticBias.main()
    parser_ = EvalModelOnHolisticBias.setup_args()
    opt_ = parser_.parse_args()
    BiasMeasurementCompiler(opt_).compile()
