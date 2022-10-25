#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from typing import Optional

from parlai.tasks.reasoning.base import (
    AbstractReasoningTeacher,
    AbstractReason,
    AbstractQuestionAnswer,
)

from parlai.tasks.reasoning.question_answer import (
    SimpleQuestionAnswer,
    PremiseHypothesisQuestionAnswer,
)

from parlai.tasks.reasoning.reason_types.step_by_step import (
    StepByStepReason,
    MWPStepsReason,
)
from parlai.tasks.reasoning.reason_types.free_form import FreeFormReason


class StepByStepReasoningTeacher(AbstractReasoningTeacher):
    """
    Downstream Teachers should implement:

    * get_data_for_fold(self, fold) which yields dicts that contain at least:
        * "steps" (to work with StepByStepReason)
        * "question" (to work with SimpleQuestionAnswer)
        * "answer" (to work with SimpleQuestionAnswer)
    """

    @classmethod
    def get_reason_class(self) -> AbstractReason:
        return StepByStepReason

    @classmethod
    def get_question_answer_class(self) -> AbstractQuestionAnswer:
        return SimpleQuestionAnswer


class MWPStepsReasoningTeacher(AbstractReasoningTeacher):
    """
    Downstream Teachers should implement:

    * get_data_for_fold(self, fold) which yields dicts that contain at least:
        * "steps" (to work with MWPStepsReason)
        * "question" (to work with SimpleQuestionAnswer)
        * "answer" (to work with SimpleQuestionAnswer)
    """

    @classmethod
    def get_reason_class(self) -> AbstractReason:
        return MWPStepsReason

    @classmethod
    def get_question_answer_class(self) -> AbstractQuestionAnswer:
        return SimpleQuestionAnswer


class NliTeacher(AbstractReasoningTeacher):
    """
    Downstream Teachers should implement:

    * get_data_for_fold(self, fold) which yields dicts that contain at least:
        * "reason" (to work with FreeFormReason)
        * "premise" (to work with PremiseHypothesisQuestionAnswer)
        * "hypotehsis" (to work with PremiseHypothesisQuestionAnswer)
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.set_defaults(
            task_prompt="Given the premise, is the hypothesis an entailment, a contradiction, or neutral?"
        )
        return parser

    @classmethod
    def get_reason_class(self) -> AbstractReason:
        return FreeFormReason

    @classmethod
    def get_question_answer_class(self) -> AbstractQuestionAnswer:
        return PremiseHypothesisQuestionAnswer
