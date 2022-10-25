#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from typing import Dict, Optional, Tuple

from parlai.tasks.reasoning.base import (
    t_QUESTION,
    t_ANSWER_PREFIX_TOKEN,
    t_ANSWER,
    AbstractQuestionAnswer,
)


##############
# Question + Answer child classes
#############
class PremiseHypothesisQuestionAnswer(AbstractQuestionAnswer):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("PremiseHypothesisQuestionAnswer args")
        group.add_argument(
            "--premise-token",
            default="PREMISE: ",
            type=str,
        )
        group.add_argument(
            "--hypothesis-token",
            default="HYPOTHESIS: ",
            type=str,
        )
        return parser

    def get_question_answer(
        self, example_dict: Dict
    ) -> Tuple[t_QUESTION, t_ANSWER_PREFIX_TOKEN, t_ANSWER, Dict]:
        premise_token = self.opt.get("premise_token", "PREMISE: ")
        premise = example_dict["premise"]
        hypothesis_token = self.opt.get("hypothesis_token", "HYPOTHESIS: ")
        hypothesis = example_dict["hypothesis"]
        question = f"{premise_token}{premise}{self.inner_separator}{hypothesis_token}{hypothesis}"
        answer = example_dict["answer"]
        qa_dict = {
            "premise": premise,
            "hypothesis": hypothesis,
            "answer": answer,
        }
        return question, self.answer_token, answer, qa_dict


class SimpleQuestionAnswer(AbstractQuestionAnswer):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("SimpleQuestionAnswer args")
        group.add_argument(
            "--question-token",
            default="QUESTION: ",
            type=str,
        )
        return parser

    def get_question_answer(
        self, example_dict: Dict
    ) -> Tuple[t_QUESTION, t_ANSWER_PREFIX_TOKEN, t_ANSWER, Dict]:
        question_token = self.opt.get("question_token", "QUESTION: ")
        question = example_dict["question"]
        question_text = f"{question_token}{question}"
        answer = example_dict["answer"]
        qa_dict = {
            "question": question,
            "answer": answer,
        }
        return question_text, self.answer_token, answer, qa_dict


class MultipleChoiceQuestionAnswer(AbstractQuestionAnswer):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("MultipleChoiceQuestionAnswer args")
        group.add_argument(
            "--question-token",
            default="QUESTION: ",
            type=str,
        )
        group.add_argument(
            "--choices-token",
            default="ANSWER_CHOICES: ",
            type=str,
        )
        group.add_argument(
            "--choices-prefix",
            default=["(a)", "(b)", "(c)", "(d)", "(e)"],
            nargs="+",
            type=str,
        )
        return parser

    def get_question_answer(
        self, example_dict: Dict
    ) -> Tuple[t_QUESTION, t_ANSWER_PREFIX_TOKEN, t_ANSWER, Dict]:
        question_token = self.opt.get("question_token", "QUESTION: ")
        question = example_dict["question"]

        answer_choices = [
            f"{token} {choice}"
            for choice, token in zip(
                example_dict["answer_choices"], self.opt["choices_prefix"]
            )
        ]
        answer_choices_token = self.opt.get("choices_token", "ANSWER_CHOICES: ")

        question_text = f"{question_token}{question}{self.inner_separator}{answer_choices_token}{' '.join(answer_choices)}"

        answer_idx = example_dict["answer_idx"]
        answer = answer_choices[answer_idx]

        qa_dict = {
            "question": question,
            "answer": answer,
            "answer_idx": answer_idx,
            "choices": answer_choices,
        }
        return question_text, self.answer_token, answer, qa_dict
