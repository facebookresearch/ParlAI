#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.core.teachers import DialogTeacher
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional, Tuple

"""
Base classes for dealing with datasets that represent Reasoning

We do this because reasoning datasets oftentimes have the same structure and it's obnoxious
having to deal with some of the nitty logistical details (ex. special tokens) over and over again.

Also makes it easy to have a common sent of synthetic pertubations for reasoning steps.
"""

# Couple typedefs to make it clearer what we're doing
t_QUESTION = str
t_ANSWER_PREFIX_TOKEN = str
t_ANSWER = str

t_REASON_PREFIX_TOKEN = str
t_REASON = str
t_REASON_PROCESSED_RESULT = Tuple[t_REASON_PREFIX_TOKEN, t_REASON, Dict]


class AbstractQuestionAnswer:
    """
    Used inside of an `AbstractReasoningTeacher` to define how the "Question" and
    "Answer" part of an example should be serialized out to a string, given a dict
    object that includes all necessary information for the example.

    The Question includes all context (ex. input background) necessary to generate the
    Answer. This class is responsible for handling its own tokenization.
    """

    def __init__(self, opt: Opt):
        self.opt = opt
        self.inner_separator = opt.get("inner_separator_token", "\t")
        self.answer_token = opt.get("answer_token", "ANSWER: ")

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        group = parser.add_argument_group("Abstract Question Answer args")
        group.add_argument(
            "--inner-separator-token",
            type=str,
            default="\t",
            help="See abstract reasoning teacher help for usage. Including here as well for consistency + testing purposes",
        )
        group.add_argument(
            "--answer-token",
            type=str,
            default="ANSWER: ",
            help="Token in front of the answer. Child classes can set the default to something else if desired",
        )
        return parser

    def get_question_answer(
        self, example_dict: Dict
    ) -> Tuple[t_QUESTION, t_ANSWER_PREFIX_TOKEN, t_ANSWER, Dict]:
        """
        Given a dict object that includes all of the information necessary for an
        example, serialize Question and Answer strings that will be used to construct
        the `text` object in a ParlAI Teacher Message. Called from `setup_data()` in
        `AbstractReasoningTeacher`.

        For example, the 'question text' for a multiple choice version of this class may include different
        presentation of prefixes to different answer choices (ex. a,b,c vs 1,2,3). Depending on external flags,
        the `answer prefix token` may be a straightforward "ANSWER:" or it may be something like "CHOICE:";
        the `answer text` itself may be the answer itself or the prefix, etc.

        Output is:

            str - question text (Including any structured prefix token)
            str - answer prefix token
            str - answer text
            Dict - any metadata to append to the final Message object

        We return the `answer prefix token` as a string out of this function in case we want to prompt the
        message with it as the last line of the input, rather than as the first part of the label.
        """
        raise NotImplementedError("Derived class must implement")


class AbstractReason:
    """
    Used inside of an `AbstractReasoningTeacher` to define how the "Reason" part of an
    example should be serialized out to a string, given a dict object that includes all
    necessary information for the example.

    Downstream classes can also implement `perturbations` on these reasons. (See
    `reason_types/step_by_step.py` for examples of this.)
    """

    def __init__(self, opt: Opt):
        self.opt = opt
        self.inner_separator = opt.get("inner_separator_token", "\t")

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        group = parser.add_argument_group("Abstract Reason args")
        group.add_argument(
            "--inner-separator-token",
            type=str,
            default="\t",
            help="See abstract reasoning teacher help for usage. Including here as well for consistency + testing purposes",
        )
        return parser

    def get_full_reason_text(self, example_dict: Dict) -> t_REASON_PROCESSED_RESULT:
        """
        Given a dict object that includes all of the information necessary for an
        example, serialize Reason strings that will be used to construct the `text`
        object in a ParlAI Teacher Message. Called from `setup_data()` in
        `AbstractReasoningTeacher`.

        Perturbations to reasoning chains should be implemented in this function.

        Output is:

            str - reason prefix token
            str - reason string
            Dict - any metadata to append to the final Message object

        We return the `reason prefix token` as a string out of this class in case we want to
        prompt the message with it as the last line of the input, rather than as the first part of the label.
        """

        raise NotImplementedError("Derived class must implement")

    def batch_get_full_reason_text(
        self, examples: List[Dict]
    ) -> List[t_REASON_PROCESSED_RESULT]:
        """
        Can be overridden to parallelize expensive parts of the get_full_reason_text
        operation.

        Only called when --preload-batch-size is set to a value greater than one.
        """
        return [self.get_full_reason_text(e) for e in examples]


class AbstractReasoningTeacher(DialogTeacher):
    """
    Teacher that builds up Message objects based on child `question and answer` and
    `reason` classes.

    Also handles making exemplars.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group("Abstract Reasoning Teacher args")
        #####
        # Token options
        #####
        group.add_argument(
            "--inner-separator-token",
            type=str,
            default="\t",
            help="Token that we use to separate components within an example. Intentially the same"
            + "as in AbstractReason and AbstractQuestionAnswer",
        )
        group.add_argument(
            "--example-separator-token",
            type=str,
            default="\n",
            help="Token that we use to separate different examples. Set to '\\n' as a default since"
            + "some agents (ex. GPTZ) will use '\\n' as a way to separate turns",
        )
        #####
        # Options for combining/including reasons with questions/answers
        #####
        group.add_argument(
            "--include-reason",
            type=bool,
            default=True,
            help="Should we include the reason as part of the generation/exemplars?",
        )
        group.add_argument(
            "--reason-before-answer",
            type=bool,
            default=True,
            help="When we include a reason, should it come before or after the answer? Useful for"
            + "designating pre-hoc vs post-hoc justification",
        )
        group.add_argument(
            "--initial-generation-target",
            type=str,
            choices=["reason", "answer", "other"],
            default="reason",
            help="If we are not including the reason, this does nothing and goes with assuming `answer`"
            + "is the target since that is the only option. This, in conjunction with `--reason-before-answer`"
            + "determines what is included in the input vs the output; we will construct the input/output"
            + "appropriately based on these two flags if `reason` or `answer` is selected. If 'other' is selected,"
            + "`self.make_custom_input_text_label()` will be called. (Last of these most useful for if we"
            + "want to do say, step-by-step reasoning",
        )
        group.add_argument(
            "--include-generation-target-token-in-input",
            type=bool,
            default=True,
            help="Sometimes we want to include the token representing the reason or the answer at the end"
            + "of our input, since this can help prime the model to do the correct generation",
        )
        #####
        # Options for task prompts + exemplars
        #####
        group.add_argument(
            "--use-task-prompt",
            type=bool,
            default=True,
            help="Use the prompt from `get_task_prompt()` before all reasons, questions, and answers",
        )
        group.add_argument(
            "--task-prompt",
            type=str,
            default="",
            help="`get_task_prompt()` looks at this by default. Child classes can override this value.",
        )
        group.add_argument(
            "--exemplar-idx",
            type=int,
            nargs="*",
            default=[],
            help="IDX of exemplars to use. Looks data up from a `self.exemplars_raw` field that should be"
            + "set in `__init__()`. NOTE: Including this in the base class as a convenience; individual"
            + "datasets may find it useful to have their own `--<dataset name>-exemplar-idx` arguments for"
            + "multitasking, since this will be shared across tasks",
        )
        #####
        # Data loading
        #####
        group.add_argument(
            "--preload-batch-size",
            type=int,
            default=1,
            help=(
                "If greater than 1, data will be preloaded in batches of this size. This is useful when "
                "data loading involves an expensive operation, e.g. paraphrase perturbation."
            ),
        )
        group.add_argument(
            "--skip-perturbation-failures",
            type=bool,
            default=False,
            help=(
                "Some chains can not be perturbes for certain perturbation types. Set this flag to return "
                "only perturbed chains and drop non-perturbed examples"
            ),
        )

        parser = cls.get_reason_class().add_cmdline_args(parser, partial_opt)
        parser = cls.get_question_answer_class().add_cmdline_args(parser, partial_opt)

        return parser

    @classmethod
    def get_reason_class(self) -> AbstractReason:
        raise NotImplementedError("Derived class must implement")

    @classmethod
    def get_question_answer_class(self) -> AbstractQuestionAnswer:
        raise NotImplementedError("Derived class must implement")

    def get_task_prompt(self) -> str:
        """
        To keep flexibility, do not override this function.
        """
        prompt = self.opt.get("task_prompt", "")
        if prompt == "":
            raise RuntimeError(
                "Got an empty string for the prompt - is this intentional? If not, set with "
                "`parser.set_defaults(task_prompt='<INSERT PROMPT>')` in `add_cmdline_args`."
            )
        return prompt

    def __init__(self, opt, shared=None):
        self.reason_instance = self.get_reason_class()(opt)
        self.question_answer_instance = self.get_question_answer_class()(opt)
        super().__init__(opt, shared)
        self.paraphrase_cache = {}

    def get_data_for_fold(self, fold):
        raise NotImplementedError("Derived class must implement")

    def make_custom_input_text_label(self):
        raise NotImplementedError("Derived class to (optionally) implement")

    def get_exemplar(self, idx):
        return self.exemplars_raw[idx]

    def setup_data(self, fold) -> Iterable[Tuple[Message, bool]]:
        if self.opt.get('preload_batch_size', 1) > 1:
            if not hasattr(self, '_preloaded_data'):
                self._preloaded_data = self._preload_data(fold)
            data_iterator = self._preloaded_data
        else:
            data_iterator = self._data_generator(fold)
        for message in data_iterator:
            yield message, True

    def _data_generator(self, fold) -> Iterable[Message]:
        example_separator = self.opt.get("example_separator_token")
        inner_separator = self.opt.get("inner_separator_token")
        skip_perturbation_failures = self.opt.get("skip_perturbation_failures")

        prompt_and_examplars = self._build_prompt_and_exemplars(
            inner_separator,
            example_separator,
        )

        #### Set up individual examples
        for example in self.get_data_for_fold(fold):
            reason_tuple = None
            if self.opt.get("include_reason", True):
                try:
                    reason_tuple = self.reason_instance.get_full_reason_text(example)
                except (RuntimeError, IndexError) as e:
                    if skip_perturbation_failures:
                        print(f"WARNING: {e}")
                        continue
                    else:
                        raise e
            message = self._message_from_example(
                example=example,
                prompt_and_examplars=prompt_and_examplars,
                inner_separator=inner_separator,
                reason_tuple=reason_tuple,
            )
            yield message

    def _preload_data(self, fold) -> List[Message]:
        example_separator = self.opt.get("example_separator_token")
        inner_separator = self.opt.get("inner_separator_token")

        prompt_and_examplars = self._build_prompt_and_exemplars(
            inner_separator,
            example_separator,
        )

        all_messages = []
        try:
            print("Pre-loading data...")
            batch = []
            for example in tqdm(list(self.get_data_for_fold(fold))):
                batch.append(example)
                if len(batch) >= self.opt['preload_batch_size']:
                    all_messages.extend(
                        self._load_data_batch(
                            batch,
                            prompt_and_examplars,
                            inner_separator,
                        )
                    )
                    batch = []
            if len(batch) > 0:
                all_messages.extend(
                    self._load_data_batch(
                        batch,
                        prompt_and_examplars,
                        inner_separator,
                    )
                )
        finally:
            self.reason_instance.save_cache()

        return all_messages

    def _load_data_batch(
        self,
        example_batch: List[Dict],
        prompt_and_examplars: str,
        inner_separator: str,
    ) -> List[Message]:
        full_reason_texts = [None] * len(example_batch)
        if self.opt.get("include_reason", True):
            full_reason_texts = self.reason_instance.batch_get_full_reason_text(
                example_batch
            )
        message_list = [
            self._message_from_example(
                example=example,
                prompt_and_examplars=prompt_and_examplars,
                inner_separator=inner_separator,
                reason_tuple=full_reason,
            )
            for example, full_reason in zip(example_batch, full_reason_texts)
        ]
        return message_list

    def _build_prompt_and_exemplars(
        self, inner_separator: str, example_separator: str
    ) -> str:
        #### Set up our prompt and any exemplars we would like to use for the whole dataset
        prompt_and_examplars_raw = []
        if self.opt.get("use_task_prompt", True):
            prompt_and_examplars_raw.append(self.get_task_prompt())
        for idx in self.opt.get("exemplar_idx", []):
            (
                question,
                answer_prefix,
                answer,
                _,
            ) = self.question_answer_instance.get_question_answer(
                self.get_exemplar(idx)
            )
            exemplar = f"{question}{inner_separator}"
            if self.opt.get("include_reason", True):
                reason_prefix, reason, _ = self.reason_instance.get_full_reason_text(
                    self.get_exemplar(idx)
                )
                if self.opt.get("reason_before_answer", True):
                    exemplar += f"{reason_prefix}{reason}{inner_separator}{answer_prefix}{answer}"
                else:
                    exemplar += f"{answer_prefix}{answer}{inner_separator}{reason_prefix}{reason}"
            else:
                exemplar += f"{answer_prefix}{answer}"
            prompt_and_examplars_raw.append(exemplar)
        prompt_and_examplars = example_separator.join(prompt_and_examplars_raw)
        if len(prompt_and_examplars) > 0:
            prompt_and_examplars += example_separator
        return prompt_and_examplars

    def _message_from_example(
        self,
        example: Dict[str, Any],
        prompt_and_examplars: str,
        inner_separator: str,
        reason_tuple: Optional[t_REASON_PROCESSED_RESULT] = None,
    ) -> Message:
        (
            question,
            answer_prefix,
            answer,
            qa_dict,
        ) = self.question_answer_instance.get_question_answer(example)

        message_dict = qa_dict

        if self.opt.get("include_reason", True):
            assert reason_tuple is not None
            (
                reason_prefix,
                reason,
                reason_dict,
            ) = reason_tuple

            message_dict.update(reason_dict)

            if self.opt.get("initial_generation_target", "reason") == "other":
                (
                    example_text,
                    label_prefix,
                    label,
                ) = self.make_custom_input_text_label(
                    question,
                    answer_prefix,
                    answer,
                    reason_prefix,
                    reason,
                    message_dict,
                )
            else:
                example_text = f"{question}{inner_separator}"
                label_prefix = ""
                label = ""
                if self.opt.get("reason_before_answer", True):
                    if self.opt.get("initial_generation_target", "reason") == "reason":
                        label_prefix = reason_prefix
                        label = f"{reason}{inner_separator}{answer_prefix}{answer}"
                    else:
                        example_text += f"{reason_prefix}{reason}{inner_separator}"
                        label_prefix = answer_prefix
                        label = answer
                else:  # answer before reason
                    if self.opt.get("initial_generation_target", "reason") == "answer":
                        label_prefix = answer_prefix
                        label = f"{answer}{inner_separator}{reason_prefix}{reason}"
                    else:
                        example_text += f"{answer_prefix}{answer}{inner_separator}"
                        label_prefix = reason_prefix
                        label = reason

        else:  # case without reason is simple, just get the answer
            example_text = f"{question}{inner_separator}"
            label_prefix = answer_prefix
            label = answer

        if self.opt.get("include_generation_target_token_in_input", True):
            example_text += label_prefix
        else:
            label = label_prefix + label

        message_dict["text"] = prompt_and_examplars + example_text
        message_dict["label"] = label

        return Message(message_dict)
