#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import json
import os
from enum import Enum
from typing import List, Optional, Tuple

from parlai.core.message import Message
from parlai.core.metrics import F1Metric
from parlai.core.mutators import EpisodeMutator, MessageMutator, register_mutator
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import DialogTeacher
from parlai.tasks.cmu_dog.build import build
from parlai.tasks.wizard_of_wikipedia.agents import (
    RareWordF1Calculator,
    TOKEN_KNOWLEDGE,
    TOKEN_END_KNOWLEDGE,
)
from parlai.utils.data import DatatypeHelper
from parlai.utils.io import PathManager
from parlai.utils.logging import logger
from parlai.utils.typing import TShared


SILENCE = '__SILENCE__'


class SplitType(Enum):
    ORIGINAL = "original"
    ORIGINAL_DEDUPED = "deduped"
    SEEN = "seen"
    UNSEEN = "unseen"


def _datapath(opt: Opt) -> str:
    build(opt)
    return os.path.join(opt['datapath'], 'cmu_dog')


def _datafile(split: str, split_type: SplitType) -> str:
    """
    Returns the filename, e.g. train.json.
    """
    if split_type == SplitType.ORIGINAL:
        return f"{split}.json"
    if split_type == SplitType.SEEN:
        if 'test' in split:
            return "test_seen_split_seen_unseen.json"
        return f"{split}_split_seen_unseen.json"
    if split_type == SplitType.UNSEEN:
        if "test" not in split:
            logger.warning(
                "Trying to use a non-test dataset with split `unseen`. `unseen` "
                "only returns the unseen test set. Are you sure you didn't mean to "
                "use `seen` here?"
            )
        return "test_unseen_split_seen_unseen.json"
    return f"{split}_deduped.json"


def _all_split_datafiles(opt: Opt) -> List[str]:
    datafiles = []
    split_type = SplitType(opt.get("cmu_dog_split_type"))
    if split_type in {SplitType.SEEN, SplitType.UNSEEN}:
        # For seen/unseen split, the full set of dialogs is split
        # across train, valid, test seen, and test unseen
        for split in ['train', 'valid', 'test']:
            datafiles.append(_datafile(split, SplitType.SEEN))
        datafiles.append(_datafile('test', SplitType.UNSEEN))
    else:
        for split in ['train', 'valid', 'test']:
            datafiles.append(_datafile(split, split_type))
    return datafiles


def _collapse_multi_msgs(history, multi_msg_delim):
    """
    This dataset allows for a single user to send multiple messages in a row.

    Here we use a delimiter to represent this, like: "Hey!|Nice to meet you."
    """
    collapsed = []
    last_msg = history[0]
    for msg in history[1:]:
        if last_msg["uid"] == msg["uid"]:
            last_msg["text"] = multi_msg_delim.join((last_msg["text"], msg["text"]))
        else:
            collapsed.append(last_msg)
            last_msg = msg
    # don't forget to add back the last message!
    collapsed.append(last_msg)
    return collapsed


def _article_section_to_text(
    section, fact_delimiter: str, knowledge_keys: List[str] = None
) -> str:
    """
    Example input:
    {
      "cast": [
        "Ben Affleck as Batman",
        "Henry Cavill as Superman",
      ],
      "director": "Zack Snyder"
    }
    Example output:
    "cast:Ben Affleck as Batman,Henry Cavill as Superman;director:Zack Snyder"
    """
    if not section:
        return section
    if isinstance(section, str):
        return section
    texts = []
    for k, v in section.items():
        if knowledge_keys and k not in knowledge_keys:
            continue
        fact = f"{k}:"
        if isinstance(v, str):
            fact += v
        else:
            fact += ",".join(v)
        texts.append(fact)
    return fact_delimiter.join(texts)


def _build_rare_word_f1(opt: Opt) -> RareWordF1Calculator:
    datapath = _datapath(opt)

    def _collect_convo_text(convo_data):
        convo_texts = []
        for conv in convo_data.values():
            # get all messages
            convo_texts.append(' '.join([m['text'] for m in conv['history']]))
        return convo_texts

    # use conversation data from all splits for consistency
    convos = []
    convo_files = _all_split_datafiles(opt)

    for fname in convo_files:
        with PathManager.open(os.path.join(datapath, f"conversations/{fname}")) as f:
            data = json.load(f)
        convos += _collect_convo_text(data)

    return RareWordF1Calculator(' '.join(convos), top_p=0.5)


class CMUDocumentGroundedConversationsTeacher(DialogTeacher):
    """
    CMU Document Grounded Conversations Dataset (aka CMU_DoG)

    Paper: https://arxiv.org/pdf/1809.07358.pdf
    Source: https://github.com/festvox/datasets-CMU_DoG
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        opt = copy.deepcopy(opt)
        self.delimiter = opt.get('delimiter', '\n')
        split_type = SplitType(opt.get("cmu_dog_split_type"))
        if split_type == SplitType.ORIGINAL:
            logger.warning(
                "`original` split type contains duplicate conversations across train, "
                "valid, and test. See https://github.com/festvox/datasets-CMU_DoG/issues/2 "
                "for more detail."
            )
        opt['datafile'] = _datafile(
            split=DatatypeHelper.fold(opt['datatype']), split_type=split_type
        )
        super().__init__(opt, shared)
        if shared:
            self.rare_word_f1 = shared['rare_word_f1']
        else:
            self.rare_word_f1 = _build_rare_word_f1(opt)

    @classmethod
    def add_cmdline_args(
        cls, parser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        cmu_dog = parser.add_argument_group("CMU Document Grounded Conversations")
        cmu_dog.add_argument(
            "--cmu-dog-rating",
            type=int,
            default=[1, 2, 3],
            choices=[1, 2, 3],
            nargs="+",
            help='The higher the number, the better quality the conversation. '
            'For each rating, the number of conversations is as follows: 1-1443, 2-2142, 3-527',
        )
        cmu_dog.add_argument(
            "--cmu-dog-only-with-knowledge",
            type=bool,
            default=True,
            help="Optionally train only the sides of the conversation that have access to knowledge.",
        )
        cmu_dog.add_argument(
            "--cmu-dog-multi-msg-delimiter",
            type=str,
            default=" ",
            help="When one agent is to send multiple messages in a row, they will be concatenated with this delimiter.",
        )
        cmu_dog.add_argument(
            "--cmu-dog-fact-delimiter",
            type=str,
            default=";",
            help="When a section of the knowledge contains multiple facts, they will be concatenated with this delimiter.",
        )
        cmu_dog.add_argument(
            "--cmu-dog-include-knowledge-keys",
            type=str,
            default='cast,critical_response,director,genre,introduction,movieName,rating,year',
            help="Comma-separated list of keys into the knowledge to include as the general conversational context",
        )
        cmu_dog.add_argument(
            "--cmu-dog-provide-movie-context",
            type=bool,
            default=True,
            help="Provide movie facts as the general conversational context",
        )
        cmu_dog.add_argument(
            "--cmu-dog-split-type",
            type=SplitType,
            default=SplitType.ORIGINAL_DEDUPED,
            choices=list(SplitType),
            help=(
                "`orginal`: train/valid/test split from the original release, "
                "`original_deduped`: duplicate conversations removed from train set, "
                "`seen`: refers to movies and is relative to training - `test seen` is conversations about movies that appear during training, "
                "`unseen`: contains conversations about movies that weren't seen in `train`/`valid`/`test seen`. "
                "When using seen/unseen, use `seen` for `train`/`valid`/`seen test` and `unseen` only for `unseen test`."
            ),
        )
        return parser

    def share(self):
        shared = super().share()
        shared['rare_word_f1'] = self.rare_word_f1
        return shared

    def setup_data(self, datafile: str):
        datapath = _datapath(self.opt)
        with PathManager.open(os.path.join(datapath, f"conversations/{datafile}")) as f:
            data = json.load(f)
        with PathManager.open(os.path.join(datapath, "wiki_data.json")) as f:
            wiki_data = json.load(f)

        # Filter by rating
        data = {
            k: c for k, c in data.items() if c["rating"] in self.opt["cmu_dog_rating"]
        }

        def _can_see_info(turn, convo):
            # Sometimes only one participant has access to the article
            return turn["uid"] in convo["whoSawDoc"]

        num_eps = len(data)
        data = list(data.items())
        # loop through conversations
        for i in range(len(data) * 2):
            conv_idx = i % num_eps
            start_idx = i // num_eps

            _conv_id, conv_data = data[conv_idx]

            dialog = _collapse_multi_msgs(
                conv_data["history"], self.opt['cmu_dog_multi_msg_delimiter']
            )
            movie_article = wiki_data[str(conv_data["wikiDocumentIdx"])]

            if self.opt["cmu_dog_only_with_knowledge"] and not _can_see_info(
                dialog[start_idx], conv_data
            ):
                continue

            # loop through turns
            for idx in range(start_idx, len(dialog), 2):
                label_turn = dialog[idx]
                label = label_turn["text"].strip()

                # The section displayed changes across the conversation
                doc_idx = str(label_turn["docIdx"])
                gold_knowledge = _article_section_to_text(
                    movie_article[doc_idx], self.opt['cmu_dog_fact_delimiter']
                )
                section = (
                    movie_article[doc_idx]
                    if _can_see_info(label_turn, conv_data)
                    else None
                )
                section_text = _article_section_to_text(
                    section,
                    self.opt['cmu_dog_fact_delimiter'],
                    self.opt.get('cmu_dog_include_knowledge_keys').split(','),
                )

                # By default, start conversation with silence
                if idx == start_idx:
                    context = (
                        section_text
                        if self.opt['cmu_dog_provide_movie_context']
                        else SILENCE
                    )
                else:
                    context = dialog[idx - 1]["text"].strip()

                yield Message(
                    {
                        'text': context,
                        'labels': [label],
                        'available_knowledge_raw': section,
                        'available_knowledge_text': section_text,
                        'title': movie_article['0']['movieName'],
                        'checked_sentence': gold_knowledge,
                    }
                ), idx == start_idx

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ):
        if 'text' in model_response and 'checked_sentence' in teacher_action:
            self.metrics.add(
                'knowledge_f1',
                F1Metric.compute(
                    model_response['text'], [teacher_action['checked_sentence']]
                ),
            )
        if 'text' in model_response and labels:
            self.metrics.add(
                'rare_word_f1',
                self.rare_word_f1.compute(model_response['text'], labels),
            )


@register_mutator("prepend_knowledge_to_message")
class PrependKnowledgeToMessageMutator(MessageMutator):
    def message_mutation(self, message: Message) -> Message:
        if not message.get('available_knowledge_text'):
            return message
        context = message.pop('text')
        knowledge = f'{TOKEN_KNOWLEDGE} {message["available_knowledge_text"]} {TOKEN_END_KNOWLEDGE}'
        delimiter = self.opt.get('delimiter', '\n')
        message['text'] = (
            knowledge if context == SILENCE else f'{knowledge}{delimiter}{context}'
        )
        return message


@register_mutator("knowledge_only_when_updated")
class KnowledgeWhenUpdatedMutator(EpisodeMutator):
    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        last_knowledge = None
        for msg in episode:
            knowledge = msg.pop('available_knowledge_text')
            if last_knowledge != knowledge:
                msg['available_knowledge_text'] = knowledge
            last_knowledge = knowledge
        return episode


class DefaultTeacher(CMUDocumentGroundedConversationsTeacher):
    pass
