#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
import copy
from typing import List, Optional, Union

from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import MultiTaskTeacher

#########
# Mixin #
#########


class BB3TeacherMixin(Agent):
    """
    This mixin is to ensure that messages from the teachers contain the proper fields.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if hasattr(self, 'tasks'):
            for t in self.tasks:
                t.id = self._id

    def get(self, episode_idx, entry_idx=None):
        example = super().get(episode_idx, entry_idx)
        if isinstance(example, Message) and 'id' in example:
            example.force_set('id', self._id)
        else:
            example['id'] = self._id
        return example

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, str):
        self._id = type(self).__name__


##############
# Multitasks #
##############


class BB3MultitaskTeacher(MultiTaskTeacher, ABC):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        MultiTaskTeacher.add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            '--blenderbot3-base-agent',
            type=str,
            default='r2c2',
            choices=['r2c2', 'opt'],
            help='Base agent for which to format tasks.',
        )
        return parser

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        tasks = [
            f"projects.bb3.tasks.{opt['blenderbot3_base_agent']}_{self.get_task_type()}_tasks:{teacher}"
            for teacher in self.get_teachers()
        ]
        opt['task'] = ','.join(tasks)
        opt['multitask_weights'] = self.get_multitask_weights()
        super().__init__(opt, shared)

    @abstractmethod
    def get_task_type(self) -> str:
        """
        Return the task type.
        """

    @abstractmethod
    def get_teachers(self) -> List[str]:
        """
        Get the teachers for multitask.
        """

    @abstractmethod
    def get_multitask_weights(self) -> Union[List[int], str]:
        """
        Get the weights for multitask.
        """


class AlwaysSearchTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'NQOpenSearchDecisionTeacher',
            'SquadSearchDecisionTeacher',
            'TriviaQASearchDecisionTeacher',
        ]

    def get_task_type(self) -> str:
        return 'decision'

    def get_multitask_weights(self) -> Union[List[int], str]:
        return [1] * len(self.get_teachers())


class MaybeSearchTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'WowSearchDecisionTeacher',
            'WoiSearchDecisionTeacher',
            'Convai2SearchDecisionTeacher',
            'EDSearchDecisionTeacher',
            'MSCSearchDecisionTeacher',
        ]

    def get_multitask_weights(self) -> Union[List[int], str]:
        return [1] * len(self.get_teachers())

    def get_task_type(self) -> str:
        return 'decision'


class MemoryDecisionTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'Convai2MemoryDecisionTeacher',
            'MSCMemoryDecisionTeacher',
            'BSTMemoryDecisionTeacher',
            'EDMemoryDecisionTeacher',
        ]

    def get_multitask_weights(self) -> Union[List[int], str]:
        return [3, 3, 1, 1]

    def get_task_type(self) -> str:
        return 'decision'


class SearchQueryGenerationTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return ['WoiSearchQueryTeacher', 'FitsSearchQueryTeacher']

    def get_multitask_weights(self) -> Union[List[int], str]:
        return [1, 1]

    def get_task_type(self) -> str:
        return 'search_generation'


class MemoryGenerationTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return ['MSCMemoryGeneratorTeacher']

    def get_multitask_weights(self) -> Union[List[int], str]:
        return [1]

    def get_task_type(self) -> str:
        return 'memory_generation'


class MemoryKnowledgeGenerationTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'BSTMemoryKnowledgePersOverlapTeacher',
            'BSTMemoryKnowledgeUttOverlapTeacher',
            'Convai2MemoryKnowledgePersOverlapTeacher',
            'Convai2MemoryKnowledgeUttOverlapTeacher',
            'EDMemoryKnowledgePersOverlapTeacher',
            'EDMemoryKnowledgeUttOverlapTeacher',
            'MSCMemoryKnowledgePersOverlapTeacher',
            'MSCMemoryKnowledgeUttOverlapTeacher',
        ]

    def get_multitask_weights(self) -> Union[List[int], str]:
        """
        Weighting justification:

        6 = persona overlap for Convai2/MSC
        3 = utterance overlap for Convai2/MSC
        2 = persona overlap for BST/ED
        1 = utterance overlap for BST/ED
        """
        return [2, 1, 6, 3, 2, 1, 6, 3]

    def get_task_type(self) -> str:
        return 'knowledge'


class SearchKnowledgeGenerationTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'MsMarcoSearchKnowledgeTeacher',
            'NQSearchKnowledgeTeacher',
            'NQOpenSearchKnowledgeTeacher',
            'NQOpenDialoguesSearchKnowledgeTeacher',
            'SquadSearchKnowledgeTeacher',
            'TriviaQASearchKnowledgeTeacher',
            'WowSearchKnowledgeTeacher',
            'WoiSearchKnowledgeTeacher',
            'FitsSearchKnowledgeTeacher',
        ]

    def get_multitask_weights(self) -> Union[List[int], str]:
        return [1] * len(self.get_teachers())

    def get_task_type(self) -> str:
        return 'knowledge'


class EntityKnowledgeGenerationTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'BSTEntityKnowledgeTeacher',
            'Convai2EntityKnowledgeTeacher',
            'EDEntityKnowledgeTeacher',
            'MSCEntityKnowledgeTeacher',
        ]

    def get_multitask_weights(self) -> Union[List[int], str]:
        return [1] * len(self.get_teachers())

    def get_task_type(self) -> str:
        return 'knowledge'


class SearchDialogueGenerationTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'MsMarcoSearchDialogueTeacher',
            'WowSearchDialogueTeacher',
            'WoiSearchDialogueTeacher',
            'GoogleSgdSearchDialogueTeacher',
            'TaskmasterSearchDialogueTeacher',
            'Taskmaster2SearchDialogueTeacher',
            'Taskmaster3SearchDialogueTeacher',
            'FitsSearchDialogueTeacher',
            'FunpediaWithStyleSearchDialogueTeacher',
        ]

    def get_multitask_weights(self) -> Union[List[int], str]:
        return [2] * (len(self.get_teachers()) - 1) + [1]

    def get_task_type(self) -> str:
        return 'dialogue'


class EntityDialogueGenerationTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'BSTEntityDialogueTeacher',
            'Convai2EntityDialogueTeacher',
            'EDEntityDialogueTeacher',
            'MSCEntityDialogueTeacher',
        ]

    def get_multitask_weights(self) -> Union[List[int], str]:
        """
        Justification for weights:

        Convai2 and MSC are quite larger compared to BST and ED in this task.

        Rather than have stochastic (which is like 10:1), let's just make it 4:1
        """
        return [1, 4, 1, 4]

    def get_task_type(self) -> str:
        return 'dialogue'


class MemoryDialogueGenerationTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'BSTMemoryDialogueFromPersOverlapTeacher',
            'BSTMemoryDialogueFromUttOverlapTeacher',
            'Convai2MemoryDialogueFromPersOverlapTeacher',
            'Convai2MemoryDialogueFromUttOverlapTeacher',
            'EDMemoryDialogueFromPersOverlapTeacher',
            'EDMemoryDialogueFromUttOverlapTeacher',
            'MSCMemoryDialogueFromPersOverlapTeacher',
            'MSCMemoryDialogueFromUttOverlapTeacher',
        ]

    def get_multitask_weights(self) -> Union[List[int], str]:
        """
        Justification:

        Same as Memory Knowledge Generation Teacher
        """
        return [2, 1, 6, 3, 2, 1, 6, 3]

    def get_task_type(self) -> str:
        return 'dialogue'


class VanillaDialogueGenerationTeacher(BB3MultitaskTeacher):
    def get_teachers(self) -> List[str]:
        return [
            'WowVanillaDialogueTeacher',
            'WoiVanillaDialogueTeacher',
            'Convai2VanillaDialogueTeacher',
            'EDVanillaDialogueTeacher',
            'MSCVanillaDialogueTeacher',
            'SaferdialoguesVanillaDialogueTeacher',
            'LightVanillaDialogueTeacher',
            'LightWildVanillaDialogueTeacher',
            'BSTStyleGroundingDialogueTeacher',
            'Convai2StyleGroundingDialogueTeacher',
        ]

    def get_multitask_weights(self) -> Union[List[int], str]:
        """
        Justification:

        Split up Convai2 into half with personas, half without
        """
        return [1] * len(self.get_teachers())

    def get_task_type(self) -> str:
        return 'dialogue'
