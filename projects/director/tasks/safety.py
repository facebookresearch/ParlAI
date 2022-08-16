#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple
from typing import Optional
from parlai.core.agents import (
    Agent,
    create_agent_from_model_file,
    create_agent_from_shared,
)
from parlai.core.params import ParlaiParser
from parlai.core.message import Message
from parlai.core.opt import Opt

from parlai.core.mutators import (
    register_mutator,
    ManyEpisodeMutator,
    EpisodeMutator,
)
import parlai.tasks.bot_adversarial_dialogue.agents as bad
import parlai.tasks.dialogue_safety.agents as bibifi
import parlai.utils.logging as logging

from parlai.core.metrics import AverageMetric


@register_mutator('LTR')
class LeftToRightMutator(ManyEpisodeMutator):
    """
    Mutator that breaks down episodes into all partial sequences (Left To Right).
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[Message]:
        new_episodes = []
        for message in episode:
            label_words = message['labels'][0].split()
            if len(label_words) < 2:
                continue
            for i in range(1, len(label_words) + 1):
                new_message = message.copy()
                label = ' '.join(label_words[:i])
                new_message.force_set('labels', [label])
                new_episodes.append([new_message])
        return new_episodes


@register_mutator('DIRECTOR_LTR')
class EDCLeftToRightMutator(ManyEpisodeMutator):
    """
    EDCLeftToRightMutator prepares data for training left to right (LTR) classifier for
    Encoder-Decoder Classifier (EDC) model.

    This limits to context to all but last utterance that is fed to the encoder.
    The final utterance is considered as a label for the decoder and the attribute/classifier
    labels are stored seperately marking the final utterance pos. or neg.

    This mutator also adds a is_ltr flag to differentiate classifier exs from the generator
    exs which are used to finetune the generator model.
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        new_episodes = []
        for message in episode:
            text = message['text']
            utterances = text.split('\n')

            if len(utterances) < 2:
                continue

            new_message = message.copy()
            new_message.force_set('is_ltr', True)
            new_message.force_set('classifier_label', message['labels'][0])
            new_text = '\n'.join(utterances[:-1])
            new_message.force_set('text', new_text)
            new_message.force_set('labels', [utterances[-1]])
            new_episodes.append([new_message])
        return new_episodes


@register_mutator('DIRECTOR_LTR_COPY')
class EDCLeftToRightMutatorCopy(ManyEpisodeMutator):
    """
    EDCLeftToRightMutator prepares data for training left to right (LTR) classifier for
    Encoder-Decoder Classifier (EDC) model.

    This limits to context to all but last utterance that is fed to the encoder.
    The final utterance is considered as a label for the decoder and the attribute/classifier
    labels are stored seperately marking the final utterance pos. or neg.

    This mutator also adds a is_ltr flag to differentiate classifier exs from the generator
    exs which are used to finetune the generator model.
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        new_episodes = []
        for message in episode:
            text = message['text']
            utterances = text.split('\n')

            if len(utterances) < 2:
                utterances.insert(0, utterances[0])

            new_message = message.copy()
            new_message.force_set('is_ltr', True)
            new_message.force_set('classifier_label', message['labels'][0])
            new_text = '\n'.join(utterances[:-1])
            new_message.force_set('text', new_text)
            new_message.force_set('labels', [utterances[-1]])
            new_episodes.append([new_message])
        return new_episodes


@register_mutator('DIRECTOR_LTR_EMPTY')
class EDCLeftToRightMutatorEmpty(ManyEpisodeMutator):
    """
    EDCLeftToRightMutator prepares data for training left to right (LTR) classifier for
    Encoder-Decoder Classifier (EDC) model.

    This limits to context to all but last utterance that is fed to the encoder.
    The final utterance is considered as a label for the decoder and the attribute/classifier
    labels are stored seperately marking the final utterance pos. or neg.

    This mutator also adds a is_ltr flag to differentiate classifier exs from the generator
    exs which are used to finetune the generator model.
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        new_episodes = []
        for message in episode:
            text = message['text']
            utterances = text.split('\n')

            if len(utterances) < 2:
                utterances.insert(0, "")

            new_message = message.copy()
            new_message.force_set('is_ltr', True)
            new_message.force_set('classifier_label', message['labels'][0])
            new_text = '\n'.join(utterances[:-1])
            new_message.force_set('text', new_text)
            new_message.force_set('labels', [utterances[-1]])
            new_episodes.append([new_message])
        return new_episodes


@register_mutator('neg_only')
class NegOnlyMutator(ManyEpisodeMutator):
    """
    Mutator that filters to only the neg set.
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[Message]:
        neg_kwords = ['neg', '__notok__']
        new_episodes = []
        for message in episode:
            if message['labels'][0] in neg_kwords or (
                'classifier_label' in message
                and message['classifier_label'] in neg_kwords
            ):
                new_episodes.append([message])
        return new_episodes


@register_mutator('pos_only')
class PosOnlyMutator(ManyEpisodeMutator):
    """
    Mutator that filters to only the neg set.
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[Message]:
        pos_kwords = ['pos', '__ok__']
        new_episodes = []
        for message in episode:
            if message['labels'][0] in pos_kwords or (
                'classifier_label' in message
                and message['classifier_label'] in pos_kwords
            ):
                new_episodes.append([message])
        return new_episodes


class ClassifierMetricTeacher(bibifi.DefaultTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        parser.add_argument(
            '--eval-classifier-model-file',
            required=False,
            type=str,
            help='Filepath for evaluation classifier to evaluate the model generation.',
        )
        parser.add_argument(
            '--include-label-cand-only',
            type='bool',
            default=True,
            help='When passing inputs to the classifier, use only the label targets if set to True.',
        )
        parser.add_argument(
            '--truncate-prediction-at',
            type=int,
            default=-1,
            help='Truncate the prediction of the model after n words before feeding it to the classifier.',
        )
        parser.add_argument(
            '--limit-classifier-examples-at',
            type=int,
            default=-1,
            help='Limit the amount of classifier examples (used during training).',
        )
        parser.add_argument(
            '--eval-classifier-use-cuda',
            type=bool,
            default=False,
            help='Use the gpu for the eval classifier. This does not work during distributed training.',
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if opt['limit_classifier_examples_at'] > 0:
            self.data = self.data[: opt['limit_classifier_examples_at']]
        self.include_label_cand_only = opt['include_label_cand_only']
        self.truncate_prediction_at = opt['truncate_prediction_at']
        assert (
            'eval_classifier_model_file' in opt
            and opt['eval_classifier_model_file'] is not None
        ), 'You must provide --eval-classifier-model-file for ClassifierMetricTeacher.'
        if not shared:
            self.classifier = create_agent_from_model_file(
                opt['eval_classifier_model_file'],
                opt_overrides={
                    'datatype': 'valid',
                    'no_cuda': not opt['eval_classifier_use_cuda'],
                },
            )
            self.classifier.opt.log()
        else:
            logging.info('Load the classifier from shared')
            self.classifier = create_agent_from_shared(shared['classifier'])
        self.context = []
        DEFAULT_DELIM = '\n'
        self.delimiter = opt.get('delimiter', DEFAULT_DELIM)

    def share(self):
        shared = super().share()
        shared['classifier'] = self.classifier.share()
        return shared

    def predict(self, context: str) -> Message:
        """
        Use classifier to predict given the context.

        :param context:
            The input context to classify.

        :return output:
            return output from classifier act.
        """
        assert isinstance(self.classifier, Agent)
        obs = Message({'text': context, 'episode_done': True})
        self.classifier.observe(obs)
        act = self.classifier.act()
        assert isinstance(act, Message)
        return act

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ) -> None:
        """
        Compute Classifier for a model response.

        :param teacher_action:
            The message last sent from this teacher.
        :param labels:
            The previous correct labels
        :param model_response:
            The raw response from the model
        """
        if self.classifier is None:
            return
        if not model_response or not model_response.get('text'):
            return
        self.context.append(teacher_action['text'])
        correct_class = self.classifier.ref_class
        model_text = model_response['text']
        if self.truncate_prediction_at > 0:
            model_text = ' '.join(model_text.split()[: self.truncate_prediction_at])

        if self.include_label_cand_only:
            classifier_act = self.predict(model_text)
        else:
            context = self.delimiter.join(self.context)
            classifier_act = self.predict(context + self.delimiter + model_text)

        if 'text' in classifier_act:
            predicted_class = classifier_act['text']
            correct_prediction = int(predicted_class == correct_class)

            self.metrics.add('classifier_accuracy', AverageMetric(correct_prediction))

        if teacher_action['episode_done']:
            self.context = []
        else:
            assert labels
            self.context.append(labels[0])


class SafeWikiToxicEvalTeacher(ClassifierMetricTeacher):
    pass


class SafeBADTeacher(bad.BotAdversarialDialogueTeacher):
    pass


class SafeAdvTeacher(bibifi.AdversarialTeacher):
    pass


class SafeStdTeacher(bibifi.StandardTeacher):
    pass


class SafeMultiTeacher(bibifi.MultiturnTeacher):
    pass


class SafeWikiToxicTeacher(bibifi.DefaultTeacher):
    pass


@register_mutator('safety_to_LTR')
class SafetyLTRMutator(ManyEpisodeMutator):
    """
    Mutator that takes safety data with __ok__ and __notok__ labels and converts to
    "pos" and "neg" which we use elsewhere.

    It assumes the last line of 'text' is the last dialogue utterance, and splits that
    by word for the left-to-right classifier.
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[Message]:
        new_episodes = []
        for message in episode:
            new_message = message.copy()
            label = message['labels'][0]
            if label == '__notok__':
                label = 'neg'
            else:
                label = 'pos'
            new_message.force_set('labels', [label])
            text = '\n'.join(message['text'].split('\n')[:-1])
            label_words = message['text'].split('\n')[-1].split()
            if len(label_words) < 2:
                continue
            for i in range(1, len(label_words) + 1):
                new_message2 = new_message.copy()
                label = ' '.join(label_words[:i])
                new_text = text + '\n' + label
                new_message2.force_set('text', new_text)
                new_episodes.append([new_message2])
        return new_episodes


@register_mutator('safety_relabel_classes')
class SafetyRelabelClassesMutator(EpisodeMutator):
    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        new_episodes = []
        for message in episode:
            new_message = message.copy()
            label = message['labels'][0]
            if label == '__notok__':
                label = 'neg'
            else:
                label = 'pos'
            new_message.force_set('labels', [label])
            new_episodes.append(new_message)
        return new_episodes
