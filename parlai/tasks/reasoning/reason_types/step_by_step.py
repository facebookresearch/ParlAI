#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
import parlai.utils.logging as logging
from typing import Dict, List, Optional
from collections import defaultdict
import copy
import json
import os
import random
import re
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

import spacy
from checklist.perturb import Perturb
from parlai.tasks.reasoning.base import t_REASON_PROCESSED_RESULT, AbstractReason


class StepPerturbation:
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        return parser

    def __init__(self, opt: Opt, cache: Optional[Dict[str, List[List[str]]]] = None):
        self.random = random.Random(42)
        self.cache = cache

    def perturb(self, example_dict: Dict) -> Dict:
        raise RuntimeError("Child class must implement")

    def batch_perturb(self, examples: List[Dict]) -> List[Dict]:
        """
        Override to parallelize expensive parts of perturb operation.
        """
        return [self.perturb(e) for e in examples]


class ShuffleSteps(StepPerturbation):
    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        steps = example_dict['steps']
        if len(steps) < 2:
            raise RuntimeError(
                "ShuffleSteps requires at least 2 steps.  Make sure `get_data_for_fold`()`"
                + "appropriately filters for these turns when used with this perturbation."
            )
        shuffled = copy.deepcopy(steps)
        # make sure steps are different from original
        while "".join([str(s) for s in shuffled]) == "".join([str(s) for s in steps]):
            # don't use shuffle cause that'll change the original + this a touch faster than reallocating
            shuffled = self.random.sample(steps, len(steps))
        example_dict['steps'] = shuffled
        return example_dict


class DuplicateOneStep(StepPerturbation):
    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        steps = example_dict['steps']
        if len(steps) == 0:
            raise RuntimeError(
                "DuplicateOneStep requires at least 1 step. Make sure `get_data_for_fold()`"
                + "appropriately filters for these turns when used with this perturbation"
            )
        to_dupe = self.random.choice(range(len(steps)))
        result = copy.deepcopy(steps)
        result[to_dupe:to_dupe] = [steps[to_dupe]]
        example_dict['steps'] = result
        return example_dict


class RemoveOneStep(StepPerturbation):
    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        steps = example_dict['steps']
        if len(steps) == 0:
            raise RuntimeError(
                "RemoveOneStep requires at least 1 step. Make sure `get_data_for_fold`()`"
                + "appropriately filters for these turns when used with this perturbation. "
            )
        to_dupe = self.random.choice(range(len(steps)))
        result = copy.deepcopy(steps)
        del result[to_dupe]
        example_dict['steps'] = result
        return example_dict


class SwapOneStep(StepPerturbation):
    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        steps = example_dict['steps']
        if len(steps) < 2:
            raise RuntimeError(
                "SwapOneStep requires at least 2 steps.  Make sure `get_data_for_fold`()`"
                + "appropriately filters for these turns when used with this perturbation."
            )
        idx_1 = idx_2 = 0
        result = copy.deepcopy(steps)
        while str(result[idx_1]) == str(result[idx_2]):
            to_dupe = self.random.sample(range(len(steps)), 2)
            idx_1, idx_2 = to_dupe[0], to_dupe[1]
        result[idx_1], result[idx_2] = result[idx_2], result[idx_1]
        example_dict['steps'] = result
        return example_dict


class ExtrinsicHallucinatedStep(StepPerturbation):
    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        steps = example_dict['steps']
        if 'extrinsic_step' not in example_dict:
            raise RuntimeError(
                "ExtrinsicHallucinatedStep requires 'extrinsic_step' key.  Make sure `get_data_for_fold`()`"
                + "appropriately has this key in the yielded message to use this perturbation."
            )
        extrinsic_step = example_dict['extrinsic_step']
        if len(steps) == 0:
            raise RuntimeError("ExtrinsicHallucinatedStep requires at least 1 step")
        to_dupe = self.random.choice(range(len(steps)))
        result = copy.deepcopy(steps)
        result.insert(to_dupe, extrinsic_step)
        example_dict['steps'] = result
        return example_dict


class ParaphraseSteps(StepPerturbation):
    """
    Paraphrasing steps using model from https://huggingface.co/Vamsi/T5_Paraphrase_Paws.
    """

    def __init__(self, opt: Opt, cache: Optional[Dict[str, List[List[str]]]] = None):
        super().__init__(opt, cache)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "Vamsi/T5_Paraphrase_Paws"
        ).to(self.device)
        self.model.eval()
        self.model.parallelize()

        self.cache = defaultdict(set)
        if cache is not None:
            self._populate_cache(cache)
        self.fillers = ['Well, ', 'Well we know ', 'Hm... ', 'Okay ']

    def _populate_cache(self, cache: Dict[str, List[List[str]]]) -> None:
        def _correct_type(value):
            if isinstance(value, str):
                return tuple(self._separate_sentences(value))
            if isinstance(value, list):
                return tuple(value)
            raise NotImplementedError(f'Unexpected value type: {value} ({type(value)})')

        for k, vs in cache.items():
            self.cache[k] = set(_correct_type(v) for v in vs)

    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        steps = example_dict['steps']
        joined_steps = self._joined_steps(steps)

        if joined_steps not in self.cache:
            if self._reasoning_steps_contain_math(steps):
                # don't call paraphrase model or _separate_sentences
                paraphrased_steps = self._paraphrase_mathy_steps(steps)
                # add to cache
                self.cache[joined_steps].add(tuple(paraphrased_steps))
            else:
                # run paraphrase model
                result = self._batch_run_paraphrase_model([joined_steps])[0]

                # update cache
                if result == joined_steps:
                    logging.warn(
                        "ParaphraseSteps failed for the reasoning chain "
                        + joined_steps
                        + ". Make sure get_data_for_fold()` appropriately filters for these turns"
                        + "when used with this perturbation."
                    )
                    result = self._hard_coded_paraphrase(result)
                self.cache[joined_steps].add(tuple(self._separate_sentences(result)))

        example_dict['steps'] = self.random.choice(list(self.cache[joined_steps]))
        return example_dict

    def batch_perturb(self, examples: List[Dict]) -> List[Dict]:
        steps = [ex['steps'] for ex in examples]
        joined_steps = [(self._joined_steps(s), s) for s in steps]
        joined_steps_not_in_cache = [
            (js, os) for js, os in joined_steps if js not in self.cache
        ]

        if len(joined_steps_not_in_cache) > 0:
            mathy_steps = []
            texty_steps = []
            for js, s in joined_steps_not_in_cache:
                if self._reasoning_steps_contain_math(s):
                    mathy_steps.append((js, s))
                else:
                    texty_steps.append(js)

            for joined, s in mathy_steps:
                # don't call paraphrase model or _separate_sentences
                # add to cache
                self.cache[joined].add(tuple(self._paraphrase_mathy_steps(s)))

            if len(texty_steps) > 0:
                # run paraphrase model
                result = self._batch_run_paraphrase_model(texty_steps)

                # update cache
                for original, perturbed in zip(texty_steps, result):
                    if original == perturbed:
                        perturbed = self._hard_coded_paraphrase(perturbed)
                    self.cache[original].add(tuple(self._separate_sentences(perturbed)))

        # update examples from cache
        for i, (original, _) in enumerate(joined_steps):
            examples[i]['steps'] = self.random.choice(list(self.cache[original]))
        return examples

    def _joined_steps(self, steps: List[str]) -> str:
        if len(steps) == 0:
            raise RuntimeError(
                "ParaphraseSteps requires at least 1 step.  Make sure `get_data_for_fold()`"
                "appropriately filters for these turns when used with this perturbation."
            )
        return " ".join(str(step) for step in steps)

    def _reasoning_steps_contain_math(self, steps: List[str]) -> bool:
        for step in steps:
            if isinstance(step, MWPStep):
                return True
            # TODO: if step has math symbols ( +,-,=,*,/,^,@ ) and low concentration of letters
        return False

    def _paraphrase_mathy_steps(self, steps: List[str]) -> List[str]:
        # add fillers to first step
        steps[0] = self._hard_coded_paraphrase(str(steps[0]))
        # TODO: add fillers to other steps?
        # TODO: rearrange operations into equivalent statements
        return steps

    def _batch_run_paraphrase_model(self, texts: List[str]) -> List[str]:
        encoding = self.tokenizer.batch_encode_plus(
            [f"paraphrase: {s} </s>" for s in texts],
            pad_to_max_length=True,
            return_tensors="pt",
        )
        outputs = self.model.generate(
            input_ids=encoding["input_ids"].to(self.device),
            attention_mask=encoding["attention_mask"].to(self.device),
            max_length=256,
            do_sample=True,
            top_p=0.9,
            early_stopping=True,
        )
        result = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return result

    def _hard_coded_paraphrase(self, chain_of_thought: str) -> str:
        if chain_of_thought.startswith('The'):
            chain_of_thought = 'the' + chain_of_thought[3:]
        return self.random.choice(self.fillers) + chain_of_thought

    def _separate_sentences(self, text: str) -> List[str]:
        text = text.lstrip(
            ' .!?'
        )  # nltk sentence tokenizer breaks when string starts with punctuation
        sents = nltk.tokenize.sent_tokenize(text)
        return [s for s in sents if len(s) > 1]  # remove standalone punctuation


class GrammaticalErrorStep(StepPerturbation):
    """
    Lemmatizes one step in example. - Changes the tense of the verbs in one step to infinitive

    Dependencies:
    import nltk
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


    and make sure wordnet gets unzipped in appropriate location
    """

    def __init__(self, opt: Opt, cache: Optional[Dict[str, List[List[str]]]] = None):
        super().__init__(opt, cache)
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_step(self, words):
        lemmatized_output = ' '.join([self.lemmatizer.lemmatize(w, 'v') for w in words])
        # remove extraneous spaces after joining strings back
        clean_lemmatized_output = re.sub(
            r'\s([?.!"](?:\s|$))', r'\1', lemmatized_output
        )
        return clean_lemmatized_output

    def drop_verb(self, words):
        tags = nltk.pos_tag(words)
        verb_indices = []
        for i, tag in enumerate(tags):
            # keep track of indices of verbs
            if 'V' in tag[1]:
                verb_indices.append(i)
        if not verb_indices:
            return ""
        verb_to_drop = self.random.choice(range(len(verb_indices)))
        del words[verb_indices[verb_to_drop]]
        result = ' '.join(words)
        clean_result = re.sub(r'\s([?.!"](?:\s|$))', r'\1', result)
        return clean_result

    def swap_words(self, tokenized_step):
        tags = nltk.pos_tag(tokenized_step)
        word_indices = []
        for i, tag in enumerate(tags):
            # keep track of indices of words and not punctuation
            # only swaps words and leaves punctuation in place
            if tag[1].isalpha():
                word_indices.append(i)
        if len(word_indices) < 2:
            return ""
        to_dupe = self.random.sample(range(len(word_indices)), 2)
        idx_1, idx_2 = to_dupe[0], to_dupe[1]
        (tokenized_step[word_indices[idx_1]], tokenized_step[word_indices[idx_2]],) = (
            tokenized_step[word_indices[idx_2]],
            tokenized_step[word_indices[idx_1]],
        )
        result = ' '.join(tokenized_step)
        clean_result = re.sub(r'\s([?.!"](?:\s|$))', r'\1', result)
        return clean_result

    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        steps = example_dict['steps']
        if len(steps) == 0:
            raise RuntimeError("NegateSteps requires at least 1 step")
        result = copy.deepcopy(steps)
        grammatical_error_steps = []

        for i, step in enumerate(steps):
            try:
                tok_step = nltk.word_tokenize(str(step))
            except IndexError:
                print(
                    f"WARNING: could not tokenize step {str(step)}. Proceeding to next chain."
                )
                return str(step)
            # perform all possible grammatical errors on each step, then randomly choose 1
            lemmatized_step = self.lemmatize_step(tok_step)
            if tok_step != lemmatized_step:
                grammatical_error_steps.append((i, lemmatized_step))
            dropped_verb_step = self.drop_verb(tok_step)
            if dropped_verb_step != "" and tok_step != dropped_verb_step:
                grammatical_error_steps.append((i, dropped_verb_step))
            swapped_word_step = self.swap_words(tok_step)
            if swapped_word_step != "" and tok_step != swapped_word_step:
                grammatical_error_steps.append((i, swapped_word_step))

        if not grammatical_error_steps:
            raise RuntimeError(
                "GrammaticalErrorStep failed for the reasoning chain "
                + " ".join(str(step) for step in steps)
                + ". Make sure get_data_for_fold()` appropriately filters for these turns when used with this perturbation."
            )

        to_dupe = self.random.choice(range(len(grammatical_error_steps)))
        result[grammatical_error_steps[to_dupe][0]] = grammatical_error_steps[to_dupe][
            1
        ]
        example_dict['steps'] = result
        return example_dict


class NegateStep(StepPerturbation):
    """
    Negates one step in example.

    Dependencies:
    pip3 install checklist
    python -m spacy download en_core_web_sm
    """

    def __init__(self, opt: Opt, cache: Optional[Dict[str, List[List[str]]]] = None):
        super().__init__(opt, cache)
        self.nlp = spacy.load('en_core_web_sm')

    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        steps = example_dict['steps']
        if len(steps) == 0:
            raise RuntimeError(
                "NegateSteps requires at least 1 step. Make sure `get_data_for_fold`()`"
                + "appropriately filters for these turns when used with this perturbation."
            )
        result = copy.deepcopy(steps)
        negated_steps = []
        for i, step_to_dupe in enumerate(result):
            pdata = list(self.nlp.pipe([str(step_to_dupe)]))
            try:
                ret = Perturb.perturb(pdata, Perturb.add_negation)
                if ret.data:
                    negated_steps.append((i, ret.data[0][1]))
            # Perturb.add_negation is noted as still experimental in documentation - occasionally
            # fails on certain cases with a Runtime or Index Error
            except RuntimeError:
                pass
            except IndexError:
                pass
        if not negated_steps:
            raise RuntimeError(
                "NegateStep failed for the reasoning chain "
                + " ".join(str(step) for step in steps)
                + ". Make sure get_data_for_fold()` appropriately filters for these turns"
                + "when used with this perturbation."
            )
        to_dupe = self.random.choice(range(len(negated_steps)))
        result[negated_steps[to_dupe][0]] = negated_steps[to_dupe][1]
        example_dict['steps'] = result
        return example_dict


class SemanticChangeStep(StepPerturbation):
    """
    Change the semantics of one step in example by replacing entities.

    Dependencies:
    python -m spacy download en_core_web_sm
    """

    def __init__(self, opt: Opt, cache: Optional[Dict[str, List[List[str]]]] = None):
        super().__init__(opt, cache)
        pattern = 'NP: {<DT>?<JJ>*<NN.*>}'
        self.chunker = nltk.RegexpParser(pattern)
        self.nlp = spacy.load('en_core_web_sm')

    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        question = example_dict['question']
        entities = self._extract_entities(question)
        if len(entities) == 0:
            raise RuntimeError(
                "SemanticChangeSteps failed to extract any entities from "
                + "the question "
                + str(question)
            )
        steps = example_dict['steps']
        if len(steps) == 0:
            raise RuntimeError(
                "SemanticChangeSteps requires at least 1 step. Make sure `get_data_for_fold`()`"
                + "appropriately filters for these turns when used with this perturbation."
            )

        result = copy.deepcopy(steps)
        changed_steps = []
        for i, step_to_dupe in enumerate(result):
            step_entities = self._extract_entities(step_to_dupe)
            if len(step_entities) == 0:
                continue
            # we want to sample an entity from the step and replace it with
            # another entity from the question. So we first sample from
            # step_entities, and if there are items in the question entities
            # that are not equal to it, then we use the diff to sample
            # the entity to replace it with. However, if the sampled entity
            # is the only entity in the question entities, we try to sample
            # another item from step_entities if there are more items.
            entity_to_replace = self.random.choice(step_entities)
            if (
                len(entities) == 1
                and entities[0] == entity_to_replace
                and len(step_entities) > 0
            ):
                entity_to_replace = self.random.choice(
                    [se for se in step_entities if se != entity_to_replace]
                )
            entities_to_choose_from = [
                ent for ent in entities if ent != entity_to_replace
            ]
            if len(entities_to_choose_from) == 0:
                continue
            chosen_entity = self.random.choice(entities_to_choose_from)
            changed_step = step_to_dupe.lower().replace(
                entity_to_replace, chosen_entity, 1
            )
            changed_steps.append((i, changed_step))
        if not changed_steps:
            raise RuntimeError(
                "SemanticChangeStep failed for the reasoning chain "
                + " ".join(str(step) for step in steps)
                + ". It is likely that this perturbation cannot be applied"
                + " to your dataset."
            )

        to_dupe = self.random.choice(range(len(changed_steps)))
        result[changed_steps[to_dupe][0]] = changed_steps[to_dupe][1]
        example_dict['steps'] = result
        return example_dict

    def _extract_entities(self, text: str) -> List[str]:
        entities = set()
        try:
            # first use spacy
            named_entities = self.nlp(text)
            entities.update(set([X.text for X in named_entities.ents]))
            # next use nltk pos tagger
            text_pos = nltk.pos_tag(nltk.word_tokenize(text))
            chunks = self.chunker.parse(text_pos)
            for subtree in chunks:
                if type(subtree) == nltk.tree.Tree:
                    entities.add(' '.join(t[0] for t in subtree))
        except TypeError:
            pass
        except ValueError:
            pass
        return [entity.lower() for entity in entities]


PERTURBATIONS_LIST = [
    ShuffleSteps,
    DuplicateOneStep,
    RemoveOneStep,
    SwapOneStep,
    ExtrinsicHallucinatedStep,
    ParaphraseSteps,
    GrammaticalErrorStep,
    NegateStep,
    SemanticChangeStep,
]

PERTURBATIONS = {x.__name__: x for x in PERTURBATIONS_LIST}


class MWPStep:
    """
    Class to represent Math Word Problem Steps.

    Receives the input step string and extracts the numbers and mathematical operations
    in the step
    """

    def __init__(self, step: str):
        self.step = step
        self.operations = self.extract_operations()
        self.numbers = self.extract_numbers()

    def extract_operations(self) -> List[str]:
        """
        Finds all instances of the math operations: -, +, *, ^, / in the step.
        """
        if not self.step:
            return []
        try:
            operations = re.findall(r'[-+*^/]', self.step)
        except TypeError as e:
            print(f"TYPE: {type(self.step)}")
            print(f"STEP: {self.step}")
            raise e
        return operations

    def extract_numbers(self) -> List[str]:
        """
        Finds all instances of numbers in the step.
        """
        if not self.step:
            return []
        numbers = re.findall(r'[0-9]+', self.step)
        return numbers

    def get_operations(self) -> List[str]:
        return self.operations

    def get_numbers(self) -> List[str]:
        return self.numbers

    def __str__(self) -> str:
        return self.step

    def __repr__(self) -> str:
        return f'MWPStep("{self.step}")'


class MWPPerturbation(StepPerturbation):
    def steps_empty_check(self, example_dict=Dict):
        mwp_steps = example_dict["steps"]
        if len(mwp_steps) == 0:
            raise RuntimeError(
                "MWPPerturbation requires at least 1 step. Make sure `get_data_for_fold()`"
                + "appropriately filters for these turns when used with this perturbation"
            )

    def perform_replacements(
        self, mwp_step: MWPStep, old_elements: List[str], new_elements: List[str]
    ) -> MWPStep:
        """
        Replaces the old_elements in self.step with the new_elements in the order in
        which they come.

        old_elements : List[str]
            list of the numbers or operations currently in the step in the order in which they appear
        new_elements : List[str]
            list of the numbers or operations to substitute in the step in the order in which they appear
        """

        remaining_old_step = mwp_step.step
        new_step_string = ""
        for old_element, new_element in zip(old_elements, new_elements):
            old_element_idx = remaining_old_step.find(old_element)
            new_step_string += remaining_old_step[:old_element_idx]
            new_step_string += new_element
            remaining_old_step = remaining_old_step[
                old_element_idx + len(old_element) :
            ]
        new_step_string += remaining_old_step
        return MWPStep(new_step_string)


class ShuffleNumbers(MWPPerturbation):
    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        self.steps_empty_check(example_dict)
        result = []
        shuffled = False
        for mwp_step in example_dict["steps"]:
            numbers = mwp_step.get_numbers()
            # at least 2 distinct numbers
            if len(numbers) < 2 or all(
                [numbers[i] == numbers[i + 1] for i in range(len(numbers) - 1)]
            ):
                result.append(mwp_step.step)
                continue
            shuffled_numbers = copy.deepcopy(numbers)
            # make sure steps are different from original
            while " ".join(shuffled_numbers) == " ".join(numbers):
                shuffled_numbers = self.random.sample(numbers, len(numbers))
            perturbed_mwp_step = self.perform_replacements(
                mwp_step, numbers, shuffled_numbers
            )
            result.append(perturbed_mwp_step)
            shuffled = True
        if not shuffled:
            raise RuntimeError(
                "ShuffleNumbers requires at least 2 numbers in one of the steps.  Make sure `get_data_for_fold`()`"
                + "appropriately filters for these turns when used with this perturbation."
            )
        example_dict['steps'] = result
        return example_dict


class ShuffleOperations(MWPPerturbation):
    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        self.steps_empty_check(example_dict)
        result = []
        shuffled = False
        for mwp_step in example_dict["steps"]:
            operations = mwp_step.get_operations()
            # at least 2 distinct operations
            if len(operations) < 2 or all(
                [operations[i] == operations[i + 1] for i in range(len(operations) - 1)]
            ):
                result.append(mwp_step.step)
                continue
            shuffled_operations = copy.deepcopy(operations)
            # make sure steps are different from original
            while shuffled_operations == operations:
                shuffled_operations = self.random.sample(operations, len(operations))
            perturbed_mwp_step = self.perform_replacements(
                mwp_step, operations, shuffled_operations
            )
            shuffled = True
            result.append(perturbed_mwp_step)
        if not shuffled:
            raise RuntimeError(
                "ShuffleOperations requires at least 2 operations in one of the steps.  Make sure `get_data_for_fold`()`"
                + "appropriately filters for these turns when used with this perturbation."
            )
        example_dict['steps'] = result
        return example_dict


class RandomNumber(MWPPerturbation):
    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        self.steps_empty_check(example_dict)
        result = []
        changed = False
        for mwp_step in example_dict["steps"]:
            numbers = mwp_step.get_numbers()
            if len(numbers) == 0:
                result.append(mwp_step.step)
                continue
            new_numbers = copy.deepcopy(numbers)
            while "".join(new_numbers) == "".join(numbers):
                rand_num = str(self.random.randint(0, 100))
                rand_num_idx = self.random.randint(0, len(numbers) - 1)
                new_numbers[rand_num_idx] = rand_num
            perturbed_mwp_step = self.perform_replacements(
                mwp_step, numbers, new_numbers
            )
            result.append(perturbed_mwp_step)
            changed = True
        if not changed:
            raise RuntimeError(
                "RandomNumber requires at least 1 number in the chain.  Make sure `get_data_for_fold`()`"
                + "appropriately filters for these turns when used with this perturbation."
            )
        example_dict['steps'] = result
        return example_dict


class RandomOperation(MWPPerturbation):
    def __init__(self, opt: Opt, cache: Optional[Dict[str, List[str]]] = None):
        super().__init__(opt, cache)
        self.random = random.Random(32)
        self.operations = ['+', '-', '/', '*', '^']

    def perturb(self, example_dict: Dict) -> Dict:
        example_dict = copy.deepcopy(example_dict)
        self.steps_empty_check(example_dict)
        result = []
        changed = False
        for mwp_step in example_dict["steps"]:
            operations = mwp_step.get_operations()
            if len(operations) == 0:
                result.append(mwp_step.step)
                continue
            new_operations = copy.deepcopy(operations)
            while "".join(new_operations) == "".join(operations):
                rand_operation = self.random.choice(self.operations)
                rand_operation_idx = self.random.randint(0, len(operations) - 1)
                new_operations[rand_operation_idx] = rand_operation
            perturbed_mwp_step = self.perform_replacements(
                mwp_step, operations, new_operations
            )
            result.append(perturbed_mwp_step)
            changed = True
        if not changed:
            raise RuntimeError(
                "RandomOperation requires at least 1 operation in the chain.  Make sure `get_data_for_fold`()`"
                + "appropriately filters for these turns when used with this perturbation."
            )
        example_dict['steps'] = result
        return example_dict


MATH_PERTURBATIONS_LIST = [
    ShuffleNumbers,
    ShuffleOperations,
    RandomNumber,
    RandomOperation,
]

MATH_PERTURBATIONS = {x.__name__: x for x in MATH_PERTURBATIONS_LIST}


class StepByStepReason(AbstractReason):
    """
    Class to represent step-by-step style reasoning.

    There are a few simple perturbations that we include by default. For tasks which
    have more complicated perturbations (ex. some way of modifying step-by-step
    reasoning that is also context dependent) it is highly recommended to EXTEND this
    class to a new base class that is specific for that dataset.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("StepByStepReason args")
        group.add_argument(
            "--step-by-step-style",
            choices=["scratchpad", "thought_prefix", "none"],
            default="thought_prefix",
            type=str,
            help="How do we display the step-by-step reasoning? Do we do it like scratchpad (with `scratch_start`"
            + "and `scratch_end` tokens) or with a prefix for the reasoning turns?",
        )
        group.add_argument("--thought-token", default="THINK: ", type=str)
        group.add_argument(
            "--scratch-start-token", default="__scratch_start__", type=str
        )
        group.add_argument("--scratch-end-token", default="__scratch_end__", type=str)
        group.add_argument(
            "--step-perturbations",
            default=[],
            nargs="*",
            help="Are we going to try to mess with any of the steps in this class?",
            type=str,
            choices=PERTURBATIONS.keys(),
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.cache_file_path = os.path.join(
            opt['datapath'], 'reasoning', 'step_by_step', 'perturbation_cache.json'
        )
        self.perturbation_cache = (
            json.load(open(self.cache_file_path))
            if os.path.exists(self.cache_file_path)
            else {}
        )
        self.perturbations = [
            PERTURBATIONS[p](self.opt, self.perturbation_cache.get(p, None))
            for p in self.opt.get("step_perturbations", [])
        ]

    def get_maybe_perturbed_steps(self, example_dict: Dict) -> List[str]:
        for perturbation in self.perturbations:
            example_dict = perturbation.perturb(example_dict)
        return example_dict['steps']

    def batch_get_maybe_perturbed_steps(
        self, batch_of_examples: List[Dict]
    ) -> List[List[str]]:
        for perturbation in self.perturbations:
            batch_of_examples = perturbation.batch_perturb(batch_of_examples)
        return [ex['steps'] for ex in batch_of_examples]

    def format_steps(self, steps_to_format):
        if self.opt.get("step_by_step_style") == "scratchpad":
            steps = (
                [self.opt["scratch_start_token"]]
                + steps_to_format
                + [self.opt["scratch_end_token"]]
            )
        elif self.opt.get("step_by_step_style") == "none":
            steps = [str(x) for x in steps_to_format]
        else:
            steps = [self.opt["thought_token"] + str(x) for x in steps_to_format]
        return steps

    def get_full_reason_text(self, example_dict) -> t_REASON_PROCESSED_RESULT:
        original_steps_raw = example_dict["steps"]
        processed_steps = self.format_steps(
            self.get_maybe_perturbed_steps(example_dict)
        )
        original_steps = self.format_steps(original_steps_raw)
        return (
            "",
            self.inner_separator.join(processed_steps),
            {
                "steps": processed_steps,
                "perturbations": self.opt.get("step_perturbations", []),
                "original_steps": original_steps,
            },
        )

    def batch_get_full_reason_text(
        self, examples: List[Dict]
    ) -> List[t_REASON_PROCESSED_RESULT]:
        original_steps = [self.format_steps(ex["steps"]) for ex in examples]
        processed_steps = self.batch_get_maybe_perturbed_steps(examples)
        processed_steps = [self.format_steps(s) for s in processed_steps]
        return [
            (
                "",
                self.inner_separator.join(ps),
                {
                    "steps": ps,
                    "perturbations": self.opt.get("step_perturbations", []),
                    "original_steps": os,
                },
            )
            for os, ps in zip(original_steps, processed_steps)
        ]

    def save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        with open(self.cache_file_path, 'w') as dest:
            for name, p in zip(
                self.opt.get("step_perturbations", []), self.perturbations
            ):
                self.perturbation_cache[name] = p.cache
            json.dump(
                self.perturbation_cache,
                dest,
                default=(lambda x: list(x) if isinstance(x, set) else x),
            )


class MWPStepsReason(StepByStepReason):
    """
    Class to represent step-by-step style reasoning for math datasets.

    Replaces example_dict["steps"] with MWPSteps to support functionality pertaining to
    numbers and mathematical operations contained within the steps
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("MWPStepsReason args")
        group.add_argument(
            "--math-step-perturbations",
            default=[],
            nargs="*",
            help="Are we going to try to mess with any of the math steps in this class?",
            type=str,
            choices=MATH_PERTURBATIONS.keys(),
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        for perturbation in self.opt.get("math_step_perturbations", []):
            self.perturbations.append(MATH_PERTURBATIONS[perturbation](self.opt))

    def get_full_reason_text(self, example_dict) -> t_REASON_PROCESSED_RESULT:
        out_dict = copy.deepcopy(example_dict)
        if len(self.perturbations) > 0:
            out_dict["steps"] = []
            for step in example_dict["steps"]:
                if not step:
                    continue
                if type(step) is str:
                    out_dict["steps"].append(MWPStep(step))
                elif type(step) is MWPStep:
                    out_dict["steps"].append(step)
                else:
                    raise ValueError(f"Unknown step type: {type(step)} for step {step}")
        return super().get_full_reason_text(out_dict)
