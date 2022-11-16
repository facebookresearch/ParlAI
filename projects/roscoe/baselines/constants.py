#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class UseRef(Enum):
    YES = "with_ref"
    NO = "no_ref"


@dataclass
class Example:
    context: str
    hypo: str
    ref: Optional[str] = None


@dataclass
class ScoreMe:
    hypos: List[str]
    context_ref: List[str]
    use_ref: UseRef

    def __init__(self, exs: List[Example], use_ref: UseRef):
        self.use_ref = use_ref
        if self.use_ref is UseRef.YES:
            self.context_ref = [x.ref for x in exs]
        else:
            self.context_ref = [x.context for x in exs]
        assert len(self.context_ref) > 0 and self.context_ref[0] is not None
        self.hypos = [x.hypo for x in exs]


############### SCORES
ROUGE_1 = "rouge_1"
ROUGE_2 = "rouge_2"
ROUGE_L = "rouge_l"
BLEURT = "bleurt"
BERTSCORE_F = "bertScore_f"
BARTSCORE_F = "bartScore_f"
BARTSCORE_CNN_F = "bartScore_cnn_f"
BARTSCORE_CNN_PARA_F = "bartscore_cnn_para_f"
BARTSCORE_FINETUNED_F = "bartscore_finetuned_f"
PRISM_AVG = "prism_avg"  # note: we're actually using PRISM where it changes underlying behavior depending on references or not
CTC_RELEVANCE_SUMMARY = "ctc_relevance_summary"
CTC_CONSISTENCY_SUMMARY = "ctc_consistency_summary"

BASELINE_SCORES = [  # Use this to hide metrics we don't want to use anymore
    ROUGE_1,
    ROUGE_2,
    ROUGE_L,
    BLEURT,
    BERTSCORE_F,
    BARTSCORE_F,
    #    BARTSCORE_CNN_F,
    BARTSCORE_CNN_PARA_F,
    BARTSCORE_FINETUNED_F,
    PRISM_AVG,
    CTC_RELEVANCE_SUMMARY,
    CTC_CONSISTENCY_SUMMARY,
]

DEFAULT_INPUT_PATH = f"./projects/roscoe/roscoe_data/generated"
################ Datasets
INPUT_DATA_FILES_HUMAN = {
    "drop": f"{DEFAULT_INPUT_PATH}/drop.json",
    "esnli": f"{DEFAULT_INPUT_PATH}/esnli.json",
    "cosmos": f"{DEFAULT_INPUT_PATH}/cosmos.json",
    "gsm8k": f"{DEFAULT_INPUT_PATH}/gsm8k.json",
    "semeval": f"{DEFAULT_INPUT_PATH}/semevalcommonsense.json",
}
INPUT_DATA_HUMAN = list(INPUT_DATA_FILES_HUMAN.keys())

INPUT_DATA_SYNTHETIC = [
    "aqua",
    "asdiv",
    "entailment_bank",
    "eqasc",
    "math",
    "proofwriter",
    #    "strategy_qa", # used for train + valid only
]

ALL_DATASETS = INPUT_DATA_HUMAN + INPUT_DATA_SYNTHETIC

############### Perturbations

PERTURBATIONS = [
    "ShuffleSteps",
    "DuplicateOneStep",
    "RemoveOneStep",
    "SwapOneStep",
    "ExtrinsicHallucinatedStep",
    "ParaphraseSteps",
    "GrammaticalErrorStep",
    "NegateStep",
    "SemanticChangeStep",
    "ShuffleNumbers",
    "ShuffleOperations",
    "RandomNumber",
    "RandomOperation",
]
