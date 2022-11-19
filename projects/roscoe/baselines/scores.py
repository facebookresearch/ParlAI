#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from projects.roscoe.baselines.constants import (
    BARTSCORE_CNN_F,
    BARTSCORE_CNN_PARA_F,
    BARTSCORE_F,
    BARTSCORE_FINETUNED_F,
    BERTSCORE_F,
    BLEURT,
    CTC_CONSISTENCY_SUMMARY,
    CTC_RELEVANCE_SUMMARY,
    PRISM_AVG,
    ROUGE_L,
    ROUGE_2,
    ROUGE_1,
    Example,
    ScoreMe,
    UseRef,
)
import re

from importlib.machinery import SourceFileLoader

BART_SCORE_REPO = "/path_to/BARTScore"
PRISM_SCORE_REPO = "/path_to/prism"
BLEURT_SCORE_REPO = "/path_to/bleurt"

######### Base functionality
SCORES_TO_CLASS = {}


def register_scorer(score_names: List[str]):
    """
    This decorator should be used on all scorer classes so that we can easily specify
    what we want to use based on short name.
    """

    def decorator_register(cls):
        for name in score_names:
            SCORES_TO_CLASS[name] = cls
        return cls

    return decorator_register


class BaselineScorer:
    def get_scores(self, score_me: ScoreMe) -> Dict[str, List[float]]:
        """
        Input: ScoreMe object (includes list of hypo strings + list of context/refs)
        Returns map of short name (since some scorers have multiple metrics at once) and list of scores
        """
        raise NotImplementedError("Downstream scorer must implement if used")

    def score_data(
        self, exs: List[Example], want_ref_types=List[UseRef]
    ) -> Dict[UseRef, Dict[str, List[float]]]:
        """
        Input: List of examples
        Output: Reference + Reference-free scores for metrics

        Child classes should override this if the way it uses refs/hypos/context are different. (Like in CTCScore)
        """
        assert len(exs) > 0
        result = {}
        if UseRef.NO in want_ref_types:
            result[UseRef.NO] = self.get_scores(ScoreMe(exs, UseRef.NO))
        if UseRef.YES in want_ref_types and exs[0].ref is not None:
            result[UseRef.YES] = self.get_scores(ScoreMe(exs, UseRef.YES))
        return result


DEFAULT_DEVICE = 'cuda:0'  # parallelizing this ain't happening :)

###################### SCORES

########### Rouge
try:
    from rouge_score import rouge_scorer
except ImportError:
    raise ImportError("Run `pip install rouge-score`")


@register_scorer([ROUGE_1, ROUGE_2, ROUGE_L])
class RougeBaselineScorer(BaselineScorer):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    def get_scores(self, score_me):
        by_example = [
            self.scorer.score(x, y)
            for x, y in zip(score_me.hypos, score_me.context_ref)
        ]
        return {
            ROUGE_1: [x['rouge1'].fmeasure for x in by_example],
            ROUGE_2: [x['rouge2'].fmeasure for x in by_example],
            ROUGE_L: [x['rougeL'].fmeasure for x in by_example],
        }


########### BLEURT
try:
    from bleurt import score as bleurt_score
except ImportError:
    raise ImportError(
        "Install Bleurt here: https://github.com/google-research/bleurt#installation"
    )


@register_scorer([BLEURT])
class BleurtBaselineScorer(BaselineScorer):
    def __init__(self):
        self.scorer = bleurt_score.BleurtScorer(
            BLEURT_SCORE_REPO + "/bleurt/test_checkpoint"
        )

    def get_scores(self, score_me):
        scores = self.scorer.score(
            references=score_me.context_ref, candidates=score_me.hypos
        )
        return {BLEURT: scores}


######### BERTScore
try:
    from bert_score import BERTScorer
except ImportError:
    raise ImportError("Run `pip install bert-score`")


@register_scorer([BERTSCORE_F])
class BertBaselineScorer(BaselineScorer):
    def __init__(self):
        self.scorer = BERTScorer(
            lang='en', idf=False, rescale_with_baseline=True, device=DEFAULT_DEVICE
        )

    def get_scores(self, score_me):
        (_, _, F) = self.scorer.score(cands=score_me.hypos, refs=score_me.context_ref)
        return {BERTSCORE_F: F}


######### BartScore (and its variants)
# Second argument here should be path to `bart_score.py` of the BARTScore repo
try:
    bart_score = SourceFileLoader(
        "bart_score", BART_SCORE_REPO + "/bart_score.py"
    ).load_module()
    from bart_score import BARTScorer
except ImportError:
    raise ImportError(f"Run `bart-score not found. Make sure it's in {BART_SCORE_REPO}")


class BartscoreBase(BaselineScorer):
    def __init__(self):
        raise NotImplementedError(
            'Must init with a "self.scorer" that loads the right model and a "self.score_type"'
        )

    def get_scores(self, score_me):
        scores = self.scorer.score(srcs=score_me.hypos, tgts=score_me.context_ref)
        return {self.score_type: scores}


@register_scorer([BARTSCORE_F])
class BartscoreBaselineScorer(BartscoreBase):
    def __init__(self):
        self.scorer = BARTScorer(
            device=DEFAULT_DEVICE, checkpoint='facebook/bart-large'
        )
        self.score_type = BARTSCORE_F


@register_scorer([BARTSCORE_CNN_F])
class BartscoreBaselineCNNScorer(BartscoreBase):
    def __init__(self):
        self.scorer = BARTScorer(
            device=DEFAULT_DEVICE, checkpoint='facebook/bart-large-cnn'
        )
        self.score_type = BARTSCORE_CNN_F


@register_scorer([BARTSCORE_CNN_PARA_F])
class BartscoreCNNParaBaselineScorer(BartscoreBase):
    def __init__(self):
        self.scorer = BARTScorer(
            device=DEFAULT_DEVICE, checkpoint='facebook/bart-large-cnn'
        )
        try:
            self.scorer.load(BART_SCORE_REPO + "/bart_score_para_finetuned.pth")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Path here should be to fine tuned BART model from https://github.com/neulab/BARTScore#direct-use"
            )
        self.score_type = BARTSCORE_CNN_PARA_F


@register_scorer([BARTSCORE_FINETUNED_F])
class BartscoreFineTunedBaselineScorer(BartscoreBase):
    def __init__(self):
        """
        NOTE: In the bart_score.py file above, BARTScorer.load() needs to be
        ```
        def load(self, path=None):
            if path is None:
                path = 'models/bart.pth'
            print("Loading from finetuned with custom hacks", path)
            state_dict = torch.load(path, map_location=DEFAULT_DEVICE)
            try:
                self.model.load_state_dict(state_dict)
            except Exception as _:
                state_dict = { x.replace("model.", "", 1) : v for x, v in state_dict.items() }
                self.model.load_state_dict(state_dict)
        ```
        for the loading of the model to work with our fine tuned model
        """
        self.scorer = BARTScorer(
            device=DEFAULT_DEVICE, checkpoint='facebook/bart-large-cnn'
        )
        # Path here to fine-tuend BART Model
        try:
            self.scorer.load(
                BART_SCORE_REPO + "/train/reproduce/trained/fine_tuned_bartscore.pth"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Path here should be to fine tuned BART model from"
                + "https://dl.fbaipublicfiles.com/parlai/projects/roscoe/fine_tuned_bartscore.pth"
            )
        self.score_type = BARTSCORE_FINETUNED_F


######### Prism
prism = SourceFileLoader("prism", PRISM_SCORE_REPO + "/prism.py").load_module()


@register_scorer([PRISM_AVG])
class PrismBaselineScorer(BaselineScorer):
    def __init__(self):
        self.scorer = prism.Prism(
            model_dir=PRISM_SCORE_REPO + '/m39v1/',
            lang='en',
        )

    def get_scores(self, score_me):
        # Need to be explicit about if we're using REF or not since the Prism scorer under the hood does different things
        if score_me.use_ref is UseRef.YES:
            _ref_hypo, _hypo_ref, scores = self.scorer.score(
                cand=score_me.hypos, ref=score_me.context_ref, segment_scores=True
            )
        else:
            scores = self.scorer.score(
                cand=score_me.hypos, src=score_me.context_ref, segment_scores=True
            )

        return {PRISM_AVG: scores}


######### CTC
try:
    from ctc_score import SummarizationScorer
except ImportError:
    raise ImportError("Run `pip install ctc_score`")


@register_scorer([CTC_RELEVANCE_SUMMARY, CTC_CONSISTENCY_SUMMARY])
class CTCSummaryBaselineScorer(BaselineScorer):
    def __init__(self):
        self.scorer = SummarizationScorer(align='D-cnndm')

    def score_data(
        self, exs: List[Example], want_ref_types=List[UseRef]
    ) -> Dict[UseRef, Dict[str, List[float]]]:
        """
        CTCScore uses Refs + reference free things differently, so take care for that.
        """
        assert len(exs) > 0
        result = {}

        # CTC Score is not robust to lots of formatting things, so fix that
        def _fix_text(x: Example):
            def _fix(t):
                if t is None:  # dataset without reference; don't worry about it
                    return t
                if (
                    t == ""
                ):  # hypos sometimes empty cause of perturbations that remove things
                    return "EMPTY"
                t = t.replace("\n", " ")  # tokenization
                t = re.sub(
                    r'<U\+([0-9a-fA-F]{4,6})>', lambda x: chr(int(x.group(1), 16)), t
                )  # not happy with unicode
                t = t.replace('\u00a0', " ")  # ... yup (non breaking space)
                t = t.replace('\t', " ")  # woooooow.
                return t

            x.hypo = _fix(x.hypo)
            x.context = _fix(x.context)
            x.ref = _fix(x.ref)
            return x

        exs = [_fix_text(x) for x in exs]

        if UseRef.NO in want_ref_types:
            result[UseRef.NO] = {
                CTC_RELEVANCE_SUMMARY: [
                    self.scorer.score_relevance(
                        doc=x.context,
                        refs=[
                            x.context
                        ],  # use context for ref since need to fill + this makes sense with underlying code
                        hypo=x.hypo,
                        remove_stopwords=False,  # same as default in CTC code
                    )
                    for x in exs
                ],
                CTC_CONSISTENCY_SUMMARY: [
                    self.scorer.score_consistency(
                        doc=x.context,
                        refs=[],  # unused in underlying code for this
                        hypo=x.hypo,
                        remove_stopwords=False,  # same as default in CTC code
                    )
                    for x in exs
                ],
            }
        if UseRef.YES in want_ref_types and exs[0].ref is not None:
            result[UseRef.YES] = {
                CTC_RELEVANCE_SUMMARY: [
                    self.scorer.score_relevance(
                        doc=x.context,
                        refs=[x.ref],
                        hypo=x.hypo,
                        remove_stopwords=False,  # same as default in CTC code
                    )
                    for x in exs
                ],
                CTC_CONSISTENCY_SUMMARY: [
                    self.scorer.score_consistency(
                        doc=x.ref,
                        refs=[],  # unused in underlying code for this
                        hypo=x.hypo,
                        remove_stopwords=False,  # same as default in CTC code
                    )
                    for x in exs
                ],
            }

        return result
