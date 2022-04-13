#!/usr/bin/env python3

import base64
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from glob import glob
from io import BytesIO
import itertools
import json
import math
import os
import pickle
import random
import regex
import subprocess
import tempfile
from typing import Any, Callable, List, Optional, Tuple
import uuid

import altair as alt
from flask import Flask, send_from_directory, request, escape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mephisto.core.local_database import LocalMephistoDB
from mephisto.core.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.data_model.worker import Worker


USERDIR = os.environ["PARLAI_HOME"]
MCDIR = USERDIR + "/projects/metacognition"
DATADIR0 = "/some/storage/project"
DATADIR1 = DATADIR0 + "/models"
DATADIR2 = DATADIR1 + "/results"

Certainty = Enum("Certainty", "CERTAIN UNCERTAIN DONTKNOW")


def strip_control(q):
    for suffix in "IDK TRY YEA EVA SAME DIFF".split():
        q = q.split(f"<{suffix}>")[0].strip()
    return q


def glob1(path):
    print("getting", path)
    (p,) = glob(path)
    return p


class CachedGroup:
    paths: Tuple[str]
    items: Tuple[Any]

    def loader(self, f, path):
        return pickle.load(f)

    def generator(self):
        raise NotImplementedError("No generator defined!")

    def dumper(self, obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)

    def unsafe_get_obj(self, path):
        with open(path, "rb") as f:
            return self.loader(f, path)

    def __init__(self, paths, no_cache=False):
        self.paths = paths
        if not no_cache:
            try:
                self.items = tuple(self.unsafe_get_obj(path) for path in paths)
                print("Hit on", paths)
                return
            except FileNotFoundError:
                pass
        print("Generating", paths, end="...")
        self.items = tuple(x for x in self.generator())
        print("done.")
        assert len(self.items) == len(self.paths)
        if not no_cache:
            for obj, path in zip(self.items, self.paths):
                self.dumper(obj, path)


@dataclass
class QASample:
    # User-passed
    question: str
    gold: List[str]
    prediction: str
    # Optional
    beam_texts: List[str]
    maxprob: float
    # Automatically calculated
    is_correct: bool
    certainty: Certainty

    DONTKNOW_STARTS = ["I don't", "I'm not", "I've never"]

    def __init__(
        self,
        question: str,
        gold: List[str],
        prediction: str,
        beam_texts: List[str] = None,
        maxprob: float = None,
    ):
        self.question = question
        self.prediction = prediction
        self.beam_texts = sorted(beam_texts, key=lambda t: t[1], reverse=True)
        assert not beam_texts or self.beam_texts[0][0] == self.prediction
        self.maxprob = maxprob
        # Tokenize some
        self.tok_question = self._tok(question)
        self.tok_prediction = self._tok(prediction)
        # Guess certainties
        if any(
            [self.tok_prediction.startswith(self._tok(s)) for s in self.DONTKNOW_STARTS]
        ):
            self.certainty = Certainty.DONTKNOW
        elif self.tok_prediction.startswith("i "):
            self.certainty = Certainty.UNCERTAIN
        else:
            self.certainty = Certainty.CERTAIN
        # Update gold
        self.set_gold(gold)

    def set_gold(self, gold):
        # Update gold
        self.gold = gold
        self.tok_gold = [self._tok(g) for g in gold]
        # Evaluate
        self.is_correct = self._is_this_one_correct(self.tok_prediction)
        self.beam_is_correct = tuple(
            self._is_this_one_correct(self._tok(b)) for b, _ in self.beam_texts
        )
        return self

    @classmethod
    def from_json(cls, line: str) -> Optional["QASample"]:
        gold, pred = json.loads(line)["dialog"][0]
        if set(gold.keys()) == set(["episode_done", "id"]):
            return None
        return cls(
            question=gold["text"],
            gold=gold["eval_labels"] if "eval_labels" in gold else [],
            prediction=pred["text"],
            beam_texts=pred.get("beam_texts", []),
            maxprob=pred.get("maxprob", None),
        )

    def _is_this_one_correct(self, tok_prediction):
        return any(
            (" " + tg + " ") in (" " + tok_prediction + " ")
            for tg in self.tok_gold
            if tg
            not in self.tok_question.replace("  ", " ")
            .split(" <same>")[0]
            .replace(" <eva>", "")
            .replace(" <idk>", "")
            .replace(" <try>", "")
            .replace(" <yea>", "")
        )

    @staticmethod
    def _tok(x: str) -> str:
        for s in "'/(),.!?-\"":
            x = x.replace(s, f" {s} ")
        return " ".join(x.split()).lower()


@dataclass
class TriviaQARun:
    name: str
    teacher: str
    filename: str
    samples: List[QASample]
    n: int
    stats: OrderedDict

    # (name, numerator-predicate, denominator-getter), None == separator in rendering
    METRICS = [
        ("✔", lambda s: s.is_correct, lambda r: r.n),
        ("✗", lambda s: not s.is_correct, lambda r: r.n),
        (None, None, lambda _: None),
        ("‼️", lambda s: s.certainty == Certainty.CERTAIN, lambda r: r.n),
        ("⁉️", lambda s: s.certainty == Certainty.UNCERTAIN, lambda r: r.n),
        ("❓", lambda s: s.certainty == Certainty.DONTKNOW, lambda r: r.n),
        (None, None, lambda _: None),
        (
            "✔|‼️",
            lambda s: s.is_correct and s.certainty == Certainty.CERTAIN,
            lambda r: len([s for s in r.samples if s.certainty == Certainty.CERTAIN]),
        ),
        (
            "✔|⁉️",
            lambda s: s.is_correct and s.certainty == Certainty.UNCERTAIN,
            lambda r: len([s for s in r.samples if s.certainty == Certainty.UNCERTAIN]),
        ),
        (
            "✔|❓",
            lambda s: s.is_correct and s.certainty == Certainty.DONTKNOW,
            lambda r: len([s for s in r.samples if s.certainty == Certainty.DONTKNOW]),
        ),
        (None, None, lambda _: None),
        (
            "‼️,✔",
            lambda s: s.certainty == Certainty.CERTAIN and s.is_correct,
            lambda r: r.n,
        ),
        (
            "‼️,✗",
            lambda s: s.certainty == Certainty.CERTAIN and not s.is_correct,
            lambda r: r.n,
        ),
        (
            "⁉️,✔",
            lambda s: s.certainty == Certainty.UNCERTAIN and s.is_correct,
            lambda r: r.n,
        ),
        (
            "⁉️,✗",
            lambda s: s.certainty == Certainty.UNCERTAIN and not s.is_correct,
            lambda r: r.n,
        ),
        (
            "❓,✔",
            lambda s: s.certainty == Certainty.DONTKNOW and s.is_correct,
            lambda r: r.n,
        ),
        (
            "❓,✗",
            lambda s: s.certainty == Certainty.DONTKNOW and not s.is_correct,
            lambda r: r.n,
        ),
    ]

    def __init__(self, filename):
        # Construct metadata
        assert filename.endswith("_replies.jsonl"), filename
        colonsplit = os.path.basename(filename)[:-14].split(":")
        undersplit = colonsplit[0].split("_")
        assert undersplit[-1] in [
            "triviaqa",
            "internal.projects.metacognition.agents",
            "external.projects.metacognition.agents",
        ]
        self.name = "_".join(undersplit[:-1])
        self.teacher = ":".join(colonsplit[1:])
        self.filename = filename
        # Read Samples
        with open(filename) as f:
            self.samples = [QASample.from_json(line) for line in f]
        self.samples = [s for s in self.samples if s is not None]
        self.n = len(self.samples)
        # Gather metrics
        counts = Counter()
        for s in self.samples:
            for key, pred, _ in self.METRICS:
                if key is not None and pred(s):
                    counts[key] += 1
        self.stats = OrderedDict(
            (
                (name, (counts[name], denominator_getter(self)))
                for (name, _, denominator_getter) in self.METRICS
            )
        )
        del counts

    @staticmethod
    def get_run(path, no_cache=False):
        (run,) = CachedRuns((path + ".TriviaQARun.pkl",), no_cache=no_cache).items
        return run

    @staticmethod
    def all_runs_in(path, must_contain=None):
        (_, _, filenames) = next(os.walk(path))
        for fn in sorted(filenames):
            if must_contain is not None and must_contain not in fn:
                continue
            if fn.endswith("_replies.jsonl") and "NoEvidenceUnion" in fn:
                yield TriviaQARun.get_run(os.path.join(path, fn))


class CachedRuns(CachedGroup):
    def generator(self):
        return tuple(
            TriviaQARun(path.replace(".TriviaQARun.pkl", "")) for path in self.paths
        )


class CachedQuestionEmbeddings(CachedGroup):
    embedders = ["laser", "roberta"]

    def __init__(
        self,
        embedder: str,
        laserdir: str = f"{USERDIR}/LASER",
        rolemodel: str = DATADIR2
        + "NoEvidenceUnion_blender_90M_default_triviaqa:NoEvidenceUnion_replies.jsonl",
    ):
        self.embedder = embedder
        self.laserdir = laserdir
        self.rolemodel = rolemodel
        _ts = "trainset_" if "trainset" in rolemodel else ""
        super().__init__(
            (DATADIR2 + f"{self.embedder}_{_ts}compiled_question_embeddings.pkl",)
        )

    def generator(self):
        questions = []
        with open(self.rolemodel) as f:
            for line in f:
                gold = json.loads(line)["dialog"][0][0]
                if "eval_labels" in gold:
                    questions.append(gold["text"])

        if self.embedder == "roberta":
            # This is a mean-pooled fine-tuned RoBERTa
            embedder = SentenceTransformer("roberta-large-nli-stsb-mean-tokens")
            batchsize = 512  # just doing this for a progressbar
            question_embeddings = []
            for start in tqdm(range(0, len(questions), batchsize), "Embedding batches"):
                question_embeddings += embedder.encode(
                    questions[start : start + batchsize]
                )
            embeddings = np.stack(question_embeddings)
        elif self.embedder == "laser":
            # Laser is a multilingual max-pooled BiLSTM
            with tempfile.NamedTemporaryFile("wt") as qf:
                print("\n".join(questions), file=qf)
                embfile = qf.name + ".embs"
                cmds = (
                    f"export LASER='{self.laserdir}'; cd $LASER; "
                    + f"< '{qf.name}' python3 source/embed.py "
                    + "--encoder models/bilstm.93langs.2018-12-26.pt "
                    + "--token-lang en --bpe-codes models/93langs.fcodes "
                    + f"--output '{embfile}' --verbose 2>&1"
                )
                subprocess.run(cmds, shell=True, check=True)
            embeddings = np.fromfile(embfile, dtype=np.float32, count=-1)
            assert embeddings.shape[0] == len(questions) * 1024, embeddings.shape
            embeddings.resize(len(questions), 1024)
            os.remove(embfile)
        elif self.embedder.endswith("_replies.jsonl"):
            embeddings = []
            with open(self.embedder) as f:
                for line in f:
                    gold, pred = json.loads(line)["dialog"][0]
                    if set(gold.keys()) == set(["episode_done", "id"]):
                        continue
                    embeddings.append(
                        np.load(
                            BytesIO(base64.b64decode(pred["first_decoder_attendeds"]))
                        )["arr_0"].reshape(-1)
                    )
            embeddings = np.stack(embeddings)
        else:
            raise ValueError(f"Illegal embedding embedder {self.embedder}!")
        assert len(questions) == embeddings.shape[0]
        return ((questions, embeddings),)


def get_metrics(run: TriviaQARun = None):
    return [
        (
            "Correctness",
            "red: ✗ / blue: ✔",
            {
                "red!.3": [not s.is_correct for s in run.samples],
                "blue!.8": [s.is_correct for s in run.samples],
            }
            if run is not None
            else None,
        ),
        (
            "Certainty",
            "red: ❓ / green: ⁉️ / blue: ‼️",
            {
                "red!.4": [s.certainty == Certainty.DONTKNOW for s in run.samples],
                "blue!.4": [s.certainty == Certainty.CERTAIN for s in run.samples],
                "green!1.0": [s.certainty == Certainty.UNCERTAIN for s in run.samples],
            }
            if run is not None
            else None,
        ),
    ]


class CachedReductionPlotGroup(CachedGroup):
    reductions = ["PCA", "PCA128--tSNE"]
    stdizers = [True, False]

    def __init__(self, run: TriviaQARun, embedder: str, reduction: str, stdize: bool):
        self.run = run
        self.embedder = embedder
        self.reduction = reduction
        self.stdize = stdize
        self.source = self.run.filename if self.embedder == "decoder" else self.embedder
        self.img_filename = (
            f"figures/{self.run.name}__{self.embedder}__"
            + f"{self.reduction}_stdize_{self.stdize}__{{}}.png"
        )
        super().__init__(
            tuple(DATADIR2 + self.img_filename.format(m) for m, _, _ in get_metrics())
        )

    def loader(self, _f, path):
        return path.replace(DATADIR2, "/")

    def dumper(self, _obj, _path):
        pass

    def generator(self):
        # Get and reduce embeddings
        ((_, embeddings),) = CachedQuestionEmbeddings(
            self.source, rolemodel=self.run.filename
        ).items
        if self.stdize:
            embeddings = StandardScaler().fit_transform(embeddings)
        if self.reduction == "PCA":
            embeddings = PCA(n_components=2).fit_transform(embeddings)
        elif self.reduction == "PCA128--tSNE":
            # Seems to scale pretty linearly in dimensions...
            embeddings = PCA(n_components=128).fit_transform(embeddings)
            embeddings = TSNE(n_components=2).fit_transform(embeddings)
        else:
            raise ValueError(f"Illegal reduction {self.reduction}!")

        # Render plots
        for (metric, _, mapping) in get_metrics(self.run):
            plt.clf()
            plt.figure(figsize=(4, 4))
            for color, predicate in mapping.items():
                color, alpha = color.split("!")
                alpha = float(alpha)
                plt.scatter(
                    embeddings[predicate, 0],
                    embeddings[predicate, 1],
                    marker=".",
                    s=0.2,
                    alpha=alpha,
                    color=color,
                )
            plt.title(f"{self.embedder}, {self.reduction} (stdize: {self.stdize})")
            plt.savefig(DATADIR2 + self.img_filename.format(metric))
        return tuple("/" + self.img_filename.format(m) for m in get_metrics(self.run))


class CachedPPLPlotGroup(CachedGroup):
    def __init__(self, run: TriviaQARun):
        self.run = run
        self.plot_filenames = [
            f"figures/{self.run.name}__ppl_absolute.png",
            f"figures/{self.run.name}__ppl_relative.png",
        ]
        super().__init__(tuple(DATADIR2 + filename for filename in self.plot_filenames))

    def loader(self, _f, path):
        return path.replace(DATADIR2, "/")

    def dumper(self, _obj, _path):
        pass

    def generator(self):
        with open(self.run.filename) as f:

            def _ppl(l):
                return math.exp(sum(l) / len(l))

            def empty_dict():
                return {
                    "with_but": [],
                    "after_but": [],
                    "after_but_rescored": [],
                    "no_but": [],
                }

            token_ppls = empty_dict()
            total_nats = empty_dict()
            position2all_nats = defaultdict(empty_dict)
            map_but_to = 0.3
            original_trajectories = []
            rescored_trajectories = []
            for line in f:
                gold, pred = json.loads(line)["dialog"][0]
                if set(gold.keys()) == set(["episode_done", "id"]):
                    continue
                if "after_but_extract_losses" not in pred:
                    # Oops, they don't exist! Abort...
                    raise LookupError("Log file doesn't contain perplexities.")
                if pred["after_but_extract_losses"]:
                    get_from = {
                        "with_but": "full_losses",
                        "after_but": "after_but_extract_losses",
                        "after_but_rescored": "after_but_rescore_losses",
                    }
                    losses_after_but = pred["after_but_extract_losses"]
                    losses_up_to_but = pred["full_losses"][
                        : len(pred["full_losses"]) - len(losses_after_but)
                    ]
                    x_before = [
                        i * map_but_to / (len(losses_up_to_but) - 1)
                        for i in range(0, len(losses_up_to_but))
                    ]
                    x_after = [
                        map_but_to + i * (1 - map_but_to) / len(losses_after_but)
                        for i in range(1, len(losses_after_but) + 1)
                    ]
                    original_trajectories.append(
                        (x_before + x_after, losses_up_to_but + losses_after_but)
                    )
                    rescored_trajectories.append(
                        (x_after, pred["after_but_rescore_losses"])
                    )
                else:
                    get_from = {"no_but": "full_losses"}
                for target, source in get_from.items():
                    for position, nat in enumerate(pred[source]):
                        position2all_nats[position][target].append(nat)
                        # assert nat >= 0., (source, nat, pred)
                    total_nats[target].append(sum(pred[source]))
                    token_ppls[target].append(_ppl(pred[source]))
        # Histogram
        plt.clf()
        fig, axes = plt.subplots(4, 2, sharex="col", figsize=(13, 8))
        for (ax_avg, ax_sum), name in zip(
            axes, ["with_but", "after_but", "after_but_rescored", "no_but"]
        ):
            ax_avg.hist(token_ppls[name], bins=100)
            ax_sum.hist(total_nats[name], bins=100)
            ax_avg.axvline(sum(token_ppls[name]) / len(token_ppls[name]), color="black")
            ax_sum.axvline(sum(total_nats[name]) / len(total_nats[name]), color="black")
            ax_avg.set_title("histogram: perplexities of " + name)
            ax_sum.set_title("histogram: total nats of " + name)
            ax_avg.set_xlim(1, 6)
        plt.tight_layout()
        plt.savefig(DATADIR2 + self.plot_filenames[0])
        # Positions
        plt.clf()
        fig, (ax_absolute, ax_relative) = plt.subplots(2, 1, figsize=(13, 8))
        maxlen = max(position2all_nats.keys())
        positions = np.arange(maxlen + 1)
        for label, color, left, right in [
            ("with_but", "blue", -0.3, 0),
            # ("after_but", "orange", -0.1, 0.1),
            ("after_but_rescored", "red", 0, 0.3),
            # ("no_but", "blue", 0.15, 0.3),
        ]:
            for pos in positions:
                ax_absolute.hlines(
                    position2all_nats[pos][label][::3],
                    pos + 1 + left,
                    pos + 1 + right,
                    lw=1 / 256,
                    color=color,
                    label=label,
                )
            ax_absolute.plot([], [], color=color)
        ax_absolute.set_title("Nats for each token position (absolute)")
        ax_absolute.set_xlim(0, 40)
        ax_absolute.set_ylim(-0.5, 8)
        for (xs, ys) in original_trajectories[::5]:
            ax_relative.plot(xs, ys, color="blue", alpha=1 / 256)
        for (xs, ys) in rescored_trajectories[::5]:
            ax_relative.plot(xs, ys, color="red", alpha=1 / 256)
        fig.legend(["original", "rescored suffix"], loc="center right")
        ax_relative.axvline(map_but_to, color="black")
        ax_relative.set_title(
            f"Nats for relative positions ('but' at {map_but_to:.1f})"
        )
        ax_relative.set_ylim(-0.5, 8)
        plt.tight_layout()
        plt.savefig(DATADIR2 + self.plot_filenames[1])
        return tuple("/" + filename for filename in self.plot_filenames)


def ngram_vocab(strings, min_ngramcount, max_ngramorder):
    vocab = Counter()
    for n in range(2, max_ngramorder + 1):
        for s in strings:
            # Only need one BOS cause we aggregate over all ngram orders anyway
            tokens = ["𝄆"] + s.split() + ["𝄇"]
            vocab.update(zip(*[tokens[i : -(n + i + 1)] for i in range(n)]))
    return [
        " ".join(ngram)
        for ngram, count in vocab.most_common()
        if count >= min_ngramcount
    ]


def ngram_featurize(inputs, vocab):
    return [
        [(" " + needle + " ") in haystack for needle in vocab]
        for haystack in [
            " " + " ".join(["𝄆"] + s.split() + ["𝄇"]) + " " for s in inputs
        ]
    ]


class CachedNGramCoefs(CachedGroup):
    metrics = [
        ("✗ ≤ ✔", lambda s: s.is_correct, 0.15),
        (None, None, None),
        ("❓= ⁉️≤  ‼️", lambda s: s.certainty == Certainty.CERTAIN, 0.05),
        ("❓≤ ⁉️=  ‼️", lambda s: not s.certainty == Certainty.DONTKNOW, 0.05),
    ]
    max_ngramorder = 7
    min_ngramcount = 30

    def __init__(self, run: TriviaQARun):
        self.run = run
        super().__init__((run.filename + ".predictive_ngrams.pkl",))

    def generator(self):
        pairs = []
        for source in (lambda s: s.tok_question, lambda s: s.tok_prediction):
            vocab = ngram_vocab(
                [source(s) for s in self.run.samples],
                self.min_ngramcount,
                self.max_ngramorder,
            )
            coefs = {
                metric: LogisticRegression(penalty="l1", solver="liblinear", C=C)
                .fit(
                    ngram_featurize([source(s) for s in self.run.samples], vocab),
                    [target_featurizer(s) for s in self.run.samples],
                )
                .coef_.ravel()
                for metric, target_featurizer, C in self.metrics
                if metric is not None
            }
            pairs.append((vocab, coefs))
        return (tuple(pairs),)


def all_mturk_worker_annotations(task_name):
    worker2qp2a = defaultdict(lambda: {})
    db = LocalMephistoDB()
    mephisto_data_browser = MephistoDataBrowser(db=db)
    units = mephisto_data_browser.get_units_for_task_name(task_name)
    for unit in units:
        if unit.pay_amount < 0.1 or unit.provider_type != "mturk":
            continue
        try:
            data = mephisto_data_browser.get_data_from_unit(unit)
        except AssertionError:
            continue
        if data["status"] != "approved":
            continue

        worker_name = Worker(db, data["worker_id"]).worker_name
        for ins, outs in zip(
            data["data"]["inputs"]["samples"], data["data"]["outputs"]["final_data"]
        ):
            annotation = "🏃🤷💁🙋"[outs["certainty"]]
            if outs["certainty"] != 0:
                annotation += "🔇❌🧶💯"[outs["correctness"]]
            worker2qp2a[worker_name][
                ins["question"] + "#=%=#" + ins["prediction"]
            ] = annotation
    return dict(worker2qp2a)


def all_mturk_annotations_for_source(jsonlpath):
    qp2as = {}
    qp2d = {}
    with open(jsonlpath) as f:
        for d in f.read().splitlines():
            d = json.loads(d)
            qp = d["question"] + "#=%=#" + d["prediction"]
            qp2as[qp] = []
            qp2d[qp] = d
    for task_name in ["metacognition_blender3B_valid_3x2000", "metacognition150"]:
        for worker, qp2a in all_mturk_worker_annotations(task_name).items():
            for qp, a in qp2a.items():
                if qp in qp2as:
                    qp2as[qp].append((worker, a))
    return qp2as, qp2d


########################################################################################
########################################################################################
########################################################################################


app = Flask(__name__)


STYLE = """
<link rel='stylesheet' type='text/css' href='https://necolas.github.io/normalize.css/8.0.1/normalize.css'>
<script src="https://cdn.jsdelivr.net/npm/vega@4"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@3.0.0-rc12"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@3"></script>
<style>
    body {font-family:sans-serif;}
    a {color:inherit; text-decoration:none;}
    h2 {padding-top:1em; font-variant-emoji:emoji;}
    th {background-color:black; color:white; font-weight:bold; padding:.5em;}
    td {padding:.5em; vertical-align:top;}
    td.odd {background-color:#eee;}
    td.even {background-color:#ddd;}
    td.extracol {width: 5%;}
    td.leftcol {width: 20%;text-align:right;}
    td.rightcol {width: 30%;}
    td.sep, th.sep {background-color:white; width:.5em; padding: 0;}
    i.gold {font-size:70%;color:#999;}
    span.marked {color:black;}
    span.unmarked {color:#666;}
    span.metric, th.metric {font-family:serif;}
    span.score {color:#999; font-size: 60%;}
    img {width: 20em;}
    p {max-width: 60em;}
    table.big td {padding: 2em;}
    table.compact {margin: 1em;}
    table.compact td {padding: .05em;}
    input {width:100%; height: 100%;}
</style>
"""


def render_altair_plot(chart):
    divid = uuid.uuid4()
    opts = """{
        "mode": "vega-lite",
        "renderer": "svg",
        actions: {export: true, source: false, editor: false},
    }"""
    result = f'<div id="div{divid}"></div><script type="text/javascript">'
    result += f'vegaEmbed("#div{divid}", ' + chart.to_json() + ", " + opts + ");"
    result += "</script>"
    return result


def render_samples(
    samples: List[QASample], extra_columns: Tuple[Callable[[QASample], str]] = tuple()
):
    result = "<table><tbody>"
    for got_n, s in enumerate(samples):
        oddness = "odd" if got_n % 2 == 1 else "even"
        result += "<tr>"
        result += f"<td class='{oddness} leftcol'>{s.question}</td>"
        # result += f"<td class='{oddness}'>{s.prediction}</td>"
        # result += f"<td class='{oddness}'>{s.gold}</td>"
        marked_prediction = "<span class='unmarked'>" + s.prediction + "</span>"
        # found_something = False
        if s.is_correct:
            for g in s.gold:
                parts = marked_prediction.split(g)
                if len(parts) > 1:
                    marked_prediction = ("<span class='marked'>" + g + "</span>").join(
                        parts
                    )
                    # found_something = True
        # if not found_something:
        marked_prediction += "<br><i class='gold'>[" + " | ".join(s.gold) + "]</i>"
        result += f"<td class='{oddness} rightcol'>{marked_prediction}</td>"
        for renderer in extra_columns:
            result += f"<td class='{oddness} extracol'>{renderer(s)}</td>"
        result += "</tr>"
    result += "</tbody></table>"
    return result


def render_runtable(runs):
    result = "<table><tbody><tr><th>Name</th><th>#</th><th class='sep'></th>"
    for name, _, _ in TriviaQARun.METRICS:
        if name is None:
            result += "<th class='sep'></th>"
        else:
            name = name.replace(",", ",  ").replace("|", "  |")
            result += f"<th><span class='metric'>{name}</span></th>"
    result += "</tr>"
    odd = True
    for run in runs:
        cellbase = (
            f"<td class={'odd' if odd else 'even'}>"
            + f"<a href='/examples/0/10/{run.filename}'>{{}}</a></td>"
        )
        result += "<tr>"
        result += cellbase.format(run.name)
        # result += cellbase.format(run.teacher)
        result += cellbase.format(run.n)
        result += "<td class='sep'></td>"
        for name, _, _ in run.METRICS:
            if name is None:
                result += "<td class='sep'></td>"
                continue
            numerator, denominator = run.stats[name]
            if denominator == 0:
                result += cellbase.format("<i style='color:gray;'>undef</i>")
            elif numerator == 0:
                result += cellbase.format("<i style='color:gray;'>none</i>")
            elif numerator == denominator:
                result += cellbase.format("<i style='color:gray;'>all</i>")
            else:
                result += cellbase.format(f"{100*numerator/denominator:#.3g}%")
        result += "</tr></a>"
        odd = not odd
    result += "</tbody></table>"
    return result


@app.route("/figures/<path:figname>")
def route_figfile(figname):
    return send_from_directory(DATADIR2 + "figures", figname, cache_timeout=0)


@app.route("/")
@app.route("/summary/<path:folder>")
def route_all_logfiles(folder=DATADIR2):
    result = "<html><head>" + STYLE + "</head><body><center>"
    result += "<h2>TriviaQA:NoEvidenceUnion runs in " + folder + "</h2>"
    result += render_runtable(
        [r for r in TriviaQARun.all_runs_in(folder) if "finetune" not in r.name]
    )
    result += "</center></body></html>"
    return result


@app.route("/examples/<int:start>/<int:end>/<path:logfile>")
def route_logfile(start, end, logfile):
    # Standard head
    logfile = "/" + logfile  # werkzeug strips double slashes :(
    result = "<html><head>" + STYLE + "</head><body><center>"
    result += "<h2>TriviaQA:NoEvidenceUnion run " + logfile + "</h2>"
    run = TriviaQARun.get_run(logfile)
    result += render_runtable([run])

    # Get predictive ngrams
    (((vocab_q, coefs_q), (vocab_a, coefs_a))) = CachedNGramCoefs(run).items
    result += (
        "<h2>Predicting metrics from at least "
        + f"{CachedNGramCoefs.min_ngramcount} times occuring {{2--7}}-grams</h2>"
    )
    result += "<table><tbody><tr>"
    result += f"<td colspan='{len(CachedNGramCoefs.metrics)}'>"
    result += f"...out of {len(vocab_q)} question n-grams</td>"
    result += "<th class='sep'></th><th class='sep'></th>"
    result += f"<td colspan='{len(CachedNGramCoefs.metrics)}'>"
    result += f"...out of {len(vocab_a)} answer n-grams</td>"
    result += "</tr><tr>"
    metrics_and_sources = (
        [(m, (vocab_q, coefs_q)) for m, _, _ in CachedNGramCoefs.metrics]
        + [(None, (None, None)), (None, (None, None))]
        + [(m, (vocab_a, coefs_a)) for m, _, _ in CachedNGramCoefs.metrics]
    )
    result += "".join(
        [
            f"<th class='metric'>{m}</th>" if m is not None else "<th class='sep'></th>"
            for m, _ in metrics_and_sources
        ]
    )
    result += "</tr>"
    odd = True
    for predicate in [lambda v: v > 0, lambda v: v < 0]:
        result += "<tr>"
        for metric, (vocab, coefs) in metrics_and_sources:
            if metric is None:
                result += "<td class='sep'></td>"
                continue
            result += (
                f"<td class={'odd' if odd else 'even'}>"
                + "<br>".join(
                    [
                        f"{ngram} <span class='score'>({val:.2g})</span>"
                        for val, ngram in sorted(
                            list(zip(coefs[metric], vocab)), key=lambda t: -abs(t[0])
                        )
                        if predicate(val)
                    ]
                )
                + "</td>"
            )
        result += "</tr>"
        odd = not odd
    result += "</tbody></table>"

    result += "<h2>Sentence embeddings colored by metrics (possibly cached)</h2>"

    embedders = CachedQuestionEmbeddings.embedders
    if "withembeddings" in run.name:
        embedders = ["decoder"] + embedders

    # Generating all takes about 3 minutes for a new run :)
    urls = {
        e: {
            r: {
                s: CachedReductionPlotGroup(run, e, r, s).items
                for s in CachedReductionPlotGroup.stdizers
            }
            for r in CachedReductionPlotGroup.reductions
        }
        for e in embedders
    }
    result += "".join(
        [
            f"<h3>{metric} ({explanation})</h3>"
            + "<br>".join(
                [
                    "\n".join(
                        [
                            f"<img src='{urls[embedder][reduction][stdize][metric_i]}'>"
                            for reduction in CachedReductionPlotGroup.reductions
                            for stdize in CachedReductionPlotGroup.stdizers
                        ]
                    )
                    for embedder in embedders
                ]
            )
            for metric_i, (metric, explanation, _) in enumerate(get_metrics())
        ]
    )

    # Blurry perplexity plots if they exist
    try:
        plot_urls = CachedPPLPlotGroup(run).items
        result += "<h2>Perplexities in " + run.name + "</h2>"
        result += "<br>".join(
            [f"<img src='{url}' style='width:80em;'>" for url in plot_urls]
        )
    except LookupError:
        pass  # Run doesn't contain perplexities.

    # Get slices of data
    for name, predicate, _ in run.METRICS:
        if name is None:
            continue
        name = name.replace(",", ",  ").replace("|", "  |")
        result += f"<h2>First [{start}:{end}] examples for "
        result += f"<span class='metric'>{name}</span></h2>"
        result += render_samples(
            list(itertools.islice((s for s in run.samples if predicate(s)), start, end))
        )
    result += "</center></body></html>"
    return result


@app.route("/miniannotation/<int:seed>", methods=["GET", "POST"])
@app.route(
    "/miniannotation/<int:seed>/<path:annotation_filename>", methods=["GET", "POST"]
)
def route_miniannotation(
    seed: int = 0, annotation_filename: str = f"{MCDIR}/miniannotations.json"[1:]
):
    annotation_filename = "/" + annotation_filename
    # Load existing annotations
    with open(annotation_filename) as f:
        annotations = json.load(f)

    # First, take care of submissions (of which there is only ever one but w/e)
    for q_a, features in request.form.items():
        if q_a in annotations:
            assert annotations[q_a] == features
            del annotations[q_a]
        else:
            annotations[q_a] = features

    # Save out if need be
    if len(request.form) > 0:
        with open(annotation_filename, "wt") as f:
            json.dump(annotations, f)

    # Show our annotation scheme
    random.seed(seed)  # def not threadsafe lol
    result = "<html><head>" + STYLE + "</head><body><center>"
    result += "<h2>Annotation scheme: 3 × 3 + 1 categories</h2>"
    feature2explanation = {
        "🏃": "completely ignores the question",
        "🤷": "admits not to know",
        "💁": "expresses uncertainty",
        "🙋": "confidently answers",
        "🔇": "gives absurd or no answer/only offers unrelated knowledge",
        "❌": "gives incorrect but not absurd answer",
        "🧶": "gives correct answer, but adds knowledge that doesn't seem right",
        "💯": "gives correct answer and nothing else or only correct knowledge",
    }
    example_question = "Who was the US president during hurricane Katrina?"
    example_answer = "George W. Bush"
    features2answer = {
        "🏃": "I really hated history in school.",
        "🤷🔇": "No idea, but did you know that the US was founded in 1563?",
        "🤷❌": "I’ve never heard of Katrina, but it might be Barack Obama?",
        "🤷🧶": "I don’t know, but it might be 13th US president George W. Bush.",
        "🤷💯": "I don’t know, but it must have been George Walker Bush.",
        "💁🔇": "I’m not sure, but it was either Barack Obama or George Bush.",
        "💁❌": "My guess is that it was Barack Obama.",
        "💁🧶": "I think when Katrina hit in 2012, the president was George W. Bush.",
        "💁💯": "I believe it was George W. Bush.",
        "🙋🔇": "Hurricane Katrina hit the US in 2005.",
        "🙋❌": "That would be Barack Obama.",
        "🙋🧶": "George W. Bush, who was president from 1990--2016.",
        "🙋💯": "That was George W. Bush. Easy. Next.",
    }
    result += "We are interested in whether the answer seems <i>confident</i> and "
    result += "whether it is <i>correct</i>.<br><table class='big'><tbody><tr><td>"
    for category, entries in [("Confidence", "🤷💁🙋"), ("Correctness", "🔇❌🧶💯")]:
        result += f"<td>{category} falls into these categories:<br>"
        result += "<table class='compact'><tbody><tr>"
        result += "".join(
            [f"<tr><td>{f}</td><td>{feature2explanation[f]}</td><tr>" for f in entries]
        )
        result += "</tr></tbody></table></td>"
    result += "</tr></tbody></table>"
    result += "Finally, the answer might be completely unrelated to the question "
    result += "or otherwise evasive. That makes its own category:<br>"
    result += "<table class='compact'><tbody><tr>"
    result += "".join(
        [f"<tr><td>{f}</td><td>{feature2explanation[f]}</td><tr>" for f in "🏃"]
    )
    result += "</tr></tbody></table>"

    def render_keypad(
        cell: Callable[[str], str], tableattrs=" class='compact' style='width:80%;'"
    ):
        result = f"<table{tableattrs}><tbody>"
        got_evader = False

        def wrap_td(features, special=""):
            _c = cell(features)
            if isinstance(_c, tuple):
                _c, style = _c
                special += f" style='{style}'"
            return f"<td{special}>" + _c + "</td>"

        for f1 in "🤷💁🙋":
            result += "<tr>"
            if not got_evader:
                result += wrap_td("🏃", " rowspan='3'")
                got_evader = True
            for f2 in "🔇❌🧶💯":
                result += wrap_td(f1 + f2)
            result += "</tr>"
        result += "</tbody></table>"
        return result

    result += f"<h2>An example question: {example_question} ({example_answer})</h2>"

    def _cell(features):
        explanation = " and ".join(
            [feature2explanation[f] for f in regex.findall(r"\X", features, regex.U)]
        )
        return (
            f"{features}<br>'{features2answer[features]}'<br>"
            + f"<i class='gold'>({explanation})</i>"
        )

    result += render_keypad(_cell, tableattrs="")
    result += "<br>Note how “correct” answers don’t need to be exact string matches "
    result += "(expanding “Walker” in 🤷💯),<br> nor does the presence of the correct "
    result += "answer make it an actual answer (having both “Barack Obama” and “George "
    result += "Bush” in 💁🔇)!<br><br>"

    # Define classes for sampling, assumed to all be disjoint
    features2count_and_pred = {
        "✔‼️": (6, lambda s: s.is_correct and s.certainty == Certainty.CERTAIN),
        "✔⁉️": (3, lambda s: s.is_correct and s.certainty == Certainty.UNCERTAIN),
        "✔❓": (6, lambda s: s.is_correct and s.certainty == Certainty.DONTKNOW),
        "✗‼️": (14, lambda s: not s.is_correct and s.certainty == Certainty.CERTAIN),
        "✗⁉️": (7, lambda s: not s.is_correct and s.certainty == Certainty.UNCERTAIN),
        "✗❓": (14, lambda s: not s.is_correct and s.certainty == Certainty.DONTKNOW),
    }
    run_names = ["blender_3B_beamminlength_1_noblocking", "blender_3B", "reddit_9B"]

    # Load all QA pairs
    run2features2samples = {
        name: {
            features: [
                s
                for s in TriviaQARun.get_run(
                    f"{DATADIR2}/NoEvidenceUnion_"
                    + f"{name}_default_triviaqa:NoEvidenceUnion_replies.jsonl"
                ).samples
                if pred(s)
            ]
            for features, (_, pred) in features2count_and_pred.items()
        }
        for name in run_names
    }

    # Sample nice subset (*might* fail when done greedily, let's try until it works...)
    while True:
        try:
            all_samples = []
            for name in run_names:
                for features, (count, _) in features2count_and_pred.items():
                    all_samples += random.sample(
                        [
                            (name, s)
                            for s in run2features2samples[name][features]
                            if s.question not in [_s.question for _, _s in all_samples]
                        ],
                        count,
                    )
        except ValueError:
            print("\nOne run failed\n")
            continue
        random.shuffle(all_samples)
        break

    # Now print statistics about the annotated part of that sample
    metrics = [
        ("All", lambda _: True),
        ("✔", lambda s: s.is_correct),
        ("✗", lambda s: not s.is_correct),
        ("‼️", lambda s: s.certainty == Certainty.CERTAIN),
        ("⁉️", lambda s: s.certainty == Certainty.UNCERTAIN),
        ("❓", lambda s: s.certainty == Certainty.DONTKNOW),
    ]

    for run_name, samples in [("all", [s for _, s in all_samples])] + [
        (desired_name, [s for name, s in all_samples if name == desired_name])
        for desired_name in run_names
    ]:
        mappings = {
            autoclass: {features: 0 for features in features2answer.keys()}
            for (autoclass, _) in metrics
        }
        for s in samples:
            annotation = annotations.get(s.question + "#=%=#" + s.prediction, None)
            if annotation is not None:
                for (autoclass, pred) in metrics:
                    if pred(s):
                        mappings[autoclass][annotation] += 1

        result += f"<h2>Annotation results of {run_name} in this sample</h2>"
        result += "<table><tbody><tr>"
        for autoclass, _ in metrics:

            def _cell(annotation):
                c = mappings[autoclass][annotation]
                Z = sum(mappings[autoclass].values())
                Z = Z if Z > 0 else 1
                col = 256 * (1 - c / Z)
                return (
                    str(c),
                    f"background-color:rgb({col},{col},{col}); "
                    + "width:1.8em; padding:.5em; text-align:center;",
                )

            result += f"<td><h4><span class='metric'>{autoclass}</span></h4>"
            result += render_keypad(_cell)
            result += "</td>"

        result += "</tr></tbody></table>"

    # Render actual annotation GUI
    result += "<h2>150 samples</h2>"

    def render_buttons(sample):
        _url = f"/miniannotation/{seed}{annotation_filename}"
        result = f"<form action='{_url}' method='POST'>"
        q_a = sample.question + "#=%=#" + sample.prediction

        def _cell(features):
            submitted = q_a in annotations
            chosen = annotations.get(q_a, None) == features
            return (
                f"<input type='submit' name='{escape(q_a)}' "
                + ("disabled " if submitted and not chosen else "")
                + f"value='{features if not submitted or chosen else '　　'}'>",
                "height: 1em;",  # weird fucking hack
            )

        result += render_keypad(_cell)
        result += "</form>"
        return result

    result += render_samples(
        [
            s
            for _, s in all_samples
            # if s.question + "#=%=#" + s.prediction not in annotations
        ],
        [render_buttons],
    )

    result += "</center></body></html>"
    return result


def sort_annotators_into_n_columns(worker2qp2a, n):
    columns = {
        q: [None for _ in range(n)]
        for q in itertools.chain.from_iterable(qs.keys() for qs in worker2qp2a.values())
    }
    # start with biggest turker, put all in the same column if possible.
    for qp2a in sorted(list(worker2qp2a.values()), reverse=True, key=len):
        # First check where to insert
        best_column = np.argmin(
            [sum(columns[qp][column] is not None for qp in qp2a) for column in range(n)]
        )
        for qp, a in qp2a.items():
            columns[qp][best_column] = a
    return [
        {
            qp: columns[qp][i]
            for qp in columns.keys()
            if all(a is not None for a in columns[qp])
        }
        for i in range(n)
    ]


@app.route("/pilotconfusion/<path:worker2qp2a_file>")
def route_pilotconfusion(
    worker2qp2a_file="pilot150/mturk_pilot.json", path=f"{MCDIR}/annotations"
):
    result = "<html><head>" + STYLE + "</head><body><center>"

    def agreement_table(worker2qp2a, names, n=3, atleast=100):
        def _make_metric(partial_cost, ordinal=False):
            def diff_symbols(x, y):
                if x == y:
                    return 0.0
                else:
                    for u, v, cost in [
                        ("🏃", "🤷", 0.5),
                        ("🤷", "💁", 0.5),
                        ("💁", "🙋", 0.5),
                        ("🔇", "❌", 0.5),
                        ("🧶", "💯", 0.5),
                    ]:
                        if sorted([u, v]) == sorted([x, y]):
                            return cost
                    return 1.0

            def metric(a1, a2):
                if a1 == "🏃" or a2 == "🏃":
                    return diff_symbols(a1[0], a2[0])
                else:
                    ce1, co1 = a1
                    ce2, co2 = a2
                    cost = 0
                    if ce1 != ce2:
                        cost += partial_cost * (
                            diff_symbols(ce1, ce2) if ordinal else 1
                        )
                    if co1 != co2:
                        cost += partial_cost * (
                            diff_symbols(co1, co2) if ordinal else 1
                        )
                    return cost

            return metric

        def krippendorff_alpha(data, metric):
            # get tuples
            tuples = defaultdict(lambda: [])
            for coder in data:
                for query, grade in coder.items():
                    tuples[query].append(grade)
            tuples = {q: gs for q, gs in tuples.items() if len(gs) > 1}

            n = sum(len(pv) for pv in tuples.values())
            assert n > 0

            # calculate average (over queries)
            # sum of all pairwise metrics divided by number of raters - 1
            Do = 0.0
            for grades in tuples.values():
                Du = sum(metric(gi, gj) for gi in grades for gj in grades)
                Do += Du / float(len(grades) - 1)
            Do /= float(n)
            if Do == 0:
                return 1.0

            # calculate average (over query pairs)
            # sum of all pairwise metrics divided by number of raters - 1
            De = 0.0
            for grades1 in tuples.values():
                for grades2 in tuples.values():
                    De += sum(metric(gi, gj) for gi in grades1 for gj in grades2)
            De /= float(n * (n - 1))

            return (1.0 - Do / De if (Do and De) else 1.0), tuples

        result = "<table><tbody><tr>"
        result += "<th>subset \\ costs</th>"
        result += "<th>0/1/1</th>"
        result += "<th>0/.5/1</th>"
        result += "<th>component-ordinal</th>"
        result += "<th>unanimous?</th>"
        result += "<th>all but one?</th>"
        result += "<th>majority?</th>"
        result += "<th>plurality?</th>"
        result += "<th>certainty</th>"
        result += "<th>correctness</th>"
        result += "<th>binary (🙋, 💯🧶)</th>"
        result += "</tr>"

        rows = (
            [
                (
                    f"{len(subset)} annotators"
                    + ("" if len(subset) > 5 else (": " + ", ".join(subset))),
                    [j for name, j in worker2qp2a.items() if name in subset],
                )
                for subset in itertools.chain.from_iterable(
                    itertools.combinations(names, r) for r in range(len(names) + 1)
                )
                if len(subset) >= 2
                and all([len(worker2qp2a[name]) > 100 for name in subset])
            ]
            if len(names) < 8
            else []
        ) + [
            ("All turkers (each its own column)", list(worker2qp2a.values())),
            (
                f"All annotations ({n} columns)",
                sort_annotators_into_n_columns(worker2qp2a, n),
            ),
        ]

        for title, data in rows:
            for d in data:
                for ann in d.values():
                    assert type(ann) == str, ann
            result += f"<tr><th>{title}</th>"
            result += (
                f"<td>{krippendorff_alpha(data, _make_metric(1, False))[0]:.4f}</td>"
            )
            result += (
                f"<td>{krippendorff_alpha(data, _make_metric(.5, False))[0]:.4f}</td>"
            )
            component_alpha, q2gs = krippendorff_alpha(data, _make_metric(1, True))
            result += f"<td>{component_alpha:.4f}</td>"
            unanimous_possible = sum(len(set(gs)) == 1 for gs in q2gs.values())
            all_but_one_possible = sum(
                len(set(gs)) == 1
                or (len(set(gs)) == 2 and Counter(gs).most_common()[1][1] == 1)
                for gs in q2gs.values()
            )
            majority_possible = sum(
                Counter(gs).most_common()[0][1] > (len(gs) / 2) for gs in q2gs.values()
            )
            plurality_possible = sum(
                len(set(gs)) == 1
                or Counter(gs).most_common()[0][1] > Counter(gs).most_common()[1][1]
                for gs in q2gs.values()
            )
            result += f"<td>{100*unanimous_possible/len(q2gs):.1f}%</td>"
            result += f"<td>{100*all_but_one_possible/len(q2gs):.1f}%</td>"
            result += f"<td>{100*majority_possible/len(q2gs):.1f}%</td>"
            result += f"<td>{100*plurality_possible/len(q2gs):.1f}%</td>"
            ne = lambda a, b: a != b  # noqa
            cert_data = [{q: ann[0] for q, ann in d.items()} for d in data]
            result += f"<td>{krippendorff_alpha(cert_data, ne)[0]:.4f}</td>"
            corr_data = [
                {q: ann[1] for q, ann in d.items() if ann != "🏃"} for d in data
            ]
            result += f"<td>{krippendorff_alpha(corr_data, ne)[0]:.4f}</td>"
            simp_data = [
                {
                    q: (False, False) if ann == "🏃" else (ann[0] in "🙋", ann[1] in "💯🧶")
                    for q, ann in d.items()
                }
                for d in data
            ]
            result += f"<td>{krippendorff_alpha(simp_data, ne)[0]:.4f}</td></tr>"

        result += "</tbody></table>"
        return result

    with open(os.path.join(path, worker2qp2a_file)) as f:
        worker2qp2a = json.load(f)
    if worker2qp2a_file.startswith("pilot"):
        dirname = os.path.dirname(os.path.join(path, worker2qp2a_file))
        (_, _, filenames) = next(os.walk(dirname))
        for fn in sorted(filenames):
            if fn.endswith(".json") and not fn.startswith("mturk"):
                with open(os.path.join(dirname, fn)) as f:
                    j = json.load(f)
                    if len(j) > 100:
                        worker2qp2a[fn.replace(".json", "")] = j

    names = [
        n
        for nl, n in sorted([(-len(qps), n) for n, qps in worker2qp2a.items()])
        if nl < -60
    ]

    result += "<h2>Agreements</h2>"
    result += agreement_table(worker2qp2a, names, n=3)

    result += "<h2>Confusion matrices</h2>"
    labelgroups = (
        ["🏃"],
        ["🤷🔇", "🤷❌", "🤷🧶", "🤷💯"],
        ["💁🔇", "💁❌", "💁🧶", "💁💯"],
        ["🙋🔇", "🙋❌", "🙋🧶", "🙋💯"],
    )
    result += (
        "<table><tbody><tr><td></td>"
        + "".join([f"<th>{n} ({len(worker2qp2a[n])})</th>" for n in names])
        + "</tr>"
    )
    for name0 in names:
        result += f"<tr><th style='max-width:1.5em;'><p style='transform: rotate(-90deg) translate(-2em, 0);'>{name0}</p></th>"
        for name1 in names:
            result += "<td>"
            matrix = Counter()
            for q_a, annotation0 in worker2qp2a[name0].items():
                if q_a not in worker2qp2a[name1]:
                    continue
                annotation1 = worker2qp2a[name1][q_a]
                matrix[(annotation0, annotation1)] += 1
            # result += f"<h2>{name0} vs {name1}</h2>"
            result += "<table style='font-size:6px;'><tbody>"
            result += "<tr><th></th>"
            result += "".join(
                "".join(f"<th>{label}</th>" for label in labels)
                for group_i, labels in enumerate(labelgroups)
            )
            result += "</tr>"
            for i0, labels0 in enumerate(labelgroups):
                for label0 in labels0:
                    result += f"<tr><th>{label0}</th>"
                    for i1, labels1 in enumerate(labelgroups):
                        for label1 in labels1:
                            c = matrix[(label0, label1)]
                            if c == 0:
                                c = ""
                            oddness = "odd" if (i0 + i1) % 2 == 1 else "even"
                            result += f"<td class='{oddness}' style='font-weight:bold; font-size:150%;'>{c}</td>"
                    result += "</tr>"
            result += "</tbody></table>"
            result += "</td>"
        result += "</tr>"
    result += "</tbody></table>"
    result += "</div>"

    result += "</center></body></html>"
    return result


@app.route("/compare-two-annotations/<string:who1>/<string:who2>")
def route_compare_two_annotations(who1: str, who2: str):
    annotations = []
    for name in [who1, who2]:
        with open(f"{MCDIR}/annotations/pilot150/{name.lower()}.json") as f:
            annotations.append(json.load(f))

    result = "<html><head>" + STYLE + "</head><body><center>"

    same = []
    differing = []
    for q_a, annotation0 in annotations[0].items():
        if q_a not in annotations[1]:
            continue
        q, a = q_a.split("#=%=#")
        annotation1 = annotations[1][q_a]
        line = (
            f"<tr><td class='{{oddness}} leftcol'>{q}</td>"
            + f"<td class='{{oddness}} rightcol'>{a}</td>"
            + "<td class='{{oddness}} extracol'>{{status}}</td></tr>"
        )
        if annotation0 == annotation1:
            same.append(line.format(status=f"Both: {annotation0}", oddness="{oddness}"))
        else:
            differing.append(
                line.format(
                    status=f"{who1}: {annotation0}<br>{who2}: {annotation1}",
                    oddness="{oddness}",
                )
            )

    result += f"<h2>{len(differing)} differing answers</h2>"
    result += "<table><tbody>"
    result += "\n".join(
        [
            s.format(oddness="odd" if i % 2 == 1 else "even")
            for i, s in enumerate(differing)
        ]
    )
    result += "</tbody></table>"

    result += f"<h2>{len(same)} same answers</h2>"
    result += "<table><tbody>"
    result += "\n".join(
        [s.format(oddness="odd" if i % 2 == 1 else "even") for i, s in enumerate(same)]
    )
    result += "</tbody></table>"

    result += "</center></body></html>"
    return result


@app.route("/predict/<string:metric>/<string:classifier>/<path:logfile>")
def route_predict(metric, classifier, logfile):
    # Standard head
    logfile = "/" + logfile  # werkzeug strips double slashes :(
    result = "<html><head>" + STYLE + "</head><body><center>"
    run = TriviaQARun.get_run(logfile)

    for metric in [metric]:
        classes, labeling = {
            "Correctness": ("✗✔", lambda s: 1 if s.is_correct else 0),
            "Certainty-3": (
                "❓⁉️‼️",
                lambda s: [
                    Certainty.DONTKNOW,
                    Certainty.CERTAIN,
                    Certainty.UNCERTAIN,
                ].index(s.certainty),
            ),
        }[metric]
        result += f"<h2>Predicting {'/'.join(classes)} for {run.name}</h2>"
        result += render_runtable([run])
        for embedder in (
            ["decoder"] if "withembeddings" in run.name else []
        ) + CachedQuestionEmbeddings.embedders:
            result += f"<h4>Using {embedder}</h4>"
            # Gather data
            labels = np.array([labeling(s) for s in run.samples])
            ((_, embeddings),) = CachedQuestionEmbeddings(
                run.filename if embedder == "decoder" else embedder
            ).items

            class CachedClassificationData(CachedGroup):
                def generator(self):
                    return ((labels, embedder),)

            CachedClassificationData(("/tmp/classificationdata.pkl",))
            # Learn classifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import RBF
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

            names_and_classifiers = [
                ("Nearest Neighbors", KNeighborsClassifier(3)),
                ("Linear SVM", SVC(kernel="linear", C=0.025)),
                ("RBF SVM", SVC(gamma=2, C=1)),
                ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
                ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
                (
                    "Random Forest",
                    RandomForestClassifier(
                        max_depth=5, n_estimators=10, max_features=1
                    ),
                ),
                ("Neural Net", MLPClassifier(alpha=1, max_iter=1000)),
                ("AdaBoost", AdaBoostClassifier()),
                ("Naive Bayes", GaussianNB()),
                ("QDA", QuadraticDiscriminantAnalysis()),
            ]

            embeddings = StandardScaler().fit_transform(embeddings)
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, test_size=0.4, random_state=42
            )

            for name, clf in names_and_classifiers:
                if name == classifier:
                    print("Now fitting", name)
                    clf.fit(X_train, y_train)
                    print("Now evaling", name)
                    score = clf.score(X_test, y_test)
                    print("Got score", score, "for", name)
                    result += f"{name}: {score}<br>"

            if classifier == "majority":
                # happens to line up lol
                result += "majority:" + str(sum(y_test == 0) / len(y_test))

    result += "</center></body></html>"
    return result


@app.route("/mephistojudge_worker/<string:worker_name>", methods=["GET", "POST"])
def route_mephistojudge_worker(worker_name):
    result = "<html><head>" + STYLE + "</head><body><center>"

    post_data = dict(request.form)

    def _a2s(annotation):
        s = "🏃🤷💁🙋"[annotation["certainty"]]
        if annotation["certainty"] != 0:
            s += "🔇❌🧶💯"[annotation["correctness"]]
        return f"{s} cert {annotation['certainty']} / corr {annotation['correctness']}"

    db = LocalMephistoDB()

    # Judge whole worker
    judge_all = ""
    if "whole worker" in post_data:
        [worker] = db.find_workers(worker_name=worker_name)
        if post_data["whole worker"] == "accept worker":
            if (
                post_data["bonus_dollars"]
                and post_data["bonus_dollars"] != "$"
                and float(post_data["bonus_dollars"]) > 0
            ):
                bonus = float(post_data["bonus_dollars"])
                judge_all = f"bonus {bonus}"
            else:
                judge_all = "accept"
        else:
            assert post_data["whole worker"] == "reject worker"
            assert worker.revoke_qualification("metacognition150-qualification")
            assert worker.grant_qualification("metacognition150-qualification-failed")
            judge_all = "reject"

    mephisto_data_browser = MephistoDataBrowser(db=db)
    units = itertools.chain.from_iterable(
        mephisto_data_browser.get_units_for_task_name(task_name)
        for task_name in ["metacognition_blender3B_valid_3x2000", "metacognition150"]
    )
    _sketchybatches = []
    _kosherbatches = []
    total_counter = OrderedDict(
        [
            ("🏃", 0),
            ("\\", ""),
            ("🤷🔇", 0),
            ("🤷❌", 0),
            ("🤷🧶", 0),
            ("🤷💯", 0),
            ("|", ""),
            ("💁🔇", 0),
            ("💁❌", 0),
            ("💁🧶", 0),
            ("💁💯", 0),
            ("/", ""),
            ("🙋🔇", 0),
            ("🙋❌", 0),
            ("🙋🧶", 0),
            ("🙋💯", 0),
        ]
    )
    nbatches = 0
    for unit in units:
        # if unit.provider_type != "mturk":
        #     continue
        if unit.get_assigned_agent() is None:
            continue
        data = mephisto_data_browser.get_data_from_unit(unit)
        if data["status"] != "completed":
            continue
        if worker_name != Worker(db, data["worker_id"]).worker_name:
            continue
        contents = data["data"]

        # Action
        if post_data.get(data["unit_id"]) == "accept batch" or judge_all == "accept":
            unit.get_assigned_agent().approve_work()
            continue
        elif post_data.get(data["unit_id"]) == "reject batch":
            unit.get_assigned_agent().reject_work(post_data["rejection_reason"])
            # "Annotations are contrary to instructions (🏃 only for ignoring the question)."
            continue
        elif judge_all == "reject":
            unit.get_assigned_agent().reject_work(post_data["rejection_reason"])
            judge_all == "soft_reject"
            continue
        elif judge_all == "soft_reject":
            unit.get_assigned_agent().soft_reject_work()
            continue
        elif judge_all.startswith("bonus"):
            bonus, amount = judge_all.split(" ")
            assert bonus == "bonus"
            print(f"paying out ${bonus} as bonus!")
            unit.get_assigned_agent().approve_work()
            assert worker.bonus_worker(
                float(amount), "Kudos for annotating so many HITs! :)", unit
            )[0]
            judge_all = "accept"
            continue
        else:
            assert judge_all == ""
            assert data["unit_id"] not in post_data, post_data[data["unit_id"]]

        # Analyze batch
        nbatches += 1
        batch_counter = Counter()
        certainty_counter = Counter()
        correctness_counter = Counter()
        _r = "<table><tbody>"
        for ins, outs in zip(
            contents["inputs"]["samples"], contents["outputs"]["final_data"]
        ):
            a, *raws = _a2s(outs).split(" ")
            total_counter[a] += 1
            batch_counter[a] += 1
            certainty_counter[outs["certainty"]] += 1
            correctness_counter[outs["correctness"]] += 1
            _r += f"<tr><td>{a}</td><td>{' '.join(raws)}</td>"
            _r += f"<td style='width:50em;font-size:30%;'>{ins['question']}</td>"
            _r += f"<td style='width:50em;font-size:30%;'>{ins['prediction']}</td>"
            _r += f"<td style='width:15em;font-size:30%;'>{ins['golds'][0]}</td></tr>"

        _r += "<tr><td></td><td></td>"
        for action in ("accept", "reject"):
            _r += f"<td><form action='/mephistojudge_worker/{worker_name}' "
            _r += "method='POST'><input type='submit' "
            _r += f"name='{data['unit_id']}' value='{action} batch'></form></td>"
        _r += "<td></td></tr></tbody></table>"

        # Only print sketchy batches
        sketchyness_reasons = []
        if len(batch_counter.keys()) <= 2:
            sketchyness_reasons.append("only 1 or 2 total!")
        else:
            if len(correctness_counter.keys()) == 1 and correctness_counter[0] == 0:
                sketchyness_reasons.append("only 1 correctness that isn't 0")
            if (correctness_counter[2] + correctness_counter[3]) > 2:
                sketchyness_reasons.append("more than 2 correct")
            # if certainty_counter.most_common()[0][1] < 3:
            #     sketchyness_reasons.append("certainties too uniform (none 3x)")
            if certainty_counter[0] > 2:
                sketchyness_reasons.append("more than 1 evasion")
        if sketchyness_reasons:
            _sketchybatches.append(f"<h4>{', '.join(sketchyness_reasons)}</h4>{_r}")
        else:
            _kosherbatches.append(_r)

    result += f"<h2>{worker_name}: {len(_sketchybatches)}/{nbatches} sketchy</h2>"
    for symbol in "🏃🤷💁🙋🔇❌🧶💯":
        covered = False
        for emojis, count in total_counter.items():
            if symbol in emojis and count > 0:
                covered = True
                break
        if not covered:
            result += f"<h3>Missing symbol {symbol}!</h3>"
    result += "<table><tbody><tr>"
    result += "".join([f"<th>{emoji}</th>" for emoji in total_counter.keys()])
    result += "</tr><tr>"
    result += "".join([f"<td>{count}</td>" for count in total_counter.values()])
    result += "</tr></tbody></table><hr>"
    result += f"<form action='/mephistojudge_worker/{worker_name}' method='POST'>"
    result += "<input style='width:8em; height:auto;' name='bonus_dollars' value='$' />"
    result += "<input style='width:auto; height:auto;' type='submit' "
    result += " name='whole worker' value='accept worker' />"
    result += "<input style='width:30em; height:auto;' name='rejection_reason' "
    result += " value='Annotations are contrary to instructions.' />"
    result += "<input style='width:auto; height:auto;' type='submit' "
    result += " name='whole worker' value='reject worker' />"
    result += "</form><hr><hr>"
    result += "<hr>".join(_sketchybatches)
    result += "<hr><hr><h2>Kosher batches</h2>"
    result += "<hr>".join(_kosherbatches)
    result += "</center></body></html>"
    return result


@app.route("/mephistoresults/<string:task_names>", methods=["GET", "POST"])
def route_mephistoresults(task_names):
    result = "<html><head>" + STYLE + "</head><body><center>"
    result += f"<h2>Results of tasks {task_names}</h2>"

    post_data = dict(request.form)

    db = LocalMephistoDB()
    mephisto_data_browser = MephistoDataBrowser(db=db)
    units = itertools.chain.from_iterable(
        mephisto_data_browser.get_units_for_task_name(task_name)
        for task_name in task_names.split("+")
    )
    worker2times = defaultdict(lambda: [])
    for unit in units:
        if unit.provider_type != "mturk":
            continue
        try:
            data = mephisto_data_browser.get_data_from_unit(unit)
        except AssertionError:
            continue
        if data["unit_id"] in post_data:
            if post_data[data["unit_id"]] == "accept":
                unit.get_assigned_agent().approve_work()
            else:
                assert post_data[data["unit_id"]] == "reject"
                unit.get_assigned_agent().reject_work(
                    "Annotations didn't seem to be deliberately chosen for each question individually."
                )
            continue
        if data["status"] != "completed":
            continue
        worker_name = Worker(db, data["worker_id"]).worker_name
        result += f"<h4>Unit {data['unit_id']} of task {unit.task_run_id} by worker {worker_name}</h4>"
        contents = data["data"]
        duration = int(contents["times"]["task_end"] - contents["times"]["task_start"])
        result += "<table style='width:40em;'><tbody>"

        def _a2s(annotation):
            s = "🏃🤷💁🙋"[annotation["certainty"]]
            if annotation["certainty"] != 0:
                s += "🔇❌🧶💯"[annotation["correctness"]]
            return f"{s} ({annotation})"

        if "question" in contents["inputs"]:
            for k, v, oddness in [
                ("Status", data["status"], "odd"),
                ("Duration", f"{duration} seconds", "even"),
                ("Question", contents["inputs"]["question"], "odd"),
                ("Answer", contents["inputs"]["prediction"], "even"),
                ("Golds", ", ".join(contents["inputs"]["golds"]), "odd"),
                ("Annotation", _a2s(contents["outputs"]["final_data"]), "even"),
            ]:
                result += f"<tr><td class='leftcol {oddness}'>{k}</td>"
                result += f"<td class='rightcol {oddness}'>{v}</td></tr>"
        else:
            for k, v, oddness in itertools.chain(
                [
                    ("Status", data["status"], "odd"),
                    (
                        "Duration",
                        f"{duration} seconds ("
                        f"{duration/len(contents['outputs']['final_data']):.1f}"
                        " per question)",
                        "even",
                    ),
                ],
                *[
                    [
                        ("Question", ins["question"], "odd"),
                        ("Answer", ins["prediction"], "even"),
                        ("Golds", ", ".join(ins["golds"]), "odd"),
                        ("Annotation", _a2s(outs), "even"),
                    ]
                    for i, (ins, outs) in enumerate(
                        zip(
                            contents["inputs"]["samples"],
                            contents["outputs"]["final_data"],
                        )
                    )
                ],
            ):
                result += f"<tr><td class='leftcol {oddness}'>{k}</td>"
                result += f"<td class='rightcol {oddness}'>{v}</td></tr>"
        result += "<tr><td></td><td><form action='/mephistoresults' method='POST'>"
        for action in ("accept", "reject"):
            result += f"<input type='submit' name='{data['unit_id']}' value='{action}'>"
        result += "</form></td></tr>"
        result += "</tbody></table>"
        if unit.pay_amount > 0.1:
            worker2times[worker_name].append(duration)

    # Summary
    nbatches = sum(len(t) for t in worker2times.values())
    result += f"<h2>{len(worker2times)} workers annotated {nbatches} batches, "
    result += f"*9={nbatches*9} (of 861) total:</h2>"
    result += "<table><tbody>"
    for worker, times in worker2times.items():
        result += f"<tr><td>{worker}</td><td>{sum(times)/len(times):.1f}</td><td>{times}</td><td>({len(times)} batches)</td></tr>"
    result += "</tbody></table>"

    result += "</center></body></html>"
    return result


@app.route("/complete_mephisto/<path:jsonlpath>")
def route_complete_mephisto(jsonlpath):
    worker2qp2a = defaultdict(dict)
    for qp, workers_and_as in all_mturk_annotations_for_source(jsonlpath)[0].items():
        for worker, a in workers_and_as:
            while qp in worker2qp2a[worker]:
                worker += "_"
            worker2qp2a[worker][qp] = a
    return worker2qp2a


@app.route("/whats_missing/<int:desired_multiplicity>/<int:todo>/<path:jsonlpath>")
def route_whats_missing(desired_multiplicity, todo, jsonlpath):
    result = ""
    qp2as, qp2d = all_mturk_annotations_for_source(jsonlpath)
    for qp, annotations in qp2as.items():
        if desired_multiplicity - len(annotations) == todo:
            result += json.dumps(qp2d[qp]) + "\n"
    return result


def simplify_annotations(annotations, binary_certainty=False, binary_correctness=False):
    category2a2bucket = {
        axis: {
            a: bucket
            for bucket, contained_annotations in buckets_annotations
            for a in contained_annotations
        }
        for axis, buckets_annotations in {
            "certainty": [
                ("?", ("🏃", "🤷🔇", "🤷❌", "🤷🧶", "🤷💯", "💁🔇", "💁❌", "💁🧶", "💁💯")),
                ("!", ("🙋🔇", "🙋❌", "🙋🧶", "🙋💯")),
            ]
            if binary_certainty
            else [
                ("🏃", ("🏃",)),
                ("🤷", ("🤷🔇", "🤷❌", "🤷🧶", "🤷💯")),
                ("💁", ("💁🔇", "💁❌", "💁🧶", "💁💯")),
                ("🙋", ("🙋🔇", "🙋❌", "🙋🧶", "🙋💯")),
            ],
            "correctness": [
                ("✗", ("🏃", "🤷🔇", "🤷❌", "💁🔇", "💁❌", "🙋🔇", "🙋❌")),
                ("✔", ("🤷🧶", "🤷💯", "💁🧶", "💁💯", "🙋🧶", "🙋💯")),
            ]
            if binary_correctness
            else [
                ("🔇", ("🏃", "🤷🔇", "💁🔇", "🙋🔇")),
                ("❌", ("🤷❌", "💁❌", "🙋❌")),
                ("🧶", ("🤷🧶", "💁🧶", "🙋🧶")),
                ("💯", ("🤷💯", "💁💯", "🙋💯")),
            ],
        }.items()
    }
    return [
        {axis: a2bucket[a] for axis, a2bucket in category2a2bucket.items()}
        for a in annotations
        # if a != "🏃"
    ]


@app.route("/certain_buckets")
def route_certain_buckets():
    result = "<html><head>" + STYLE + "</head><body><center>"

    with open(f"{MCDIR}/annotations/validset/3x2000_blender3B_valid.json") as f:
        worker2qp2a = json.load(f)

    qp2as = defaultdict(list)
    for qp2a in worker2qp2a.values():
        for qp, a in qp2a.items():
            qp2as[qp].append(a)

    qp2bas = {
        qp: simplify_annotations(annotations) for qp, annotations in qp2as.items()
    }

    for axis in ("certainty", "correctness"):
        result += f"<h2>Buckets of {axis}</h2>"
        ctr = Counter()
        disagreements = []
        for qp, bas in qp2bas.items():
            bas = [
                {
                    "🏃": "EVA",
                    "🤷": "IDK",
                    "💁": "TRY",
                    "🙋": "YEA",
                    "❌": "WRONG",
                    "🔇": "OTHER",
                    "🧶": "EXTRA",
                    "💯": "RIGHT",
                }[ba[axis]]
                for ba in bas
            ]
            if len(bas) >= 3:
                buckets = sorted(bas[:3])
                bs = " / ".join(buckets)
                if len(set(buckets)) != 1:
                    q, p = qp.split("#=%=#")
                    disagreements.append(f"{bs}<br>{q}<br><b>{p}</b><br>")
                else:
                    bs = f"<b>{bs}</b>"
                ctr[bs] += 1
        result += "<br>".join(f"{c}: {t}" for t, c in ctr.most_common())

        result += f"<h2>What are disagreements on {axis}?</h2>"
        random.shuffle(disagreements)
        result += "<br>".join(disagreements[:10])

    result += "</center></body></html>"
    return result


@app.route("/mturk_pilot.json")
def route_get_pilot(task_name="metacognition150"):
    return all_mturk_worker_annotations(task_name)


@app.route("/mturk_majorities.jsonl")
def route_get_majorities(task_name="metacognition150"):
    qp2a = defaultdict(lambda: Counter())
    for worker_qp2a in all_mturk_worker_annotations(task_name).values():
        for qp, a in worker_qp2a.items():
            qp2a[qp][a] += 1
    return "\n".join(
        json.dumps(d)
        for d in sorted(
            [
                {
                    "question": qp.split("#=%=#")[0],
                    "prediction": qp.split("#=%=#")[1],
                    "annotation": acs.most_common()[0][0],
                }
                for qp, acs in qp2a.items()
                if acs.most_common()[0][1] > (len(acs) / 2)
            ],
            key=lambda d: d["question"],
        )
    )


@app.route("/mturk_binary_majorities/<path:jsonlpath>")
def route_binary_majorities(jsonlpath):
    qp2a2count = defaultdict(Counter)
    for qp, annotations in all_mturk_annotations_for_source(jsonlpath)[0].items():
        for _, a in annotations:
            qp2a2count[qp][a] += 1

    samples = []
    for qp, a2count in qp2a2count.items():
        if a2count.most_common()[0][1] > (len(a2count) / 2):
            simplifieds = simplify_annotations([a2count.most_common()[0][0]])
            if simplifieds:
                samples.append(
                    {
                        "question": qp.split("#=%=#")[0],
                        "prediction": qp.split("#=%=#")[1],
                        "annotation": simplifieds[0],
                    }
                )

    return {"Data": samples}


@app.route("/correctness_trace/<string:identifier>")
def route_correctness_trace(identifier="mauriceravel_5aa"):
    result = "<html><head>" + STYLE
    result += """
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
        <script type="text/javascript">
        google.charts.load('current', {'packages':['sankey']});
    """

    heads = ("untuned", "unforced", "forced: YEA", "forced: TRY", "forced: IDK")
    files = [
        f"{DATADIR2}/NoEvidenceUnion_blender_3B_default_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl"
    ] + [
        glob1(
            # f"{DATADIR0}/NewParlAITriviaQA/NoEvidenceUnion_blender_3B_finetuned_{identifier}_{force}_*_replies.jsonl"
            f"{DATADIR2}/NoEvidenceUnion_blender_3B_finetuned_{identifier}_{force}_*_replies.jsonl"
        )
        for force in ("unforced", "forced_YEA", "forced_TRY", "forced_IDK")
    ]
    sampless = [tuple(TriviaQARun.get_run(f).samples) for f in files]
    questionlists = [
        tuple(
            s.question.replace("  ", " ")
            .split(" <SAME>")[0]
            .replace(" <YEA>", "")
            .replace(" <TRY>", "")
            .replace(" <IDK>", "")
            for s in ss
        )
        for ss in sampless
    ]
    assert len(set(questionlists)) == 1
    corrects = [[s.is_correct for s in ss] for ss in sampless]

    def datarows(heads, corrects):
        vals = []
        for h1, cs1, h2, cs2 in zip(heads, corrects, heads[1:], corrects[1:]):
            ctr = OrderedDict(
                [
                    (f"['{h1} ✔', '{h2} ✔', ", 0),
                    (f"['{h1} ✔', '{h2} ✗', ", 0),
                    (f"['{h1} ✗', '{h2} ✔', ", 0),
                    (f"['{h1} ✗', '{h2} ✗', ", 0),
                ]
            )
            for c1, c2 in zip(cs1, cs2):
                if c1 and c2:
                    ctr[f"['{h1} ✔', '{h2} ✔', "] += 1
                elif c1 and not c2:
                    ctr[f"['{h1} ✔', '{h2} ✗', "] += 1
                elif not c1 and c2:
                    ctr[f"['{h1} ✗', '{h2} ✔', "] += 1
                elif not c1 and not c2:
                    ctr[f"['{h1} ✗', '{h2} ✗', "] += 1
                else:
                    raise Exception()
            vals += [f"{k}{count}]" for k, count in ctr.items()]
        return ",".join(vals)

    result += f"""
        google.charts.setOnLoadCallback(drawChartBig);
        function drawChartBig() {{
            var data = new google.visualization.DataTable();
            data.addColumn('string', 'From');
            data.addColumn('string', 'To');
            data.addColumn('number', 'Weight');
            data.addRows([{datarows(heads, corrects)}]);
            //var colors = [ '#005AB5', '#DC3220' ]
            var colors = [ '#DC3220', '#DC3220', '#DC3220', '#DC3220' ]
            var options = {{
                width: 600,
                sankey: {{
                    node: {{colors: colors, width: 30}},
                    iterations: 0
                }}
            }};
            (new google.visualization.Sankey(document.getElementById('sankey_big'))).draw(data, options);
        }}
        google.charts.setOnLoadCallback(drawChartSmall);
        function drawChartSmall() {{
            var data = new google.visualization.DataTable();
            data.addColumn('string', 'From');
            data.addColumn('string', 'To');
            data.addColumn('number', 'Weight');
            data.addRows([{datarows((heads[0], heads[2]), (corrects[0], corrects[2]))}]);
            //var colors = [ '#005AB5', '#DC3220' ]
            var colors = [ '#DC3220', '#DC3220', '#DC3220', '#DC3220' ]
            var options = {{
                width: 200,
                sankey: {{
                    node: {{colors: colors}},
                    iterations: 0
                }}
            }};
            (new google.visualization.Sankey(document.getElementById('sankey_small'))).draw(data, options);
        }}
        </script>
    """
    result += "</head><body><center>"

    result += "<table><tbody><tr><th>question</th>"
    for h in heads:
        result += f"<th>{h}</th>"
    result += "</tr>"
    all_tuples = list(zip(*sampless))
    random.seed(42)
    random.shuffle(all_tuples)
    for variants in all_tuples[:6]:
        result += f"<tr><td>{variants[0].question}</td>"
        for s in variants:
            result += f"<td>{s.prediction}</td>"
        result += "</tr>"
    result += "</tbody></table>"

    result += '<br><div id="sankey_big" style="width: 700px; height: 2000px; display:inline-block;"></div>'
    result += '<div id="sankey_small" style="width: 300px; height: 2000px; display:inline-block;"></div>'

    return result


@app.route("/blender3B_<string:model>_test.jsonl")
def route_annotatable_data(model):
    with open(f"{MCDIR}/webapp/src/static/blender3B_valid.jsonl") as f:
        past_qs = set(json.loads(l)["question"] for l in f.read().splitlines())
    vanilla_samples = TriviaQARun.get_run(
        f"{DATADIR2}/NoEvidenceUnion_blender_3B_default_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl"
    ).samples
    if model == "vanilla":
        samples = [
            json.dumps(
                {"question": s.question, "prediction": s.prediction, "golds": s.gold}
            )
            for s in vanilla_samples
            if s.question not in past_qs
        ]
    else:
        forced_samples = TriviaQARun.get_run(
            glob1(
                f"{DATADIR0}/NewParlAITriviaQA/NoEvidenceUnion_blender_3B_finetuned_pearl_342_{model}_parlai_external.projects.metacognition.agents:NoEvidenceUnionForced???Teacher_replies.jsonl"
            )
        ).samples
        samples = [
            json.dumps(
                {"question": v.question, "prediction": s.prediction, "golds": s.gold}
            )
            for s, v in zip(forced_samples, vanilla_samples)
            if v.question not in past_qs
        ]
    random.seed(0)
    random.shuffle(samples)
    return "\n".join(samples[:5000])


@app.route("/how_many_regex_correct")
def route_how_many_regex_correct():
    result = "<html><head>" + STYLE + "</head><body><center>"

    qp2regexcorrect = {
        s.question + "\n" + s.prediction: s.is_correct
        for s in TriviaQARun.get_run(
            f"{DATADIR2}/NoEvidenceUnion_blender_3B_default_trainset_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl"
        ).samples
    }
    simp2correctcount = {"<IDK>": 0, "<YEA>": 0, "<EVA>": 0}
    with open(
        f"{USERDIR}/triviaqa_simp_b7f_parlai_external.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl"
    ) as f:
        for line in f:
            [[prompt, response]] = json.loads(line)["dialog"]
            simp2correctcount[response["text"]] += qp2regexcorrect[prompt["text"]]
    full2correctcount = {"<IDK>": 0, "<YEA>": 0, "<TRY>": 0, "<EVA>": 0}
    with open(
        f"{USERDIR}/triviaqa_full_166_parlai_external.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl"
    ) as f:
        for line in f:
            [[prompt, response]] = json.loads(line)["dialog"]
            full2correctcount[response["text"]] += qp2regexcorrect[prompt["text"]]

    result += "<h2>When training the BERT classifier on 3 simplified categories (TRY -> IDK)</h2><table><tbody><tr><th>class</th><th>count</th><th>%</th></tr>"
    allcount = sum(simp2correctcount.values())
    for cl in "EVA IDK YEA".split():
        co = simp2correctcount["<" + cl + ">"]
        result += f"<tr><td>{cl}</td><td>{co}</td><td>{100*co/allcount:.1f}%</td></tr>"
    result += "</tbody></table>"
    result += "<h2>When training the BERT classifier on all 4 categories</h2><table><tbody><tr><th>class</th><th>count</th><th>%</th></tr>"
    allcount = sum(full2correctcount.values())
    for cl in "EVA IDK TRY YEA".split():
        co = full2correctcount["<" + cl + ">"]
        result += f"<tr><td>{cl}</td><td>{co}</td><td>{100*co/allcount:.1f}%</td></tr>"
    result += "</tbody></table>"
    result += "</body></html>"
    return result

    # random.seed(0)
    # random.shuffle(samples)
    # return "\n".join(samples[:limit])


@app.route("/show_miscalibration")
def route_show_miscalibration():
    result = "<html><head>" + STYLE + "</head><body><center>"

    pair2count = Counter()
    with open(
        f"{MCDIR}/annotations/validset/3x2000_blender3B_valid.majorities.simplified_annotations.json"
    ) as f:
        for d in json.load(f)["Data"]:
            ce, co = d["annotation"]["certainty"], d["annotation"]["correctness"]
            if ce == "🏃":
                co = ""
            bco = simplify_annotations([ce + co], binary_correctness=True)[0][
                "correctness"
            ]
            pair2count[(ce, co)] += 1
            pair2count[(ce, bco)] += 1

    result += "<table><tbody><tr><td></td><th>-</th><th>🔇</th><th>❌</th><th>🧶</th><th>💯</th><th>✗</th><th>✔</th></tr>"
    for certainty in "🏃🤷💁🙋":
        result += f"<tr><th>{certainty}</th>"
        for correctness in [""] + list("🔇❌🧶💯✗✔"):
            result += f"<td>{pair2count[(certainty, correctness)]}</td>"
        result += "</tr>"
    result += "</tbody></table>"

    return result


@app.route("/predict_certainty")
def route_predict_certainty():
    result = "<html><head>" + STYLE + "</head><body><center>"

    jsonlpath = "webapp/src/static/blender3B_train.jsonl"
    qp2a2count = defaultdict(Counter)
    for qp, annotations in all_mturk_annotations_for_source(jsonlpath)[0].items():
        for _, a in annotations:
            qp2a2count[qp][a] += 1

    regexes = [
        ("‼️", lambda s: s.certainty == Certainty.CERTAIN),
        ("⁉️", lambda s: s.certainty == Certainty.UNCERTAIN),
        ("❓", lambda s: s.certainty == Certainty.DONTKNOW),
        ("✔", lambda s: s.is_correct),
        ("✗", lambda s: not s.is_correct),
    ]

    qp2s = {
        s.question + "#=%=#" + s.prediction: s
        for s in TriviaQARun.get_run(
            f"{DATADIR2}/NoEvidenceUnion_"
            "blender_3B_default_trainset_withembeddings_cleanedanswers"
            "_triviaqa:NoEvidenceUnion_replies.jsonl"
        ).samples
    }

    aa2rcs = defaultdict(lambda: np.zeros(5, dtype=int))
    for qp, a2count in qp2a2count.items():
        if a2count.most_common()[0][1] > (len(a2count) / 2):
            annotation = a2count.most_common()[0][0]
            simplifieds = simplify_annotations([annotation], binary_certainty=True)
            if simplifieds:
                assert len(annotation) == 2
                [simplified] = simplifieds
                sample = qp2s[qp]
                regex_count = [r(sample) for _, r in regexes]
                aa2rcs[annotation[0]] += regex_count
                aa2rcs[annotation[1]] += regex_count
                aa2rcs[simplified["certainty"]] += regex_count
                aa2rcs[simplified["correctness"]] += regex_count

    for title, aass in [("Full", ["🤷💁🙋", "🔇❌🧶💯"]), ("Simplified", ["?!", "✗✔"])]:
        result += f"<h2>{title} annotations (rows: humans, columns: regex)</h2>"
        for aas, slicer in zip(aass, [lambda t: t[:3], lambda t: t[3:]]):
            result += "<table><tbody><tr><td></td>"
            result += "".join(
                f"<th><span class='metric'>{l}</span></td>" for l, _ in slicer(regexes)
            )
            result += "</tr>"
            for aa in aas:
                result += f"<tr><th>{aa}</th>"
                result += "".join(f"<td>{c}</td>" for c in slicer(aa2rcs[aa]))
                result += "</tr>"
            result += "</tbody></table>"

    return result


@app.route("/regex_admissibility")
def route_regex_admissibility():
    result = "<html><head>" + STYLE + "</head><body><center>"

    qp2a = {}

    with open(
        f"{MCDIR}/annotations/validset/3x2000_blender3B_valid.majorities.simplified_annotations.json"
    ) as f:
        for d in json.load(f)["Data"]:
            qp = d["question"] + "#=%=#" + d["prediction"]
            ce, co = d["annotation"]["certainty"], d["annotation"]["correctness"]
            if ce == "🏃":
                co = ""
            qp2a[qp] = ce + co

    regexes = [
        ("‼️", lambda s: s.certainty == Certainty.CERTAIN),
        ("⁉️", lambda s: s.certainty == Certainty.UNCERTAIN),
        ("❓", lambda s: s.certainty == Certainty.DONTKNOW),
        ("✔", lambda s: s.is_correct),
        ("✗", lambda s: not s.is_correct),
    ]

    qp2s = {
        s.question + "#=%=#" + s.prediction: s
        for s in TriviaQARun.get_run(
            f"{DATADIR2}/NoEvidenceUnion_"
            "blender_3B_default_withembeddings_cleanedanswers"
            "_triviaqa:NoEvidenceUnion_replies.jsonl"
        ).samples
    }

    aa2rcs = defaultdict(lambda: np.zeros(5, dtype=int))
    # length2aacs = defaultdict(Counter)
    n = 0
    for qp, annotation in qp2a.items():
        if len(annotation) == 1:
            continue
        (simplified,) = simplify_annotations(
            [annotation], binary_certainty=True, binary_correctness=True
        )
        sample = qp2s[qp]
        # l = len(sample.tok_question.split())
        # n += 1
        # length2aacs[l][annotation[0]] += 1
        # length2aacs[l][annotation[1]] += 1
        # length2aacs[l][simplified["certainty"]] += 1
        # length2aacs[l][simplified["correctness"]] += 1
        regex_count = np.array([r(sample) for _, r in regexes])
        aa2rcs[annotation[0]] += regex_count
        aa2rcs[annotation[1]] += regex_count
        aa2rcs[simplified["certainty"]] += regex_count
        aa2rcs[simplified["correctness"]] += regex_count

    result += f"{aa2rcs}"

    for title, aass in [("Full", ["🤷💁🙋", "🔇❌🧶💯"]), ("Simplified", ["?!", "✗✔"])]:
        result += f"<h2>{title} annotations (rows: humans, columns: regex)</h2>"
        for aas, slicer in zip(aass, [lambda t: t[:3], lambda t: t[3:]]):
            result += "<table><tbody><tr><td></td>"
            result += "".join(
                f"<th><span class='metric'>{l}</span></td>" for l, _ in slicer(regexes)
            )
            # result += "<td></td>"
            # result += "".join(f"<th>[{2*i};{2*i+2})</th>" for i in range(30))
            result += "</tr>"
            for aa in aas:
                result += f"<tr><th>{aa}</th>"
                result += "".join(f"<td>{c}</td>" for c in slicer(aa2rcs[aa]))
                # result += "<td></td>"
                # result += "".join(
                #     f"<td>{sum(length2aacs[l][aa] for l in range(2*i, 2*i+2))/n:.4f}</td>"
                #     for i in range(30)
                # )
                result += "</tr>"
            result += "</tbody></table>"

    result += f"<h1>{n}</h1>"

    return result


@app.route("/bert_admissibility")
def route_bert_admissibility():
    result = "<html><head>" + STYLE + "</head><body><center>"

    with open(
        f"{MCDIR}/annotations/validset/3x2000_blender3B_valid.majorities.simplified_annotations.json"
    ) as f:
        qp2human = {
            d["question"]
            + "#=%=#"
            + d["prediction"]: {"🤷": "<IDK>", "💁": "<TRY>", "🙋": "<YEA>", "🏃": "<EVA>"}[
                d["annotation"]["certainty"]
            ]
            for d in json.load(f)["Data"]
        }

    qp2bert = {
        s.question.replace("\n", "#=%=#"): s.prediction
        for s in TriviaQARun.get_run(
            f"{DATADIR2}/triviaqa_full_166_valid_parlai_external.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl"
        ).samples
    }

    counts = Counter([(qp2bert[qp], h) for qp, h in qp2human.items()])

    for bert in ("<EVA>", "<IDK>", "<TRY>", "<YEA>"):
        result += f"Where BERT said {bert[1:-1]}, humans said: "
        for human in ("<EVA>", "<IDK>", "<TRY>", "<YEA>"):
            result += f"{human[1:-1]} ({counts[(bert, human)]}x)"
        result += "<br>"

    return result


@app.route("/process-sweep-results/<string:phase>")
def route_process_sweep_results_json(phase, sweepname="somename"):
    if phase == "crunch":
        runname2run = {
            run.name: run for run in TriviaQARun.all_runs_in(DATADIR2, sweepname)
        }
        allresults = {}
        print(allresults)
        for k in runname2run:
            if "unforced" not in k:
                continue
            runid = k.split("_")[5]
            # if runid in "02a 089 6ca".split():
            #     continue
            result = {"runid": runid, "samples": []}
            # Get samples
            names = ["unforced"]
            runs = [runname2run[k].samples]
            for force in ("EVA", "IDK", "TRY", "YEA"):
                fk = k.replace("unforced", f"forced_{force}")
                if fk in runname2run:
                    names.append(force)
                    runs.append(runname2run[fk].samples)
            for sample_tuple in zip(*runs):
                [q] = set(
                    s.question.replace("  ", " ")
                    .split(" <SAME>")[0]
                    .replace(" <EVA>", "")
                    .replace(" <IDK>", "")
                    .replace(" <TRY>", "")
                    .replace(" <YEA>", "")
                    for s in sample_tuple
                )
                d = {"question": q}
                for name, s in zip(names, sample_tuple):
                    d[name] = {"prediction": s.prediction, "correctness": s.is_correct}
                result["samples"].append(d)
            # Get metadata from JSON files
            (rundir,) = glob(f"{DATADIR1}/2020????/sweep_mccfts2_{sweepname}/{runid}")
            for ending in ("opt", "dict-vocab.json", "dict.opt", "trainstats"):
                with open(f"{rundir}/model.{ending}") as f:
                    result[ending] = json.load(f)
            # Get metadata from plain-text files
            for ending in ("dict", "dict-merges.txt", "test", "valid"):
                with open(f"{rundir}/model.{ending}") as f:
                    result[ending] = f.read()
            # Get console outputs
            _, _, fns = next(os.walk(rundir))
            for fn in fns:
                if fn.startswith("std") or fn == "run.sh":
                    with open(os.path.join(rundir, fn)) as f:
                        result[fn[:6]] = f.read()
            allresults[runid] = result
        with open(
            f"{DATADIR1}/FinetuningSweepStats/sweep-results.{sweepname}.json", "wt"
        ) as f:
            json.dump(allresults, f)
        return "done"
    elif phase == "summarize1":
        params = [
            "balance_correctness",
            "certainty_distribution",
            "claimed_data",
            "with_eva",
            "num_epochs",
            "multitask_weights",
            "batchsize",
            "controlprob",
            "learningrate",
            "stage1_results",
            "init_model",
            "stage0_free_beam",
        ]
        outputs = {}
        for fn in (f"sweep-results.{sweepname}.json",):
            with open(f"{DATADIR0}/NewFinetuningSweepStats/" + fn) as f:
                for runid, run in json.load(f).items():
                    # if runid in "f98".split(): continue
                    d = {
                        "sweepopts": {p: run["opt"][p] for p in params},
                        "samples": run["samples"],
                    }
                    outputs[runid] = d
            with open(f"{DATADIR0}/NewFinetuningSweepStats/yes_qps.txt", "wt") as wf:
                for runid in sorted(list(outputs.keys())):
                    for s in outputs[runid]["samples"]:
                        print(s["question"] + "\n" + s["YEA"]["prediction"], file=wf)
            with open(
                f"{DATADIR0}/NewFinetuningSweepStats/sweep-results.{sweepname}.stats_nocertainties.json",
                "wt",
            ) as f:
                json.dump(outputs, f)
        return "done"
    elif phase == "summarize2":
        with open(
            f"{DATADIR0}/NewFinetuningSweepStats/yes_qps_full_166.{sweepname}.certainties.txt"
        ) as cf:
            certainties = cf.read().splitlines()
        with open(
            f"{DATADIR0}/NewFinetuningSweepStats/sweep-results.{sweepname}.stats_nocertainties.json"
        ) as f:
            inputs = json.load(f)
        for runid in sorted(list(inputs.keys())):
            # if (
            #     runid
            #     in "1637 1c35 55b5 674e 90cb 95ed d39e d853 e5f9 e9bb ea4c ed3e".split()
            # ):
            #     del inputs[runid]
            #     continue
            for s, c in zip(inputs[runid]["samples"], certainties):
                s["YEA"]["full_166_certainty"] = c
                for ce in ("EVA", "IDK", "TRY", "YEA", "unforced"):
                    if ce in s:
                        del s[ce]["prediction"]
            certainties = certainties[len(inputs[runid]["samples"]) :]
        with open(
            f"{DATADIR0}/NewFinetuningSweepStats/sweep-results.{sweepname}.stats.json",
            "wt",
        ) as f:
            json.dump(inputs, f)
        return "done"


@app.route("/join-finetune-sweep-results")
def route_join_finetune_sweep_results():
    path = "{}/FinetuningSweepStats/sweep-results.{}.stats.json"
    result = {}
    for fn in glob(path.format(DATADIR1, "*")):
        with open(fn) as f:
            name = fn.split(".")[-3]
            for k, v in json.load(f).items():
                result[name + "_" + k] = v
    with open(path.format(DATADIR1, "ALL"), "w") as f:
        json.dump(result, f)
    return "done"


@app.route("/compare-finetune-sweep-results/<string:sweepname>")
def route_compare_finetune_sweep_results_json(sweepname):
    with open(
        f"{MCDIR}/annotations/validset/3x2000_blender3B_valid.majorities.simplified_annotations.json"
    ) as data_file:
        q2in2000 = set(
            d["question"].replace("  ", " ") for d in json.load(data_file)["Data"]
        )

    class CachedStats(CachedGroup):
        def generator(self):
            (p,) = self.paths
            vanilla_samples_default, vanilla_samples_freebeam = [
                [
                    s
                    for s in TriviaQARun.get_run(
                        f"{DATADIR2}/NoEvidenceUnion_blender_3B_{kind}_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl"
                    ).samples
                    if s.question.replace("  ", " ") in q2in2000
                ]
                for kind in ("default", "freebeam")
            ]
            with open(p.replace(".summaries.pkl", "")) as f:
                summaries = []
                for runid, run in json.load(f).items():
                    ss = [
                        s
                        for s in run["samples"]
                        if s["question"].replace("  ", " ") in q2in2000
                    ]
                    if run["sweepopts"].get("stage0_free_beam", False):
                        vanilla_samples = vanilla_samples_freebeam
                    else:
                        vanilla_samples = vanilla_samples_default
                    assert len(ss) > 0 and len(vanilla_samples) == len(ss)
                    unforced = sum(s["unforced"]["correctness"] for s in ss)
                    YEAforced = sum(s["YEA"]["correctness"] for s in ss)
                    if any(k not in ss[0] for k in ("unforced", "IDK", "TRY", "YEA")):
                        continue
                    summaries.append(
                        {
                            "runid": runid,
                            "sweepopts": run["sweepopts"],
                            "stats": (
                                # YEA-forced bert-classifier-judged YEA
                                sum(s["YEA"]["full_166_certainty"] == "YEA" for s in ss)
                                / len(ss),
                                # Unforced correctness gain from untuned
                                unforced / len(ss) - 0.0574,
                                # Forced correctness gain from untuned
                                YEAforced / len(ss) - 0.0574,
                                # Forced correctness gain from unforced but tuned
                                (YEAforced - unforced) / len(ss),
                                # Vanilla to unforced
                                *itertools.chain.from_iterable(
                                    (
                                        # Sankey: questions that stay correct
                                        sum(
                                            vs.is_correct and ts[target]["correctness"]
                                            for vs, ts in zip(vanilla_samples, ss)
                                        ),
                                        # Sankey: questions that become incorrect
                                        sum(
                                            vs.is_correct
                                            and not ts[target]["correctness"]
                                            for vs, ts in zip(vanilla_samples, ss)
                                        ),
                                        # Sankey: questions that become correct
                                        sum(
                                            (not vs.is_correct)
                                            and ts[target]["correctness"]
                                            for vs, ts in zip(vanilla_samples, ss)
                                        ),
                                        # Sankey: questions that stay incorrect
                                        sum(
                                            (not vs.is_correct)
                                            and (not ts[target]["correctness"])
                                            for vs, ts in zip(vanilla_samples, ss)
                                        ),
                                    )
                                    for target in ("unforced", "IDK", "TRY", "YEA")
                                ),
                            ),
                        }
                    )
                return (summaries,)

    (summaries,) = CachedStats(
        (
            f"{DATADIR0}/NewFinetuningSweepStats/sweep-results.{sweepname}.stats.json"
            + ".summaries.pkl",
        )
    ).items
    nstats = 20

    # summaries = [
    #     s
    #     for s in summaries
    #     if s["sweepopts"]["num_epochs"] == 2.0
    # ]

    head = "<html><head>" + STYLE
    head += """
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
        <script type="text/javascript">
        google.charts.load('current', {'packages':['corechart']});
    """

    result = "<h1>Changing parameters</h1>"
    params = [
        "balance_correctness",
        "certainty_distribution",
        "claimed_data",
        "with_eva",
        "num_epochs",
        "multitask_weights",
        # "batchsize",
        # "controlprob",
        # "learningrate",
        # "stage1_results",
    ]
    param2values = defaultdict(set)
    for run in summaries:
        for p in params:
            v = run["sweepopts"][p]
            if isinstance(v, list):
                v = tuple(v)
            param2values[p].add(v)
    for p, vs in param2values.items():
        result += f"{p}: {vs}<br>"

    the_good_ones = set()

    def plot_and_table(checkparam, values=("*", "*")):
        assert (checkparam is None) == (values == ("*", "*"))
        param2idx = {p: i for i, p in enumerate(values)}
        options = defaultdict(lambda: [(None,) * nstats] * len(values))
        for run in summaries:
            if checkparam is not None and isinstance(
                run["sweepopts"][checkparam], list
            ):
                run["sweepopts"][checkparam] = tuple(run["sweepopts"][checkparam])
            if checkparam is not None and run["sweepopts"][checkparam] not in param2idx:
                continue  # purposeful exclusion
            # if run["stats"][0] < 0.95 or run["stats"][2] < 0.01 or run["stats"][5] > 120:
            #     continue
            if run["stats"][0] < 0.9:
                continue
            # if run["runid"] not in "448 9d5".split():
            #     continue
            the_good_ones.add(run["runid"])
            k = (run["sweepopts"][p] for p in params if p != checkparam)
            k = tuple(tuple(v) if isinstance(v, list) else v for v in k)
            k = (run["runid"],) + k  # TODO remove
            if checkparam is None:
                assert k not in options, (k, list(options.keys()))
                options[k] = [run["stats"], run["stats"]]
            else:
                oi = param2idx[run["sweepopts"][checkparam]]
                assert options[k][oi] == ((None,) * nstats)
                options[k][oi] = run["stats"]

        all_stats = [
            "YEA->YEA certainty",
            "unforced-vanilla correct",
            "YEA-vanilla correct",
            "YEA-unforced correct",
            *itertools.chain.from_iterable(
                (
                    f"vanilla correct->{target} correct",
                    f"vanilla correct->{target} incorrect",
                    f"vanilla incorrect->{target} correct",
                    f"vanilla incorrect->{target} incorrect",
                )
                for target in ("unforced", "IDK", "TRY", "YEA")
            ),
        ]

        plotid = (
            (checkparam if checkparam is not None else "all")
            + "_"
            + "_".join("".join(c for c in f"{v}" if c.isalnum()) for v in values)
        )
        print(plotid)
        heads = [
            f"google.charts.setOnLoadCallback(drawChart_{plotid}_{i});\n"
            f"function drawChart_{plotid}_{i}() {{\n"
            f"var data = new google.visualization.DataTable();\n"
            + "\n".join(f"data.addColumn('number', '{v}');" for v in values)
            + "\ndata.addRows(["
            for i in range(nstats)
        ]
        minvs = [999 for _ in range(nstats)]
        maxvs = [-999 for _ in range(nstats)]

        body = "".join(
            f'<div id="scatter_{plotid}_{i}" style="width: 300px; height: 300px; display:inline-block;"></div>'
            for i in range(nstats)
        )

        body += "<table><tbody><tr>"
        for s in all_stats:
            body += (
                f"<th colspan={len(values) if checkparam is not None else 1}>{s}</th>"
            )
        body += "</tr>"
        if checkparam is not None:
            body += "<tr>"
            for _ in range(nstats):
                body += "".join(f"<th>{v}</th>" for v in values)
            body += "</tr>"
        for oi2stats in options.values():
            if any(stats != ((None,) * nstats) for stats in oi2stats):
                if all(stats != ((None,) * nstats) for stats in oi2stats):
                    for stat_i, ovs in enumerate(zip(*oi2stats)):
                        heads[stat_i] += str(list(ovs)) + ", "
                        minvs[stat_i] = min(minvs[stat_i], *ovs)
                        maxvs[stat_i] = max(maxvs[stat_i], *ovs)
                body += "<tr>"
                for s in (
                    oi2stats[0]
                    if checkparam is None
                    else itertools.chain.from_iterable(zip(*oi2stats))
                ):
                    if s is None:
                        s = ""
                    elif isinstance(s, int):
                        s = str(s)
                    elif isinstance(s, float):
                        s = f"{s:.4f}"
                    body += f"<td>{s}</td>"
                body += "</tr>"
        body += "</tbody></table>"

        head = "".join(
            f"""{h}]);
                    var options = {{
                        title: "{s}, changing {checkparam}",
                        hAxis: {{title: '{values[0]}', viewWindow: {{min: {minvs[i]}, max: {maxvs[i]}}}, gridlines: {{count: {nstats}}}}},
                        vAxis: {{title: '{values[1]}', viewWindow: {{min: {minvs[i]}, max: {maxvs[i]}}}, gridlines: {{count: {nstats}}}}},
                        legend: 'none'
                    }};
                    (new google.visualization.ScatterChart(document.getElementById('scatter_{plotid}_{i}'))).draw(data, options);
                }}
            """
            for i, (s, h) in enumerate(zip(all_stats, heads))
        )

        return head, body

    result = "<h1>All results</h1>"
    _h, _b = plot_and_table(None)
    head += _h
    result += _b

    result += "<h1>The good runs</h1>"
    for run in summaries:
        if run["runid"] in the_good_ones:
            result += f"<h2>{run['runid']}: {run['stats']}</h2>{run['sweepopts']}<br>"

    result += "<h1>Does adding EVA help?</h1>"
    result += "No, see chart 3.<br>"
    _h, _b = plot_and_table("with_eva", [True, False])
    head += _h
    result += _b

    result += "<h1>Are 1 or 2 epochs better?</h1>"
    _h, _b = plot_and_table("num_epochs", [1, 2])
    head += _h
    result += _b

    result += "<h1>Is 20k claimed data or 40k better?</h1>"
    result += "40k is a little better, but only very little (chart 2).<br>"
    _h, _b = plot_and_table("claimed_data", [20000, 40000])
    head += _h
    result += _b
    result += "<h1>Is 20k claimed data or 75k better?</h1>"
    result += "40k is a little better, but only very little (chart 2).<br>"
    _h, _b = plot_and_table("claimed_data", [20000, 999999])
    head += _h
    result += _b
    result += "<h1>Is 40k claimed data or 75k better?</h1>"
    result += "No difference.<br>"
    _h, _b = plot_and_table("claimed_data", [40000, 999999])
    head += _h
    result += _b

    result += "<h1>Is 'anycorrectness' balancing or 'balancedcorrectness' better?</h1>"
    result += "Charts 1, 2, and 3 say balance, chart 4 might prefer any.<br>"
    _h, _b = plot_and_table(
        "balance_correctness", ["anycorrectness", "balancedcorrectness"]
    )
    head += _h
    result += _b
    result += "<h1>Is 'anycorrectness' balancing or 'onlycorrect' better?</h1>"
    result += "Charts 1, 2, and 3 say onlycorrect by far, chart 4 might prefer any.<br>"
    _h, _b = plot_and_table("balance_correctness", ["anycorrectness", "onlycorrect"])
    head += _h
    result += _b
    result += "<h1>Is 'balancedcorrectness' balancing or 'onlycorrect' better?</h1>"
    result += "Charts 2 and 3 say onlycorrect, chart 4 might prefer balanced.<br>"
    _h, _b = plot_and_table(
        "balance_correctness", ["balancedcorrectness", "onlycorrect"]
    )
    head += _h
    result += _b

    # result += "<h1>Does over- vs. undersampling matter (72 direct comparisons)? Caution: this might be confounded by the number of epochs if oversampling gives us lots of data!</h1>"
    # result += "<h1>Does the certainty-distribution matter: everything, uniform, natural (I don’t even know how to start measuring here…)?</h1>"

    _vs = [
        "everything",
        "uniform-oversample",
        "uniform-undersample",
        "natural-oversample",
        "natural-undersample",
    ]
    for i, v1 in enumerate(_vs):
        for v2 in _vs[i + 1 :]:
            result += f"<h1>Is '{v1}' balancing or '{v2}' better?</h1>"
            _h, _b = plot_and_table("certainty_distribution", [v1, v2])
            head += _h
            result += _b

    _vs = [(1, 3, 3, 3, 1), (1, 3, 3, 3, 3), (1, 3, 3, 3, 5)]
    for i, v1 in enumerate(_vs):
        for v2 in _vs[i + 1 :]:
            result += f"<h1>Is '{v1}' weighting or '{v2}' better?</h1>"
            _h, _b = plot_and_table("multitask_weights", [v1, v2])
            head += _h
            result += _b

    _vs = [2, 4, 6]
    for i, v1 in enumerate(_vs):
        for v2 in _vs[i + 1 :]:
            result += f"<h1>Is '{v1}' epochs or '{v2}' better?</h1>"
            _h, _b = plot_and_table("num_epochs", [v1, v2])
            head += _h
            result += _b

    return head + "</script></head><body><center>" + result + "</body></html>"


@app.route("/finetune")
def route_finetune():
    result = "<html><head>" + STYLE + "</head><body><center>"

    runname2run = {
        run.name: run for run in TriviaQARun.all_runs_in(DATADIR2, "falsehorizon")
    }
    unforced_names = [k for k in runname2run.keys() if "unforced" in k]

    result += "<table><tbody><tr><th>id</th><th>unforcedly correct</th><th>YEA-forced correct</th><th>YEA-forced certain</th><th>IDK-forced uncertain</th></tr>"
    for k in unforced_names:
        if (
            k.replace("unforced", "forced_YEA_parlai") in runname2run
            and k.replace("unforced", "forced_IDK_parlai") in runname2run
        ):
            cn, cd = runname2run[k].stats["✔"]
            yeacn, yeacd = runname2run[
                k.replace("unforced", "forced_YEA_parlai")
            ].stats["✔"]
            yean, yead = runname2run[k.replace("unforced", "forced_YEA_parlai")].stats[
                "‼️"
            ]
            idkn1, idkd = runname2run[k.replace("unforced", "forced_IDK_parlai")].stats[
                "⁉️"
            ]
            idkn2, _ = runname2run[k.replace("unforced", "forced_IDK_parlai")].stats[
                "❓"
            ]
            result += f"<tr><td>{k.split('_')[5]}</td><td>{100*cn/cd:.2f}%</td><td>{100*yeacn/yeacd:.2f}%</td><td>{100*yean/yead:.2f}%</td><td>{100*(idkn1+idkn2)/idkd:.2f}%</td></tr>"
    result += "</tbody></table>"
    result += "</center></body></html>"
    return result


@app.route("/probe-results/<string:sweepname>/<string:dset>")
def route_probe_results(sweepname, dset):
    result = "<html><head>" + STYLE + "</head><body><center>"

    # Load all data
    runs = {}
    for fn in itertools.chain.from_iterable(
        glob(f"{DATADIR1}/2020*/mcprobe_{sn}/*/model.{dset}")
        for sn in sweepname.split("+")
    ):
        runid = "/".join(fn.split("/")[3:6])
        runs[runid] = {"stats": {}, "opts": {}}
        # Get results
        with open(fn) as f:
            ls = [l.split() for l in f.read().splitlines()[1:]]
        for names, numbers in zip(ls[:-1:2], ls[1::2]):
            if names[-1] == "\\":
                names = names[:-1]
            assert len(names) == len(numbers), (runid, names, numbers)
            for name, number in zip(names, numbers):
                assert name not in runs[runid]["stats"]
                try:
                    runs[runid]["stats"][name] = int(number)
                except ValueError:
                    runs[runid]["stats"][name] = float(number)
        # Get hyperparams
        with open(fn.replace(f"model.{dset}", "model.opt")) as f:
            runs[runid]["opts"] = json.load(f)

    result += "<h1>All found runs</h1>"
    result += "<table><tbody><tr>"
    result += "<th>run</th>"
    result += "<th>balance</th>"
    result += "<th>#layers</th>"
    result += "<th>hidsize</th>"
    result += "<th>pooling</th>"
    result += "<th>prepool</th>"
    result += "<th>claimed data</th>"
    result += "<th>lr</th>"
    result += "<th>accuracy</th>"
    result += "<th>P</th>"
    result += "<th>R</th>"
    result += "<th>F1</th>"
    result += "<th>% pred=RIGHT</th>"
    result += "</tr>"
    for runid, run in sorted(
        runs.items(),
        key=lambda t: (
            t[1]["stats"]["ALL_RIGHT_f1"],
            sum(t[1]["stats"][k] for k in ("ALL_RIGHT_FP", "ALL_RIGHT_TP")),
        ),
        reverse=True,
    ):
        if run["stats"]["ALL_accuracy"] < 0.5 and run["stats"]["ALL_RIGHT_f1"] < 0.2:
            continue
        pred_right = sum(run["stats"][k] for k in ("ALL_RIGHT_FP", "ALL_RIGHT_TP"))
        total = sum(
            run["stats"][k]
            for k in ("ALL_RIGHT_FN", "ALL_RIGHT_FP", "ALL_RIGHT_TN", "ALL_RIGHT_TP")
        )
        result += "<tr>"
        result += f"<td>{runid}</td><td>{run['opts']['balance_correctness']}</td>"
        result += f"<td>{run['opts']['n_classifier_layers']}</td>"
        result += f"<td>{run['opts']['classifier_hidsize']}</td>"
        result += f"<td>{run['opts']['classifier_state_pooling']}</td>"
        result += f"<td>{run['opts']['classifier_state_pre_pooling']}</td>"
        result += f"<td>{run['opts']['claimed_data']}</td>"
        result += f"<td>{run['opts']['learningrate']}</td>"
        if run["stats"]["ALL_accuracy"] > 0.948:
            result += f"<td><b>{100*run['stats']['ALL_accuracy']:.1f}%<b></td>"
        else:
            result += f"<td>{100*run['stats']['ALL_accuracy']:.1f}%</td>"
        result += f"<td>{run['stats']['ALL_RIGHT_prec']:.4f}</td>"
        result += f"<td>{run['stats']['ALL_RIGHT_recall']:.4f}</td>"
        result += f"<td>{run['stats']['ALL_RIGHT_f1']:.4f}</td>"
        result += f"<td>{100*pred_right/total:.1f}%</td>"
        result += "</tr>"
    result += "</tbody></table>"

    return result


@app.route("/beam-texts")
def route_beam_texts():
    result = "<html><head>" + STYLE + "</head><body><center>"

    ss = TriviaQARun.get_run(
        f"{DATADIR2}/NoEvidenceUnion_blender_3B_topbeams_triviaqa:NoEvidenceUnion_replies.jsonl"
    ).samples

    result += f"<h1>Out of {len(ss)}...</h1>"
    result += f"Correct and some other beam correct: {sum(s.is_correct and any(s.beam_is_correct) for s in ss)}<br>"
    result += f"Correct but no other beam correct: {sum(s.is_correct and (not any(s.beam_is_correct)) for s in ss)}<br>"
    result += f"Not correct but some other beam correct: {sum((not s.is_correct) and any(s.beam_is_correct) for s in ss)}<br>"
    result += f"Not correct and no other beam correct: {sum((not s.is_correct) and (not any(s.beam_is_correct)) for s in ss)}<br>"

    result += "<h1>Count correctnesses</h1>"
    result += "Each sample has 1 top prediction, that's either right or wrong.<br>"
    result += "How many of the best 9 other beam entries were right?<br><br>"
    result += "<table><tbody>"
    result += "<tr><td></td><th colspan=2>top correct?</th></tr>"
    result += "<tr><th>#correct / next 9</th><th>yes</th><th>no</th></tr>"
    bins = [{True: 0, False: 0} for _ in range(10)]
    for s in ss:
        bins[sum(s.beam_is_correct[1:10])][s.is_correct] += 1
    for i, cs in enumerate(bins):
        result += f"<tr><td>{i}</td><td>{cs[True]}</td><td>{cs[False]}</td></tr>"
    result += "</tbody></table>"

    return result


def calibration_metrics(pos_probs, blender_got_it_right, nbins=20):
    start, end = 0.0, 1.0
    bins = [[] for _ in range(nbins)]

    assert len(pos_probs) == len(blender_got_it_right)

    for pl, blender_right in zip(pos_probs, blender_got_it_right):
        if nbins > 2:
            i = int(pl * nbins - 1e-4)
        else:
            i = int(pl >= 0.375)
        bins[i].append(1.0 if blender_right else 0.0)

    print("calibrator vs. empirical -> diff")
    for i, b in enumerate(bins):
        print(
            f"{(i + 0.5) * 1 / nbins:.3f} vs. "
            + (
                f"{sum(b) / len(b):.3f} -> {abs((i + 0.5) * 1 / nbins - sum(b) / len(b)):.3f} ({len(b)} ex)"
                if b
                else "empty"
            )
        )

    # ECE / MCE
    distances = [
        abs(np.mean(b) - (i + 0.5) / nbins) if b else 0.0 for i, b in enumerate(bins)
    ]
    ece = sum(len(b) * d for b, d in zip(bins, distances)) / len(pos_probs)
    mce = max(distances)

    # NLL
    nll = np.mean(
        [
            -math.log(p) if c else -math.log(1 - p)
            for p, c in zip(pos_probs, blender_got_it_right)
        ]
    )

    return ece, mce, nll


with open(f"{MCDIR}/calibrator_training_answers.txt") as f:
    leaking_training_answers = set(f.read().splitlines())

q2noleak = {
    strip_control(s.question): all(g not in leaking_training_answers for g in s.gold)
    for s in TriviaQARun.get_run(
        f"{DATADIR0}/NewParlAITriviaQA/probe_ametsub_446_says_3x5000_blender3B_test_parlai_external.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl"
    ).samples
}


@app.route("/probe-certainties")
@app.route("/probe-certainties/<string:infix>")
def route_probe_certainties(infix="probe_ametsub_446_says"):
    result = "<html><head>" + STYLE + "</head><body><center>"

    ss = TriviaQARun.get_run(
        f"{DATADIR0}/NewParlAITriviaQA/{infix}_3x5000_blender3B_test_parlai_external.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl"
    ).samples

    with open(
        f"{MCDIR}/annotations/validset/3x5000_blender3B_test.majorities.simplified_annotations.json"
    ) as data_file:
        blender_annotations = json.load(data_file)["Data"]

    blender_got_it_right = [
        {"🔇": False, "❌": False, "🧶": True, "💯": True, "✗": False, "✔": True}[
            d["annotation"]["correctness"]
        ]
        for d in blender_annotations
        if q2noleak[d["question"]]
    ]
    ss = [s for s in ss if q2noleak[s.question]]
    print(len(ss))

    if False:
        result += "<h1>Histogram: probe certainties</h1>"

        result += render_altair_plot(
            alt.Chart(pd.DataFrame({"max p": [s.maxprob for s in ss]}))
            .mark_bar()
            .encode(x=alt.X("max p:Q", bin=alt.Bin(maxbins=90)), y="count()")
            .properties(width=200, height=150)
        )
        result += render_altair_plot(
            alt.Chart(
                pd.DataFrame(
                    {"-log (1 - max p)": [-math.log(1 - s.maxprob) for s in ss]}
                )
            )
            .mark_bar()
            .encode(x=alt.X("-log (1 - max p):Q", bin=alt.Bin(maxbins=50)), y="count()")
            .properties(width=200, height=150)
        )
        result += render_altair_plot(
            alt.Chart(
                pd.DataFrame(
                    {
                        "p(RIGHT)": [
                            {"RIGHT": s.maxprob, "WRONG": 1 - s.maxprob}[s.prediction]
                            for s in ss
                        ]
                    }
                )
            )
            .mark_bar()
            .encode(x=alt.X("p(RIGHT):Q", bin=alt.Bin(maxbins=80)), y="count()")
            .properties(width=200, height=150)
        )
        result += render_altair_plot(
            alt.Chart(
                pd.DataFrame(
                    {
                        "log p(RIGHT)": [
                            math.log(
                                {"RIGHT": s.maxprob, "WRONG": 1 - s.maxprob}[
                                    s.prediction
                                ]
                            )
                            for s in ss
                        ]
                    }
                )
            )
            .mark_bar()
            .encode(x=alt.X("log p(RIGHT):Q", bin=alt.Bin(maxbins=50)), y="count()")
            .properties(width=200, height=150)
        )

        # Take "-log (1 - max p)" as a measure of *certainty*
        certainties = [-math.log(1 - s.maxprob) for s in ss]
        start, end = min(certainties), max(certainties) + 1e-5
        nbins = 60
        binsize = (end - start) / nbins
        bins = {
            "unsplit": [[] for _ in range(nbins)],
            "by_blender": {b: [[] for _ in range(nbins)] for b in (True, False)},
            "by_probe": {b: [[] for _ in range(nbins)] for b in (True, False)},
        }

        for cert, blender_right, s in zip(certainties, blender_got_it_right, ss):
            i = int(((cert - start) / (end - start)) * nbins)
            probe_says_right = s.prediction == "RIGHT"
            acc = 1.0 if probe_says_right == blender_right else 0.0
            bins["unsplit"][i].append(acc)
            bins["by_blender"][blender_right][i].append(acc)
            bins["by_probe"][probe_says_right][i].append(acc)

        result += (
            "<h1>Probe certainties (<i>-log (1 - max p)</i>) vs. Probe accuracy</h1>"
        )

        cert_coords = [start + (i + 0.5) * binsize for i in range(nbins)]

        def getchart(bins, label):
            bin_sizes = [len(cs) for cs in bins]
            bin_sizes = [s / max(bin_sizes) for s in bin_sizes]
            return (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "certainty_left": [
                                x - 3 * s / nbins
                                for x, s in zip(cert_coords, bin_sizes)
                            ],
                            "certainty_right": [
                                x + 3 * s / nbins
                                for x, s in zip(cert_coords, bin_sizes)
                            ],
                            label: [
                                -np.mean(cs) if "orange" in label else np.mean(cs)
                                for cs in bins
                            ],
                        }
                    )
                )
                .mark_rect()
                .encode(
                    x="certainty_left",
                    x2="certainty_right",
                    y=label,
                    color=alt.value(
                        "green"
                        if "green" in label
                        else ("orange" if "orange" in label else "black")
                    ),
                )
                .properties(width=600, height=400 if "chance" in label else 200)
            )

        result += render_altair_plot(
            alt.hconcat(
                getchart(bins["unsplit"], "correctness chance"),
                alt.vconcat(
                    alt.layer(
                        getchart(
                            bins["by_blender"][True], "where Blender is right (green)"
                        ),
                        getchart(bins["by_blender"][False], "wrong (orange)"),
                    ),
                    alt.layer(
                        getchart(
                            bins["by_probe"][True], "where probe said RIGHT (green)"
                        ),
                        getchart(bins["by_probe"][False], "WRONG (orange)"),
                    ),
                ),
            )
        )

    # We want a single x-y point cloud and so it shall be:
    result += "<h1>Probe positive ([log] p(RIGHT)) vs. Actual [log] p(RIGHT), i.e., average RIGHT-ness</h1>"

    row = []
    for name, transform, minval, maxval in (
        ("", lambda x: x, 0, 1),
        # ("log ", np.log, -6.5, 0),
    ):
        pos_probs = [
            transform({"RIGHT": s.maxprob, "WRONG": 1 - s.maxprob}[s.prediction])
            for s in ss
        ]
        start, end = min(pos_probs), max(pos_probs) + 1e-3
        nbins = 20
        binsize = (end - start) / nbins
        bins = [[] for _ in range(nbins)]

        for pl, blender_right in zip(pos_probs, blender_got_it_right):
            i = int(((pl - start) / (end - start)) * nbins)
            bins[i].append(1.0 if blender_right else 0.0)

        chart = alt.Chart(
            pd.DataFrame(
                {
                    name
                    + "Output of correctness classifier": [
                        start + (i + 0.5) * binsize for i in range(nbins)
                    ],
                    name
                    + "Actual correctness": [transform(np.mean(cs)) for cs in bins],
                    "number of data points": [len(cs) for cs in bins],
                }
            )
        )
        bubbles = chart.mark_point().encode(
            x=alt.X(
                name + "Output of correctness classifier:Q",
                scale=alt.Scale(domain=(minval, maxval)),
            ),
            y=alt.Y(
                name + "Actual correctness:Q", scale=alt.Scale(domain=(minval, maxval))
            ),
            size="number of data points:Q",
        )
        labels = chart.mark_text(align="left", baseline="middle", dx=5, dy=-5).encode(
            x=alt.X(
                name + "Output of correctness classifier:Q",
                scale=alt.Scale(domain=(minval, maxval)),
            ),
            y=alt.Y(
                name + "Actual correctness:Q", scale=alt.Scale(domain=(minval, maxval))
            ),
            text="number of data points",
        )

        row.append((bubbles + labels).properties(width=250, height=250))
        ece, mcc, nll = calibration_metrics(pos_probs, blender_got_it_right)
        row[-1].title = f"ece {ece:.3f} mcc {mcc:.3f} nll {nll:.3f}"
    result += render_altair_plot(alt.hconcat(*row).configure_title(fontSize=32))

    # What do the top scoring examples look like?
    result += "<h1>Top scoring (under probe) examples</h1>"
    result += "<table><tbody><tr><th>p(RIGHT)</th><th>correct</th><th>question</th><th>prediction</th></tr>"
    samples = [
        (
            {"RIGHT": s.maxprob, "WRONG": 1 - s.maxprob}[s.prediction],
            {
                "🔇": "WRONG",
                "❌": "WRONG",
                "🧶": "RIGHT",
                "💯": "RIGHT",
                "✗": False,
                "✔": True,
            }[d["annotation"]["correctness"]],
            d["question"],
            d["prediction"],
        )
        for s, d in zip(ss, blender_annotations)
    ]
    for pp, co, qu, pr in sorted(samples, reverse=True)[:200]:
        result += f"<tr><td>{pp:.3f}</td><td>{co}</td><td>{qu}</td><td>{pr}</td></tr>"
    result += "</tbody></table>"

    return result


@app.route("/probe-ablations")
def route_probe_ablations():
    result = "<html><head>" + STYLE + "</head><body><center>"

    with open(
        f"{MCDIR}/annotations/validset/3x5000_blender3B_test.majorities.simplified_annotations.json"
    ) as data_file:
        blender_annotations = json.load(data_file)["Data"]

    blender_got_it_right = [
        {"🔇": False, "❌": False, "🧶": True, "💯": True, "✗": False, "✔": True}[
            d["annotation"]["correctness"]
        ]
        for d in blender_annotations
    ]

    noleak = [q2noleak[d["question"]] for d in blender_annotations]

    columns = []
    for kind, infix in [
        ("+enc +dec", "probe_luca_76f_says"),
        ("-enc +dec", "probe_luca_04b_says_no_enc"),
        ("+enc -dec", "probe_luca_5b2_says_no_dec"),
        ("-enc -dec", "probe_luca_bb4_says_no_both"),
        # ("+enc +dec", "probe_ametsub_446_says"),
        # ("-enc +dec", "probe_ametsub_446_says_no_enc"),
        # ("+enc -dec", "probe_ametsub_446_says_no_dec"),
        # ("-enc -dec", "probe_ametsub_446_says_no_both"),
        # ("bert-qp", "bert_calibrator_0020_bert-qp_says"),
        # ("bert-q", "bert_calibrator_0020_bert-q_says"),
        # ("bert-p", "bert_calibrator_0020_bert-p_says"),
    ]:
        ss = TriviaQARun.get_run(
            f"{DATADIR0}/NewParlAITriviaQA/{infix}_3x5000_blender3B_test_parlai_external.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl"
        ).samples

        column = []
        for name, transform, minval, maxval in (
            ("", lambda x: x, 0.0, 1.0),
            # ("log ", np.log, -9.0, 0.0),
        ):
            pos_probs = [
                transform({"RIGHT": s.maxprob, "WRONG": 1 - s.maxprob}[s.prediction])
                for s in ss
            ]
            start, end = min(pos_probs), max(pos_probs) + 1e-3
            nbins = 20
            binsize = (end - start) / nbins
            bins = [[] for _ in range(nbins)]

            for pl, blender_right, nl in zip(pos_probs, blender_got_it_right, noleak):
                if not nl:
                    continue
                i = int(((pl - start) / (end - start)) * nbins)
                bins[i].append(1.0 if blender_right else 0.0)

            chart = alt.Chart(
                pd.DataFrame(
                    {
                        name
                        + "Output of correctness classifier": [
                            start + (i + 0.5) * binsize for i in range(nbins)
                        ],
                        name
                        + "Actual correctness": [transform(np.mean(cs)) for cs in bins],
                        "number of data points": [len(cs) for cs in bins],
                    }
                )
            )
            bubbles = chart.mark_point().encode(
                x=alt.X(
                    name + "Output of correctness classifier:Q",
                    scale=alt.Scale(domain=(minval, maxval)),
                ),
                y=alt.Y(
                    name + "Actual correctness:Q",
                    scale=alt.Scale(domain=(minval, maxval)),
                ),
                size="number of data points:Q",
            )
            labels = chart.mark_text(
                align="left", baseline="middle", dx=5, dy=7
            ).encode(
                x=alt.X(
                    name + "Output of correctness classifier:Q",
                    scale=alt.Scale(domain=(minval, maxval)),
                ),
                y=alt.Y(
                    name + "Actual correctness:Q",
                    scale=alt.Scale(domain=(minval, maxval)),
                ),
                text="number of data points",
            )

            column.append((bubbles).properties(width=300, height=300))

        # columns.append(alt.vconcat(*column))
        columns.append(column[0])
        columns[-1].title = (
            kind + f" {calibration_metrics(pos_probs, blender_got_it_right)}"
        )
    result += render_altair_plot(alt.hconcat(*columns).configure_title(fontSize=32))

    return result


@app.route("/probe-ablations-all")
def route_probe_ablations_all():
    result = "<html><head>" + STYLE + "</head><body><center>"

    with open(
        f"{MCDIR}/annotations/validset/3x5000_blender3B_test.majorities.simplified_annotations.json"
    ) as data_file:
        blender_annotations = json.load(data_file)["Data"]

    blender_got_it_right = [
        {"🔇": False, "❌": False, "🧶": True, "💯": True, "✗": False, "✔": True}[
            d["annotation"]["correctness"]
        ]
        for d in blender_annotations
    ]

    noleak = [q2noleak[d["question"]] for d in blender_annotations]

    columns = []
    for kind in ("bert-qp",):
        column = []
        for fn in [
            f"{DATADIR0}/bert_calibrator/bert4calibrator_843",
            f"{DATADIR0}/bert_calibrator/bert4calibrator_0d0",
            f"{DATADIR0}/bert_calibrator/bert4calibrator_474",
        ]:
            ss = TriviaQARun.get_run(
                f"{fn}_parlai_external.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl"
            ).samples
            pos_probs = [
                {"RIGHT": s.maxprob, "WRONG": 1 - s.maxprob}[s.prediction] for s in ss
            ]
            start, end = min(pos_probs), max(pos_probs) + 1e-3
            nbins = 20
            binsize = (end - start) / nbins
            bins = [[] for _ in range(nbins)]

            for pl, blender_right, nl in zip(pos_probs, blender_got_it_right, noleak):
                if not nl:
                    continue
                i = int(((pl - start) / (end - start)) * nbins)
                bins[i].append(1.0 if blender_right else 0.0)

            chart = alt.Chart(
                pd.DataFrame(
                    {
                        "Output of correctness classifier": [
                            start + (i + 0.5) * binsize for i in range(nbins)
                        ],
                        "Actual correctness": [np.mean(cs) for cs in bins],
                        "number of data points": [len(cs) for cs in bins],
                    }
                )
            )
            bubbles = chart.mark_point().encode(
                x=alt.X(
                    "Output of correctness classifier:Q",
                    scale=alt.Scale(domain=(0.0, 1.0)),
                ),
                y=alt.Y("Actual correctness:Q", scale=alt.Scale(domain=(0.0, 1.0))),
                size="number of data points:Q",
            )
            labels = chart.mark_text(
                align="left", baseline="middle", dx=5, dy=7
            ).encode(
                x=alt.X(
                    "Output of correctness classifier:Q",
                    scale=alt.Scale(domain=(0.0, 1.0)),
                ),
                y=alt.Y("Actual correctness:Q", scale=alt.Scale(domain=(0.0, 1.0))),
                text="number of data points",
            )

            columns.append((bubbles + labels).properties(width=300, height=300))
            columns[-1].title = (
                fn.split("/")[-1]
                + f" {calibration_metrics(pos_probs, blender_got_it_right)}"
            )

        # columns.append(alt.vconcat(*column))
        # # columns.append(column[0])
        # columns[-1].title = kind
    result += render_altair_plot(alt.hconcat(*columns).configure_title(fontSize=16))

    return result


def mc_paired_perm_test(xs, ys, nsamples=100_000, statistic=np.sum):
    assert len(xs) == len(ys)

    def effect(xs, ys):
        return np.abs(statistic(xs) - statistic(ys))

    n, k = xs.shape, 0
    observed_difference = effect(xs, ys)
    for j in range(nsamples):
        if j > 0 and j % 10_000 == 0:
            print(f"{k} / {j}")
        swaps = np.random.randint(0, 2, n).astype(bool)
        xs_sampled = np.select([swaps, ~swaps], [xs, ys])
        ys_sampled = np.select([~swaps, swaps], [xs, ys])
        k += observed_difference <= effect(xs_sampled, ys_sampled)
    # fraction of random samples that achieved at least the observed difference
    p = k / nsamples
    print(f"{k} / {nsamples}")
    return p


@app.route("/calibrate")
def route_calibrate():
    result = "<html><head>" + STYLE + "</head><body><center>"

    with open(f"{MCDIR}/3x4000_blender3B_test_fourmodels.jsonl") as f:
        all_samples = [json.loads(d) for d in f.read().splitlines()]

    testvalid_qs = set(s["question"] for s in all_samples)
    pos_probs = {
        s.question: {"RIGHT": s.maxprob, "WRONG": 1 - s.maxprob}[s.prediction]
        for s in TriviaQARun.get_run(
            f"{DATADIR0}/NewParlAITriviaQA/probe_ametsub_446_says_3x5000_blender3B_test_parlai_external.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl"
            # f"{DATADIR0}/NewParlAITriviaQA/probe_luca_bb4_says_no_both_3x5000_blender3B_test_parlai_external.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl"
        ).samples
        if s.question in testvalid_qs
        # d["dialog"][0][0]["text"].splitlines()[0]: d["dialog"][0][1]["posprob"]
        # for d in [json.loads(l) for l in open(
        #     f"{DATADIR0}/bert_calibrator/bert4calibrator_0d0_parlai_external.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl"
        # ).read().splitlines()]
        # if "text" in d["dialog"][0][0] and d["dialog"][0][0]["text"].splitlines()[0] in testvalid_qs
    }

    # all_samples = [s for s in all_samples if s["question"] in pos_probs]

    assert all(p > 0 and p < 1 for p in pos_probs.values())
    assert len(pos_probs) == len(all_samples), [len(pos_probs), len(all_samples)]

    four_cos_ces_ps = {
        t.split("_")[-1]: {
            s["question"]: (
                {"✗": False, "✔": True}[s[t]["correctness"]],
                {"🤷": "<IDK>", "💁": "<TRY>", "🙋": "<YEA>", "🏃": "<EVA>"}[
                    s[t]["certainty"]
                ],
                s[t]["prediction"],
            )
            for s in all_samples
        }
        for t in ("vanilla", "forced_IDK", "forced_TRY", "forced_YEA")
    }

    with open(f"{MCDIR}/3x5000_blender3B_test_fourmodels.non_simplified.jsonl") as f:
        non_simplified_samples = [json.loads(d) for d in f.read().splitlines()]
        non_simplified_four_cos_ces_ps = {
            t.split("_")[-1]: {
                s["question"]: (
                    {"🔇": "OTHER", "❌": "WRONG", "🧶": "EXTRA", "💯": "RIGHT"}[
                        s[t]["correctness"]
                    ],
                    {"🤷": "<IDK>", "💁": "<TRY>", "🙋": "<YEA>", "🏃": "<EVA>"}[
                        s[t]["certainty"]
                    ],
                    s[t]["prediction"],
                )
                for s in non_simplified_samples
            }
            for t in ("vanilla", "forced_IDK", "forced_TRY", "forced_YEA")
        }

    result += f"<h1>{len(all_samples)}, {len(pos_probs)}, {[len(x) for x in four_cos_ces_ps.values()]}</h1>"

    if False:

        def qpa2t(q, p, a):
            return (
                q,
                (
                    {
                        "🔇": False,
                        "❌": False,
                        "🧶": True,
                        "💯": True,
                        "✗": False,
                        "✔": True,
                    }[a["correctness"]],
                    {"🤷": "<IDK>", "💁": "<TRY>", "🙋": "<YEA>", "🏃": "<EVA>"}[
                        a["certainty"]
                    ],
                    p,
                ),
            )

        def get_cos_ces_ps(path_co, path_ce):
            samples = [
                s
                for s in TriviaQARun.get_run(glob1(path_co)).samples
                if strip_control(s.question) in pos_probs
            ]
            ps = {strip_control(s.question): s.prediction for s in samples}
            cos = {strip_control(s.question): s.is_correct for s in samples}
            ces = {
                strip_control(s.question.split("\n")[0]): s.prediction
                for s in TriviaQARun.get_run(glob1(path_ce)).samples
                if strip_control(s.question.split("\n")[0]) in pos_probs
            }
            assert len(set((len(cos), len(ces), len(ps), len(pos_probs)))) == 1
            return {
                q: (cos[q], ces[q], ps[q]) for q in cos.keys() | ces.keys() | ps.keys()
            }

        finetune_run = "pearl_342"
        # finetune_run = "oathbreaker_448"
        # finetune_run = "alcest_94c"
        four_cos_ces_ps = {
            target: get_cos_ces_ps(
                path_co=f"{DATADIR0}/NewParlAITriviaQA/NoEvidenceUnion_blender_3B_finetuned_{finetune_run}_forced_{target}_parlai_??ternal.projects.metacognition.agents:NoEvidenceUnionForced{target}Teacher_replies.jsonl",
                path_ce=f"{DATADIR0}/NewParlAITriviaQA/triviaqa_full_166_valid_finetuned_{finetune_run}_forced_{target}_parlai_??ternal.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl",
            )
            for target in ("IDK", "TRY", "YEA")
        }
        four_cos_ces_ps["vanilla"] = get_cos_ces_ps(
            path_co=f"{DATADIR2}/NoEvidenceUnion_blender_3B_default_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl",
            path_ce=f"{DATADIR2}/triviaqa_full_166_valid_parlai_??ternal.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl",
        )

        with open(
            f"{MCDIR}/annotations/validset/3x2000_blender3B_valid.majorities.simplified_annotations.json"
        ) as data_file:
            vanilla_human_cos_ces_ps = dict(
                qpa2t(d["question"], d["prediction"], d["annotation"])
                for d in json.load(data_file)["Data"]
            )
        assert vanilla_human_cos_ces_ps.keys() == pos_probs.keys()

    # The rest is just source-agnostic rendering!

    result += "<table><tbody>"
    # <tr><th></th>"
    # for field in ("EVA", "IDK", "TRY", "YEA"):
    #     result += f"<th>{field}</th>"
    # result += "</tr>"

    def thresholds2mixedsamples(idk_try, try_yea):
        return {
            q: four_cos_ces_ps["IDK"][q]
            if pos_prob <= idk_try
            else (
                four_cos_ces_ps["TRY"][q]
                if pos_prob <= try_yea
                else four_cos_ces_ps["YEA"][q]
            )
            for q, pos_prob in pos_probs.items()
        }

    for name, merged_cos_ces_ps in [
        ("<b>Vanilla</b>", four_cos_ces_ps["vanilla"]),
        ("forced_IDK", four_cos_ces_ps["IDK"]),
        ("forced_TRY", four_cos_ces_ps["TRY"]),
        ("forced_YEA", four_cos_ces_ps["YEA"]),
        *[
            (f"Mix: {a} / {b}", thresholds2mixedsamples(a, b))
            # for a in np.arange(0, 1.01, 0.025)
            # for b in np.arange(a, 1.01, 0.025)
            for a, b in [
                (0.025, 0.025),
                (0.0, 0.275),
                (0.0, 0.375),  ##
                # (0.0, 0.5),
                # (0.0, 0.525),
                # (0.0, 0.55),
                # (0.0, 0.875),
            ]
        ],
        (
            "Oracle",
            {
                q: four_cos_ces_ps["YEA"][q]
                if four_cos_ces_ps["vanilla"][q][0]
                else four_cos_ces_ps["IDK"][q]
                for q, _ in pos_probs.items()
            },
        ),
    ]:
        result += f"<tr><th>{name}</th>"
        for field in ("EVA", "IDK", "TRY", "YEA"):
            result += f"<th>{field}</th>"
        result += "</tr>"

        ce2cos = {target: [] for target in ("<EVA>", "<IDK>", "<TRY>", "<YEA>")}
        for co, ce, _ in merged_cos_ces_ps.values():
            ce2cos[ce].append(co)

        # if "Mix" in name and (not ce2cos["<YEA>"] or sum(ce2cos["<YEA>"]) / len(ce2cos["<YEA>"]) < .4 or len(ce2cos["<YEA>"]) < 10): continue

        cos_vanilla = []
        cos_thisone = []
        isyea_vanilla = []
        isyea_thisone = []
        # for q, (co, ce, _) in four_cos_ces_ps["vanilla"].items():
        for q, (co, ce, _) in thresholds2mixedsamples(0.0, 0.375).items():
            cos_vanilla.append(co)
            cos_thisone.append(merged_cos_ces_ps[q][0])
            isyea_vanilla.append(ce == "<YEA>")
            isyea_thisone.append(merged_cos_ces_ps[q][1] == "<YEA>")

        result += "<tr><td></td><td colspan=4>"
        result += f"overall accuracy: {sum(cos_thisone)} / {len(cos_thisone)} = "
        result += f"{100 * sum(cos_thisone) / len(cos_thisone):.2f}%"
        ps = [
            mc_paired_perm_test(
                xs=np.array(cos_vanilla), ys=np.array(cos_thisone), nsamples=1_000
            )
            for _ in range(1)
        ]
        result += " ||| significance over vanilla at p = "
        result += " / ".join([f"{p:.10f}" for p in ps])
        result += "</td></tr>"

        result += "<tr><td></td><td colspan=4>"
        result += f"YEA-accuracy: {100*np.mean(np.array(cos_thisone)[np.array(isyea_thisone)]):.3f}%"
        ps = [
            mc_paired_perm_test(
                xs=np.array(list(zip(cos_vanilla, isyea_vanilla))),
                ys=np.array(list(zip(cos_thisone, isyea_thisone))),
                statistic=lambda xs: np.mean(xs[:, 0][xs[:, 1]]),
                nsamples=1_000_000,
            )
            for _ in range(1)
        ]
        result += " ||| significance over vanilla at p = "
        result += " / ".join([f"{p:.10f}" for p in ps])
        result += "</td></tr>"

        result += f"<tr><td>per class</td>"

        for field in ("<EVA>", "<IDK>", "<TRY>", "<YEA>"):
            result += f"<td>{100 * sum(ce2cos[field]) / (len(ce2cos[field]) if ce2cos[field] else 1):.1f}%</td>"
        result += "</tr>"
        result += f"<tr><td></td><td colspan=4>"
        for field in ("<EVA>", "<IDK>", "<TRY>", "<YEA>"):
            p = sum(ce2cos[field]) / (len(ce2cos[field]) if ce2cos[field] else 1)
            result += f"<div style='margin:0;padding:0;display:inline-block;width:{len(ce2cos[field])/100}em;'>"
            result += f"<div style='margin:0;padding:0;background-color:#fcc;height:{(1-p)*5}em'><span style='text-color:gray;font-size:40%;'>{field[1:-1]}<span></div>"
            result += f"<div style='margin:0;padding:0;background-color:green;height:{p*5}em'></div>"
            result += "</div>"
        result += "</td></tr>"
        result += f"<tr><td></td>"
        for field in ("<EVA>", "<IDK>", "<TRY>", "<YEA>"):
            result += f"<td>{sum(ce2cos[field])} / {len(ce2cos[field])}</td>"
        result += "</tr>"

        c, t = 0, 0
        for q, (new_co, new_ce, new_p) in merged_cos_ces_ps.items():
            old_co, old_ce, old_p = four_cos_ces_ps["vanilla"][q]
            if (
                # new_ce == old_ce or set([new_ce, old_ce]) == set(["<IDK>", "<TRY>"])
                not (
                    not old_co
                    and old_ce == "<YEA>"
                    and pos_probs[q] < 0.375
                    and not new_co
                    and new_ce in ["<TRY>", "<IDK>"]
                    and len(q) > 50
                    and len(q) < 90
                    and not new_p.startswith("I'm not sure, but I do know that")
                )
            ):
                continue

            c += old_p == new_p
            t += 1
            if False:  # True or "do know" not in new_p:
                result += f"<tr><td></td><td></td><td style='text-align:right;'><b>{q}</b></td><td>{old_co}</td><td>{old_ce[1:-1]}: {old_p}<br>{new_ce[1:-1]}: {new_p}</td></tr>"

        result += f"<tr><td colspan=5>{c}, {t}</td></tr>"
        # simplified ratios
        result += "<tr><td></td>"
        ces = ("<EVA>", "<IDK>", "<TRY>", "<YEA>")
        for ce in ces:
            _c = sum(ce2cos[ce])
            _l = len(ce2cos[ce])
            result += (
                f"<td>{100 * (_l - _c) / _l:.2f}\\% & {100 * _c / _l:.2f} \\% & </td>"
            )
        result += "</tr>"
        # nonsimplified ratios
        cos = ("OTHER", "WRONG", "EXTRA", "RIGHT")
        _tot = 0
        ce2nonsimplifiedcos = {target: {co: 0 for co in cos} for target in ces}
        for q, (_, _, p) in merged_cos_ces_ps.items():
            if q not in non_simplified_four_cos_ces_ps["vanilla"]:
                continue
            candidates = [
                non_simplified_four_cos_ces_ps[target][q]
                for target in ("vanilla", "IDK", "TRY", "YEA")
            ]
            [co] = set([co for (co, ce, _p) in candidates if _p == p])
            [ce] = set([ce for (co, ce, _p) in candidates if _p == p])
            ce2nonsimplifiedcos[ce][co] += 1
            _tot += 1
        result += f"<tr><td>out of {_tot}:</td>"
        for ce in ces:
            result += "<td>"
            for co in cos:
                result += f"{100 * ce2nonsimplifiedcos[ce][co] / sum(ce2nonsimplifiedcos[ce].values()):.2f}\\% & "
            result += "</td>"
        result += "</tr>"
    result += "</tbody></table>"

    result += "<br><hr><br><table><tbody><tr><td>r_YES</td>"
    gradation = list(np.arange(0, 0.9001, 0.025))
    for b in gradation:
        result += f"<th>{b:.3f}</th>"
    result += "</tr>"
    for a in gradation:
        result += f"<tr><th>{a:.3f}</th>"
        for b in gradation:
            if a > b:
                result += "<td></td>"
                continue
            ce2cos = {target: [] for target in ("EVA", "IDK", "TRY", "YEA")}
            for co, ce, _ in thresholds2mixedsamples(a, b).values():
                ce2cos[ce[1:-1]].append(co)
            rs = {
                field: sum(ce2cos[field]) / (len(ce2cos[field]) if ce2cos[field] else 1)
                for field in ("EVA", "IDK", "TRY", "YEA")
            }
            if rs["IDK"] < rs["TRY"] and rs["TRY"] < rs["YEA"]:
                if rs["YEA"] > 0.5:
                    result += f"<td><b>{100*rs['YEA']:.1f}%</b></td>"
                else:
                    result += f"<td>{100*rs['YEA']:.1f}%</td>"
            else:
                result += f"<td></td>"
        result += "</tr>"
    result += "</tbody></table>"

    return result


def turks_to_simplified_majority(annotations, simplify=True):
    sas = simplify_annotations(annotations, binary_correctness=simplify)
    ce2count = Counter(a["certainty"] for a in sas)
    co2count = Counter(a["correctness"] for a in sas)
    if ce2count.most_common()[0][1] > (len(sas) / 2) and co2count.most_common()[0][
        1
    ] > (len(sas) / 2):
        return {
            "certainty": ce2count.most_common()[0][0],
            "correctness": co2count.most_common()[0][0],
        }
    else:
        return False


def finalmephisto_worker2qp2a():
    d = defaultdict(dict)
    for s in ["", "_need3", "_need2", "_need1", "_need1_1211"]:
        with open(f"{MCDIR}/file_you_can_send_me{s}.json") as f:
            print(s)
            for worker, qp2a in json.load(f).items():
                for qp, a in qp2a.items():
                    if qp in d[worker] and d[worker][qp] != a:
                        print(d[worker][qp], a)
                    d[worker][qp] = a
    return dict(d)


@app.route("/finalmephistoextract")
def route_finalmephistoextract():
    result = "<html><head>" + STYLE + "</head><body><center>"

    annotated_qp2as = {}

    with open(f"{MCDIR}/webapp/src/static/blender3B_all_four_dedup_test.jsonl") as f:
        for d in f.read().splitlines():
            d = json.loads(d)
            qp = d["question"] + "#=%=#" + d["prediction"]
            annotated_qp2as[qp] = []

    worker2qp2a = finalmephisto_worker2qp2a()
    for worker, qp2a in worker2qp2a.items():
        for qp, a in qp2a.items():
            if qp in annotated_qp2as:
                annotated_qp2as[qp].append([worker, a])

    targets = ("vanilla", "forced_IDK", "forced_TRY", "forced_YEA")
    target2qps = {}
    for target in targets:
        with open(f"{MCDIR}/webapp/src/static/blender3B_{target}_test.jsonl") as f:
            target2qps[target] = [
                d["question"] + "#=%=#" + d["prediction"]
                for d in [json.loads(l) for l in f.read().splitlines()]
            ]

    with open(f"{MCDIR}/3x1000_blender3B_test_fourmodels.jsonl") as f:
        taken_1k_questions = [
            d["question"] for d in [json.loads(l) for l in f.read().splitlines()]
        ]

    counts = [0, 0, 0, 0, 0, 0, 0]
    min_counts = [0, 0, 0, 0, 0, 0, 0]
    choices = []
    for qp_tuple in zip(*[target2qps[t] for t in targets]):
        min_count = 999
        for qp in qp_tuple:
            min_count = min(min_count, len(annotated_qp2as[qp]))
            counts[len(annotated_qp2as[qp])] += 1
        min_counts[min_count] += 1
        if min_count == 3:
            # Try to make majorities
            majorities = [
                turks_to_simplified_majority(
                    [a for _, a in annotated_qp2as[qp]], simplify=False
                )
                for qp in qp_tuple
            ]
            q, _ = qp_tuple[0].split("#=%=#")
            if all(majorities) and q:  # not in taken_1k_questions:
                choices.append(
                    {
                        "question": q,
                        **{
                            t: {"prediction": qp.split("#=%=#")[1], **ma}
                            for t, qp, ma in zip(targets, qp_tuple, majorities)
                        },
                    }
                )

    ctr = Counter()
    for c in choices:
        ctr[(c["vanilla"]["certainty"], c["vanilla"]["correctness"])] += 1
    t = sum(ctr.values())
    # return f"<h1>{t}</h1>" + "<br>".join(f"{(v1, v2)}: {100*ctr[(v1, v2)]/t:.2f}" for v1 in "🏃🤷💁🙋" for v2 in "✗✔")  # "🔇❌🧶💯")

    # FALSE
    random.seed(0)
    random.shuffle(choices)
    choices = choices[:10000]

    result += f"<h3>from {counts[3]} to {len(choices)}</h3>"
    with open(
        f"{MCDIR}/3x5000_blender3B_test_fourmodels.non_simplified.jsonl", "wt"
    ) as f:
        for c in choices:
            print(json.dumps(c), file=f)

    with open(
        f"{MCDIR}/annotations/validset/3x5000_blender3B_test.majorities.non_simplified_annotations.json",
        "wt",
    ) as f:
        json.dump(
            {
                "Data": [
                    {
                        "question": c["question"],
                        "prediction": c["vanilla"]["prediction"],
                        "annotation": {
                            "correctness": c["vanilla"]["correctness"],
                            "certainty": c["vanilla"]["certainty"],
                        },
                    }
                    for c in choices
                ]
            },
            f,
        )

    missing = ([], [], [], [])

    with open(f"{MCDIR}/webapp/src/static/blender3B_all_four_dedup_test.jsonl") as f:
        for l in f.read().splitlines():
            d = json.loads(l)
            qp = d["question"] + "#=%=#" + d["prediction"]
            missingness = max(0, 3 - len(annotated_qp2as[qp]))
            missing[missingness].append(d)

    for missingness in [1, 2, 3]:
        with open(
            f"{MCDIR}/webapp/src/static/blender3B_all_four_dedup_test__need_{missingness}_more.jsonl",
            "wt",
        ) as f:
            for d in missing[missingness]:
                print(json.dumps(d), file=f)

    return result + f"{min_counts}<br>{counts}<br>{[len(l) for l in missing]}"


@app.route("/mephistojudge_final", methods=["GET", "POST"])
def route_mephistojudge_final():
    result = "<html><head>" + STYLE + "</head><body><center>"

    # post_data = dict(request.form)

    worker2qp2a = finalmephisto_worker2qp2a()

    for worker, qp2a in sorted(
        worker2qp2a.items(), key=lambda t: len(t[1]), reverse=True
    ):
        total_counter = OrderedDict(
            [
                ("🏃", 0),
                ("\\", ""),
                ("🤷🔇", 0),
                ("🤷❌", 0),
                ("🤷🧶", 0),
                ("🤷💯", 0),
                ("|", ""),
                ("💁🔇", 0),
                ("💁❌", 0),
                ("💁🧶", 0),
                ("💁💯", 0),
                ("/", ""),
                ("🙋🔇", 0),
                ("🙋❌", 0),
                ("🙋🧶", 0),
                ("🙋💯", 0),
            ]
        )

        # Analyze batch
        certainty_counter = {s: 0 for s in "🏃🤷💁🙋"}
        correctness_counter = {s: 0 for s in "🔇❌🧶💯"}
        _r = "<table><tbody>"
        for qp, a in qp2a.items():
            total_counter[a] += 1
            (d,) = simplify_annotations([a])
            certainty_counter[d["certainty"]] += 1
            correctness_counter[d["correctness"]] += 1
            _r += f"<tr><td>{a}</td><td>{d['certainty']} {d['correctness']}</td>"
            _r += f"<td style='width:50em;font-size:30%;'>{qp.split('#=%=#')[0]}</td>"
            _r += f"<td style='width:50em;font-size:30%;'>{qp.split('#=%=#')[0]}</td>"
            # _r += f"<td style='width:15em;font-size:30%;'>{ins['golds'][0]}</td>
            _r += "</tr>"
        _r += "</tbody></table>"

        certainty_counter = {
            k: c / sum(certainty_counter.values()) for k, c in certainty_counter.items()
        }
        correctness_counter = {
            k: c / sum(correctness_counter.values())
            for k, c in correctness_counter.items()
        }

        sketchyness_reasons = []
        if (
            max(certainty_counter.values()) > 0.99
            and max(correctness_counter.values()) > 0.99
        ):
            sketchyness_reasons.append("all the same!")
        if len(qp2a) > 9:
            if max(certainty_counter.values()) > 0.66:
                sketchyness_reasons.append("more than 66% 1 certainty")
            if max(correctness_counter.values()) == 1.0:
                sketchyness_reasons.append("only 1 correctness")
            if (correctness_counter["🧶"] + correctness_counter["💯"]) > 0.15:
                sketchyness_reasons.append("more than 15% correct")
            if certainty_counter["🏃"] > 0.2:
                sketchyness_reasons.append("more than 20% evasion")
        if len(qp2a) > 100:
            for symbol in "🤷💁🙋🔇❌🧶💯":  # "🏃🤷💁🙋🔇❌🧶💯":
                covered = False
                for emojis, count in total_counter.items():
                    if symbol in emojis and count > 0:
                        covered = True
                        break
                if not covered:
                    sketchyness_reasons.append(f"Missing symbol {symbol}!")

        if sketchyness_reasons:
            result += f"<h2>{worker}: {len(qp2a)} questions</h2>"
            result += f"<h4>{', '.join(sketchyness_reasons)}</h4>"
            result += "<table><tbody><tr>"
            result += "".join([f"<th>{emoji}</th>" for emoji in total_counter.keys()])
            result += "</tr><tr>"
            result += "".join([f"<td>{count}</td>" for count in total_counter.values()])
            result += "</tr></tbody></table><hr>"
            result += f"<form action='/mephistojudge_worker/{worker}' method='POST'>"
            result += "<input style='width:8em; height:auto;' name='bonus_dollars' value='$' />"
            result += "<input style='width:auto; height:auto;' type='submit' "
            result += " name='whole worker' value='accept worker' />"
            result += "<input style='width:30em; height:auto;' name='rejection_reason' "
            result += " value='Annotations are contrary to instructions.' />"
            result += "<input style='width:auto; height:auto;' type='submit' "
            result += " name='whole worker' value='reject worker' />"
            result += "</form><hr>"
            # result += _r
            # result += "<hr>"

    result += "</center></body></html>"
    return result


@app.route("/certaintysankey")
def route_certaintysankey():
    result = "<html><head>" + STYLE
    result += """
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
        <script type="text/javascript">
        google.charts.load('current', {'packages':['sankey']});
        google.charts.setOnLoadCallback(drawChartBig);
        function drawChartBig() {
            var data = new google.visualization.DataTable();
            data.addColumn('string', 'From');
            data.addColumn('string', 'To');
            data.addColumn('number', 'Weight');
            data.addRows([
    """

    q2posprob = {
        s.question: {"RIGHT": s.maxprob, "WRONG": 1 - s.maxprob}[s.prediction]
        for s in TriviaQARun.get_run(
            f"{DATADIR0}/NewParlAITriviaQA/probe_ametsub_446_says_3x4000_blender3B_test_parlai_external.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl"
        ).samples
    }

    ctr = OrderedDict(
        (f"['vanilla: {a}', 'calibrator-controlled: {b}', ", 0)
        for a in ("EVA", "IDK", "TRY", "YEA")
        for b in ("EVA", "IDK", "TRY", "YEA")
    )
    chatbot2certainty2cos = {
        "vanilla": {"EVA": [], "IDK": [], "TRY": [], "YEA": []},
        "calibrator-controlled": {"EVA": [], "IDK": [], "TRY": [], "YEA": []},
    }
    with open(f"{MCDIR}/3x4000_blender3B_test_fourmodels.jsonl") as f:
        for l in f.read().splitlines():
            d = json.loads(l)
            target = "forced_TRY" if q2posprob[d["question"]] <= 0.375 else "forced_YEA"
            a = {"🤷": "<IDK>", "💁": "<TRY>", "🙋": "<YEA>", "🏃": "<EVA>"}[
                d["vanilla"]["certainty"]
            ]
            b = {"🤷": "<IDK>", "💁": "<TRY>", "🙋": "<YEA>", "🏃": "<EVA>"}[
                d[target]["certainty"]
            ]
            ctr[f"['vanilla: {a[1:-1]}', 'calibrator-controlled: {b[1:-1]}', "] += 1
            chatbot2certainty2cos["vanilla"][a[1:-1]].append(
                {"✗": False, "✔": True}[d["vanilla"]["correctness"]]
            )
            chatbot2certainty2cos["calibrator-controlled"][b[1:-1]].append(
                {"✗": False, "✔": True}[d[target]["correctness"]]
            )
    result += ",".join([f"{k}{count}]" for k, count in ctr.items()])

    result += f"""]);
            var colors = [ '#005AB5', '#DC3220' ]
            //var colors = [ '#DC3220', '#DC3220', '#DC3220', '#DC3220' ]
            var options = {{
                width: 300,
                sankey: {{
                    node: {{colors: colors, width: 30}},
                    iterations: 0
                }}
            }};
            (new google.visualization.Sankey(document.getElementById('sankey_big'))).draw(data, options);
        }}
        </script>
    """

    result += "<br><br>"
    result += '&nbsp;&nbsp;<div id="sankey_big" style="height: 300px; display:inline-block;"></div>'

    certainties = []
    p_corrects = []
    chatbots = []
    for chatbot, cert2cos in chatbot2certainty2cos.items():
        for ce, cos in cert2cos.items():
            certainties.append(ce)
            p_corrects.append(sum(cos) / len(cos))
            chatbots.append(chatbot)

    assert len(certainties) == len(p_corrects)
    assert len(certainties) == len(chatbots)

    df = pd.DataFrame(
        {"certainty": certainties, "percent correct": p_corrects, "chatbot": chatbots}
    )

    result += render_altair_plot(
        alt.layer(
            alt.Chart(df)
            .mark_bar(size=20)
            .encode(
                y=alt.Y(
                    "percent correct:Q",
                    scale=alt.Scale(domain=(0, 1)),
                    axis=alt.Axis(tickCount=5, format=".2%"),
                ),
                color=alt.Color("chatbot:O", scale=alt.Scale(scheme="tableau10")),
                x=alt.X(
                    "chatbot:O",
                    sort="descending",
                    title="",
                    axis=alt.Axis(labels=False),
                ),
            ),
            alt.Chart(df)
            .mark_text(align="center", baseline="middle", dx=0, dy=-10)
            .encode(
                y=alt.Y(
                    "percent correct:Q",
                    scale=alt.Scale(domain=(0, 1)),
                    axis=alt.Axis(tickCount=5, format=".0%"),
                ),
                color=alt.Color("chatbot:O", scale=alt.Scale(scheme="tableau10")),
                x=alt.X(
                    "chatbot:O",
                    sort="descending",
                    title="",
                    axis=alt.Axis(labels=False),
                ),
                text=alt.Text("percent correct:Q", format=".1%"),
            ),
            data=df,
        )
        .properties(width=50, height=120)
        .facet(column="certainty:O")
    )

    result += "</body></html>"
    return result


@app.route("/vanilla-ngrams")
def vanilla_ngrams():
    result = "<html><head>" + STYLE + "</head><body><center>"

    with open(f"{MCDIR}/3x5000_blender3B_test_fourmodels.jsonl") as f:
        four_models = [json.loads(l) for l in f.read().splitlines()]

    metrics = [
        ("✗ ≤ ✔", lambda d: {"✗": -1, "✔": 1}[d["vanilla"]["correctness"]], 0.2),
        (
            "EVA/IDK/TRY ≤ YEA",
            lambda d: {"🏃": -1, "🤷": -1, "💁": -1, "🙋": 1}[d["vanilla"]["certainty"]],
            0.2 / 3,
        ),
    ]
    min_ngramcount = 5

    pairs = []
    for source, C2 in (
        (lambda d: d["question"], 1),
        (lambda d: d["vanilla"]["prediction"], 1),
    ):
        vocab = ngram_vocab(
            [source(d) for d in four_models],
            min_ngramcount,
            CachedNGramCoefs.max_ngramorder,
        )
        coefs = {
            metric: LogisticRegression(penalty="l1", solver="liblinear", C=C * C2)
            .fit(
                ngram_featurize([source(d) for d in four_models], vocab),
                [target_featurizer(d) for d in four_models],
            )
            .coef_.ravel()
            for metric, target_featurizer, C in metrics
            if metric is not None
        }
        pairs.append((vocab, coefs))
    (vocab_q, coefs_q), (vocab_a, coefs_a) = pairs

    result += (
        "<h2>Predicting metrics from at least "
        + f"{min_ngramcount} times occuring {{2--7}}-grams</h2>"
    )
    result += "<table><tbody><tr>"
    result += f"<td colspan='{len(metrics)}'>"
    result += f"...out of {len(vocab_q)} question n-grams</td>"
    result += "<th class='sep'></th><th class='sep'></th>"
    result += f"<td colspan='{len(metrics)}'>"
    result += f"...out of {len(vocab_a)} answer n-grams</td>"
    result += "</tr><tr>"
    metrics_and_sources = (
        [(m, (vocab_q, coefs_q)) for m, _, _ in metrics]
        + [(None, (None, None)), (None, (None, None))]
        + [(m, (vocab_a, coefs_a)) for m, _, _ in metrics]
    )
    result += "".join(
        [
            f"<th class='metric'>{m}</th>" if m is not None else "<th class='sep'></th>"
            for m, _ in metrics_and_sources
        ]
    )
    result += "</tr>"
    odd = True
    for predicate in [lambda v: v > 0, lambda v: v < 0]:
        result += "<tr>"
        for metric, (vocab, coefs) in metrics_and_sources:
            if metric is None:
                result += "<td class='sep'></td>"
                continue
            result += (
                f"<td class={'odd' if odd else 'even'}>"
                + "<br>".join(
                    [
                        f"{val:.3f} &amp; {ngram}"
                        for val, ngram in sorted(
                            list(zip(coefs[metric], vocab)), reverse=True
                        )
                        if predicate(val)
                    ]
                )
                + "</td>"
            )
        result += "</tr>"
        odd = not odd
    result += "</tbody></table>"
    return result


if __name__ == "__main__":
    app.run()
