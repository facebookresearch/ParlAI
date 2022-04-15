#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import base64
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from glob import glob
from io import BytesIO
import json
import math
import os
import pickle
import subprocess
import tempfile
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
