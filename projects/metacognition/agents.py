#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base64
from io import BytesIO
import itertools
import json
import math  # noqa: F401
import numpy as np
import random
import os.path

import torch

from parlai.agents.bert_classifier.bert_classifier import BertClassifierAgent
from parlai.agents.transformer.modules import TransformerGeneratorModel
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, GlobalTimerMetric, SumMetric
from parlai.core.teachers import DialogTeacher
from parlai.core.torch_agent import Output
from parlai.core.torch_classifier_agent import (
    ConfusionMatrixMetric,
    TorchClassifierAgent,
)
from parlai.tasks.triviaqa.build import build
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.torch import PipelineHelper

from parlai.projects.metacognition.wsgi import (
    MCDIR,
    DATADIR0,
    DATADIR2,
    TriviaQARun,
    glob1,
    strip_control,
)


def _normalize(tensor, norm_layer):
    """Broadcast layer norm."""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


def custom_decoder_layer(layer, x, encoder_output, encoder_mask):
    decoder_mask = layer._create_selfattn_mask(x)
    # first self attn
    residual = x
    if layer.variant == "prelayernorm":
        x = _normalize(x, layer.norm1)

    # don't peak into the future!
    x, final_self_attn_incr_state = layer.self_attention(
        query=x, mask=decoder_mask, incr_state=None, static_kv=False
    )
    x = layer.dropout(x)  # --dropout
    x = x + residual
    if layer.variant == "aiayn" or layer.variant == "xlm":
        x = _normalize(x, layer.norm1)

    residual = x
    # encoder_attn_layer_norm norm 2
    if layer.variant == "prelayernorm":
        x = _normalize(x, layer.norm2)
    x, final_encoder_attn_incr_state = layer.encoder_attention(
        query=x,
        key=encoder_output,
        value=encoder_output,
        mask=encoder_mask,
        incr_state=None,
        static_kv=True,
    )
    attended = x
    x = layer.dropout(x)  # --dropout
    x = residual + x
    if layer.variant == "aiayn" or layer.variant == "xlm":
        x = _normalize(x, layer.norm2)

    # finally the ffn
    residual = x
    if layer.variant == "prelayernorm":
        x = _normalize(x, layer.norm3)
    x = layer.ffn(x)
    x = layer.dropout(x)  # --dropout
    x = residual + x
    if layer.variant == "aiayn" or layer.variant == "xlm":
        x = _normalize(x, layer.norm3)

    return attended, (x, {})


class PartialLossExtractingTransformerGeneratorAgent(TransformerGeneratorAgent):
    def eval_step(self, batch):
        # Cut down version of `TorchGeneratorAgent.eval_step`.
        self.model.eval()
        maxlen = self.label_truncate or 256
        beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen)
        preds, scores = zip(*beam_preds_scores)

        all_full_losses = []
        all_after_but_extract_losses = []
        all_after_but_rescore_losses = []
        all_compressed_attendeds = []
        for i_in_batch, (pred, _score, _beam) in enumerate(zip(preds, scores, beams)):
            # Encode, both for extraction and rescoring later
            encoder_states = self.model.encoder(batch.text_vec[i_in_batch].unsqueeze(0))

            # Extract h! Warning: this is a shitton of disgusting copypaste lol...
            all_attendeds = []
            encoder_output, encoder_mask = encoder_states
            tensor = self.model.decoder.embeddings(
                self.model.START.detach().expand(1, 1)
            )
            if self.model.decoder.embeddings_scale:
                tensor = tensor * np.sqrt(self.model.decoder.dim)
            if self.model.decoder.variant == "xlm":
                tensor = _normalize(tensor, self.model.decoder.norm_embeddings)
            positions = pred.new(1).long()
            positions = torch.arange(1, out=positions).unsqueeze(0)
            tensor = tensor + self.model.decoder.position_embeddings(
                positions
            ).expand_as(tensor)
            tensor = self.model.decoder.dropout(tensor)
            if getattr(self.model.decoder.layers, "is_model_parallel", False):
                chunks = PipelineHelper.split((tensor, encoder_output, encoder_mask))
                work_items = PipelineHelper.schedule_work_items(
                    self.model.decoder.layers, chunks
                )
                for chunk_idx, layer_nos, next_device in work_items:
                    s_tensor, s_enc_out, s_enc_mask = chunks[chunk_idx]
                    for layer_no in layer_nos:
                        attended, (s_tensor, _) = custom_decoder_layer(
                            self.model.decoder.layers[layer_no],
                            s_tensor,
                            s_enc_out,
                            s_enc_mask,
                        )
                        all_attendeds.append(attended[0, 0, :].cpu())
                        chunks[chunk_idx] = PipelineHelper.chunk_to(
                            (s_tensor, s_enc_out, s_enc_mask), next_device
                        )
            else:
                for layer in self.model.decoder.layers:
                    attended, (tensor, _) = custom_decoder_layer(
                        layer, tensor, encoder_output, encoder_mask
                    )
                    all_attendeds.append(attended[0, 0, :].cpu())
            all_attendeds = torch.stack(all_attendeds).numpy()
            compressed_attendeds = BytesIO()
            np.savez_compressed(compressed_attendeds, all_attendeds)
            all_compressed_attendeds.append(
                base64.b64encode(compressed_attendeds.getvalue()).decode()
            )

            # Safe extraction of original losses
            extract = torch.cat([pred[1:-1], pred.new([self.END_IDX])])
            scores, _ = self.model.decode_forced(encoder_states, extract.unsqueeze(0))
            [nats] = self.criterion(scores.view(-1, scores.size(-1)), extract).view(
                scores.shape[:-1]
            )
            losses = [nat.item() for nat in nats]
            all_full_losses.append(losses)

            # Then extract (and rescore) sub-losses
            [but_type] = self.dict.txt2vec("but")
            if but_type in list(pred):
                but_position = list(pred).index(but_type)

                # Extractive loss
                all_after_but_extract_losses.append(losses[but_position:])

                # Rescoring loss
                extract = torch.cat(
                    [pred[but_position + 1 : -1], pred.new([self.END_IDX])]
                )
                scores, _ = self.model.decode_forced(
                    encoder_states, extract.unsqueeze(0)
                )
                [nats] = self.criterion(scores.view(-1, scores.size(-1)), extract).view(
                    scores.shape[:-1]
                )
                all_after_but_rescore_losses.append([nat.item() for nat in nats])
            else:
                all_after_but_extract_losses.append([])
                all_after_but_rescore_losses.append([])
        return Output(
            [self._v2t(p) for p in preds],
            full_losses=all_full_losses,
            after_but_extract_losses=all_after_but_extract_losses,
            after_but_rescore_losses=all_after_but_rescore_losses,
            first_decoder_attendeds=all_compressed_attendeds,
        )


class ControllingTransformerGeneratorAgent(TransformerGeneratorAgent):
    def __init__(self, opt, shared=None):
        self.controlprob = opt["controlprob"]
        super().__init__(opt, shared)

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument_group(
            "ControllingTransformerGeneratorAgent Arguments"
        ).add_argument(
            "--controlprob",
            default=1.0,
            type=float,
            help="Add corresponding control tokens with this probability",
        )
        super().add_cmdline_args(argparser)

    def get_temp_history(self, observation):
        if "control_string" in observation and np.random.random() < self.controlprob:
            return " " + observation["control_string"].strip()

    def _get_special_tokens(self):
        return ["<IDK>", "<TRY>", "<YEA>", "<SAME>", "<DIFF>"]


def claim_from_training_data(
    balance_correctness,
    certainty_distribution,
    with_eva,
    from_the_back,
    training_samples,
    training_certainty_samples,
    training_samenesses=None,
):
    if training_samenesses is None:
        training_samenesses = [None] * len(training_samples)
        sds = (None,)
    else:
        sds = (True, False)
    # First let's see what we have
    ce2co2sd2stock = {
        "<EVA>": {b: {sd: [] for sd in sds} for b in (True, False)},
        "<IDK>": {b: {sd: [] for sd in sds} for b in (True, False)},
        "<TRY>": {b: {sd: [] for sd in sds} for b in (True, False)},
        "<YEA>": {b: {sd: [] for sd in sds} for b in (True, False)},
    }
    for s_gen, s_cla, is_same in zip(
        training_samples, training_certainty_samples, training_samenesses
    ):
        # If we do SAME/DIFF, stage1 needs to be right or at least vanilla (meaning that
        # stage1 wasn't right but they were different):
        if (
            (is_same is None)
            or s_gen.is_correct
            or (not s_gen.is_correct and not is_same)
            or True
        ):
            ce2co2sd2stock[s_cla.prediction][s_gen.is_correct][is_same].append(
                {
                    "text": strip_control(s_gen.question),
                    "label": s_gen.prediction,
                    "bert_certainty": s_cla.prediction,
                    "regex_correctness": s_gen.is_correct,
                    "is_same": is_same,
                }
            )
    distr, sampling = certainty_distribution.split("-")
    minmax = {"undersample": min, "oversample": max}[sampling]
    # First, what distribution over certainties?
    if distr == "everything":
        if balance_correctness == "anycorrectness":
            ce2co2sd2count = {
                ce: {
                    co: {sd: minmax(len(s) for s in sd2stock.values()) for sd in sds}
                    for co, sd2stock in co2sd2stock.items()
                }
                for ce, co2sd2stock in ce2co2sd2stock.items()
            }
        elif balance_correctness == "balancedcorrectness":
            ce2co2sd2count = {
                ce: {
                    co: {
                        sd: minmax(
                            minmax(len(s) for s in sd2stock.values())
                            for sd2stock in co2sd2stock.values()
                        )
                        for sd in sds
                    }
                    for co in co2sd2stock
                }
                for ce, co2sd2stock in ce2co2sd2stock.items()
            }
        elif balance_correctness == "onlycorrect":
            ce2co2sd2count = {
                ce: {
                    True: {
                        sd: minmax(len(co2sd2stock[True][sd]) for sd in sds)
                        for sd in sds
                    },
                    False: {sd: 0 for sd in sds},
                }
                for ce, co2sd2stock in ce2co2sd2stock.items()
            }
        else:
            raise ValueError(balance_correctness)
        if not with_eva:
            ce2co2sd2count["<EVA>"] = {
                co: {sd: 0 for sd in sds} for co in (True, False)
            }
    else:
        # Set weights based on certainty distribution
        if distr == "uniform":
            Z = 4 if with_eva else 3
            desired_certainty = {
                "<EVA>": (1 / Z) if with_eva else 0,
                "<IDK>": 1 / Z,
                "<TRY>": 1 / Z,
                "<YEA>": 1 / Z,
            }
        elif distr == "natural":
            Z = (102 if with_eva else 0) + 580 + 707 + 611
            desired_certainty = {
                "<EVA>": (102 / Z) if with_eva else 0,
                "<IDK>": 580 / Z,
                "<TRY>": 707 / Z,
                "<YEA>": 611 / Z,
            }
        else:
            raise ValueError(distr)
        # Now what then do all the correctnesses in each of these (weighted)
        # classes want the total to be *maximally*?
        if sampling == "undersample":
            Z = 99999999999  # driven by underrepresented
        else:
            Z = 0  # driven by overrepresented
        for ce, co2sd2stock in ce2co2sd2stock.items():
            p = desired_certainty[ce]
            if p == 0:
                continue
            else:
                assert p > 0
            # So that tells us what to multiply our weights with!
            if balance_correctness == "anycorrectness":
                Z = minmax(
                    Z,
                    sum(
                        len(sds) * minmax(len(stock) for stock in sd2stock.values())
                        for sd2stock in co2sd2stock.values()
                    )
                    / p,
                )
            elif balance_correctness == "balancedcorrectness":
                Z = minmax(
                    Z,
                    minmax(
                        2 * len(sds) * len(co2sd2stock[b][sd]) / p
                        for b in (True, False)
                        for sd in sds
                    ),
                )
            elif balance_correctness == "onlycorrect":
                Z = minmax(
                    Z, len(sds) * minmax(len(co2sd2stock[True][sd]) for sd in sds) / p
                )
            else:
                raise ValueError(balance_correctness)
        print("final Z", Z)
        # Assemble final counts
        if balance_correctness == "anycorrectness":
            ce2co2sd2count = {
                ce: {
                    co: {
                        sd: int(
                            desired_certainty[ce]
                            * Z
                            * len(co2sd2stock[co][sd])
                            / sum(
                                sum(len(stock) for stock in sd2stock.values())
                                for sd2stock in co2sd2stock.values()
                            )
                        )
                        for sd in sds
                    }
                    for co in (True, False)
                }
                for ce, co2sd2stock in ce2co2sd2stock.items()
            }
        elif balance_correctness == "balancedcorrectness":
            ce2co2sd2count = {
                ce: {
                    co: {
                        sd: int(desired_certainty[ce] * Z / (2 * len(sds)))
                        for sd in sds
                    }
                    for co in (True, False)
                }
                for ce in ce2co2sd2stock
            }
        elif balance_correctness == "onlycorrect":
            ce2co2sd2count = {
                ce: {
                    True: {sd: int(desired_certainty[ce] * Z / len(sds)) for sd in sds},
                    False: {sd: 0 for sd in sds},
                }
                for ce in ce2co2sd2stock
            }
        else:
            raise ValueError(balance_correctness)
    # Then let's compile it!
    all_data = []
    for ce, co2sd2count in ce2co2sd2count.items():
        for co, sd2count in co2sd2count.items():
            for sd, count in sd2count.items():
                stock = []
                if not ce2co2sd2stock[ce][co][sd] and not (sd and not co):
                    raise Exception(f"No available stock for {(ce, co, sd)}")
                while len(stock) < count:
                    stock += ce2co2sd2stock[ce][co][sd]
                all_data += stock[:count]
    return all_data


# -t parlai.projects.metacognition.agents:CertaintyControlTeacher
class CertaintyControlTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.claimed_triviaqa = opt["claimed_data"]
        self.is_training = opt["datatype"].startswith("train")
        self.balance_correctness = opt["balance_correctness"]
        self.certainty_distribution = opt["certainty_distribution"]
        self.with_fished = opt["with_fished"]
        self.with_eva = opt["with_eva"]
        self.stage0_free_beam = opt["stage0_free_beam"]
        self.stage1_results = opt["stage1_results"]
        if self.stage1_results:
            assert self.with_fished == 0.0
            assert not self.stage0_free_beam
        self.id = "triviaqa-controlledselftrain"
        self.dir_annotations = opt["dir_annotations"]
        self.dir_runs = opt["dir_runs"]
        if self.is_training:
            opt["datafile"] = None
        else:
            opt["datafile"] = (
                self.dir_annotations
                + "/validset/3x2000_blender3B_valid.majorities.simplified_annotations.json"
            )
        super().__init__(opt, shared)

    def setup_data(self, path):
        emoji2string = {
            "❓": "<UNCERTAIN>",
            "❗": "<CERTAIN>",
            "🤷": "<IDK>",
            "💁": "<TRY>",
            "🙋": "<YEA>",
        }
        vanilla_samples = (
            TriviaQARun.get_run(
                self.dir_runs
                + "/NoEvidenceUnion_blender_3B_default_trainset_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl",
                no_cache=True,
            ).samples
            + TriviaQARun.get_run(
                self.dir_runs
                + "/NoEvidenceUnion_blender_3B_default_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl",
                no_cache=True,
            ).samples
        )
        q2vanillap = {s.question: s.prediction for s in vanilla_samples}
        q2vanillacorrectness = {s.question: s.is_correct for s in vanilla_samples}

        if self.is_training and self.stage1_results:
            raw_samples = []
            raw_certainty_samples = []
            raw_samenesses = []
            for t in ("unforced", "forced_IDK", "forced_TRY", "forced_YEA"):
                training_samples = TriviaQARun.get_run(
                    glob1(
                        self.dir_runs
                        + f"/NoEvidenceUnion_blender_3B_trainset_finetuned_{self.stage1_results}_{t}*_replies.jsonl"
                    ),
                    no_cache=True,
                ).samples
                raw_samples += training_samples[: self.claimed_triviaqa]
                raw_certainty_samples += TriviaQARun.get_run(
                    glob1(
                        self.dir_runs
                        + f"/triviaqa_full_166_finetuned_{self.stage1_results}_{t}*_replies.jsonl"
                    ),
                    no_cache=True,
                ).samples[: self.claimed_triviaqa]
                raw_samenesses += [
                    q2vanillacorrectness[strip_control(s.question)] == s.is_correct
                    for s in training_samples
                ]

            all_data = [
                {
                    "control_string": (
                        d["bert_certainty"]
                        + " "
                        + ("<SAME>" if d["is_same"] else "<DIFF>")
                        + " "
                        + q2vanillap[strip_control(d["text"])]
                    ).strip(),
                    **d,
                }
                for d in claim_from_training_data(
                    balance_correctness=self.balance_correctness,
                    certainty_distribution=self.certainty_distribution,
                    with_eva=self.with_eva,
                    from_the_back=False,
                    training_samples=raw_samples,
                    training_certainty_samples=raw_certainty_samples,
                    training_samenesses=raw_samenesses,
                )
            ]

            # Get statistics about parts and the whole
            ce2co2sd2data = {
                "<EVA>": {co: {True: [], False: []} for co in (True, False)},
                "<IDK>": {co: {True: [], False: []} for co in (True, False)},
                "<TRY>": {co: {True: [], False: []} for co in (True, False)},
                "<YEA>": {co: {True: [], False: []} for co in (True, False)},
            }
            set_all_data = set()
            for d in all_data:
                tup = (d["text"], d["label"], d["control_string"])
                certainty, sameness, *_ = d["control_string"].split(" ")
                sameness = {"<SAME>": True, "<DIFF>": False}[sameness]
                # correctness = q2vanillacorrectness[strip_control(d["text"])]
                correctness = d["regex_correctness"]
                ce2co2sd2data[certainty][correctness][sameness].append(tup)
                set_all_data.add(tup)
            print("\n\n")
            print(f"all data: {len(all_data):5} ({len(set_all_data):5} unique)\n")
            print("cer", "|              counts               |            uniques")
            print("tai", "|     correct     |    incorrect    " * 2)
            print("nty", "|  SAME  |  DIFF  |  SAME  |  DIFF  " * 2)
            print("---", "+--------+--------+--------+--------" * 2, sep="-")
            for (certainty, co2sd2data) in ce2co2sd2data.items():
                print(
                    certainty[1:-1],
                    f"{len(co2sd2data[True][True]):6}",
                    f"{len(co2sd2data[True][False]):6}",
                    f"{len(co2sd2data[False][True]):6}",
                    f"{len(co2sd2data[False][False]):6}",
                    f"{len(set(co2sd2data[True][True])):6}",
                    f"{len(set(co2sd2data[True][False])):6}",
                    f"{len(set(co2sd2data[False][True])):6}",
                    f"{len(set(co2sd2data[False][False])):6}",
                    sep=" | ",
                )
            print("---", "+--------+--------+--------+--------" * 2, sep="-")
            print(
                "sum",
                f"{sum([len(c[True][True]) for c in ce2co2sd2data.values()]):6}",
                f"{sum([len(c[True][False]) for c in ce2co2sd2data.values()]):6}",
                f"{sum([len(c[False][True]) for c in ce2co2sd2data.values()]):6}",
                f"{sum([len(c[False][False]) for c in ce2co2sd2data.values()]):6}",
                f"{sum([len(set(c[True][True])) for c in ce2co2sd2data.values()]):6}",
                f"{sum([len(set(c[True][False])) for c in ce2co2sd2data.values()]):6}",
                f"{sum([len(set(c[False][True])) for c in ce2co2sd2data.values()]):6}",
                f"{sum([len(set(c[False][False])) for c in ce2co2sd2data.values()]):6}",
                sep=" | ",
            )
            print("\n")
        elif self.is_training:
            all_data = [
                {"control_string": d["bert_certainty"], **d}
                for d in claim_from_training_data(
                    balance_correctness=self.balance_correctness,
                    certainty_distribution=self.certainty_distribution,
                    with_eva=self.with_eva,
                    from_the_back=False,
                    training_samples=TriviaQARun.get_run(
                        self.dir_runs
                        + f"/NoEvidenceUnion_blender_3B_{'freebeam' if self.stage0_free_beam else 'default'}_trainset_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl",
                        no_cache=True,
                    ).samples[: self.claimed_triviaqa],
                    training_certainty_samples=TriviaQARun.get_run(
                        glob1(
                            self.dir_runs
                            + f"/triviaqa_{'freebeam_' if self.stage0_free_beam else ''}full_166_parlai_??ternal.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl"
                        ),
                        no_cache=True,
                    ).samples[: self.claimed_triviaqa],
                )
            ]
            # Now add fishing data
            if self.with_fished > 0:
                take_for_each_fish = int(self.with_fished * len(all_data) / 2)
                fishpond = self.dir_annotations + "/fishing"
                for name in ("reddit_full_166.jsonl", "bst_full_166.jsonl"):
                    with open(os.path.join(fishpond, name)) as data_file:
                        all_data += list(
                            itertools.islice(
                                (
                                    {
                                        "text": "\n".join(
                                            text["text"].splitlines()[:-1]
                                        ),
                                        "label": text["text"].splitlines()[-1],
                                        "control_string": label["text"],
                                    }
                                    for text, label in [
                                        json.loads(l)["dialog"][0] for l in data_file
                                    ]
                                    if "text" in label and label["text"] != "<EVA>"
                                ),
                                take_for_each_fish,
                            )
                        )

            # Sanity check: what did we get? (use with dd or something)
            certainty2correctness2finaldata = {
                "<EVA>": {True: [], False: []},
                "<IDK>": {True: [], False: []},
                "<TRY>": {True: [], False: []},
                "<YEA>": {True: [], False: []},
            }
            set_all_data = set()
            for d in all_data:
                tup = (d["text"], d["label"], d["control_string"])
                certainty2correctness2finaldata[d["control_string"][:5]][
                    q2vanillacorrectness[strip_control(d["text"])]
                ].append(tup)
                set_all_data.add(tup)
            print("\n")
            print(f">>> all_data: {len(all_data):5} ({len(set_all_data):5} unique)")
            for (
                certainty,
                correctness2finaldata,
            ) in certainty2correctness2finaldata.items():
                print(">>>", certainty[1:-1], end=": ")
                for label, data in [
                    ("correct", correctness2finaldata[True]),
                    ("incorrect", correctness2finaldata[False]),
                    (
                        "total",
                        correctness2finaldata[True] + correctness2finaldata[False],
                    ),
                ]:
                    print(
                        f"{len(data):5}x",
                        f"({100 * len(data) / (len(data) if data else 1):3.0f}%,",
                        f"{len(set(data)):5} unique)",
                        label,
                        end="\n" if label == "total" else " / ",
                    )
            print("\n")
        else:
            with open(path) as data_file:
                all_data = [
                    {
                        "text": d["question"],
                        "label": d["prediction"],
                        "control_string": emoji2string[d["annotation"]["certainty"]]
                        + (
                            " <SAME> " + q2vanillap[d["question"]]
                            if self.stage1_results
                            else ""
                        ),
                    }
                    for d in json.load(data_file)["Data"]
                    if d["annotation"]["certainty"] != "🏃"
                ]

        # shuffle and return
        random.seed(42)
        random.shuffle(all_data)
        for d in all_data:
            yield d, True

    @classmethod
    def add_cmdline_args(cls, argparser):
        group = argparser.add_argument_group("CertaintyControlTeacher Arguments")
        group.add_argument(
            "--claimed-data",
            default=999999,
            type=int,
            help="Only consider this many training examples from TriviaQA (~75k).",
        )
        group.add_argument(
            "--with-fished",
            default=0.0,
            type=float,
            help="How much (as a ratio of the main data) to add in fishing, divided equally among BST and Reddit.",
        )
        group.add_argument(
            "--balance-correctness",
            type=str,
            help="Balance correctness in the training data? (one of: anycorrectness / balancedcorrectness / onlycorrect)",
        )
        group.add_argument(
            "--certainty-distribution",
            type=str,
            help="How much should each certainty be represented in training data? (one of: everything / {uniform,natural}-{over,under}sample)",
        )
        group.add_argument(
            "--with-eva",
            default=False,
            type=bool,
            help="Include <EVA> as a controllable category",
        )
        group.add_argument(
            "--stage1-results",
            type=str,
            default="",
            help="Path to the `_replies.jsonl` on the entire TriviaQA training set that is to be compared to the vanilla answers to get SAME/DIFF control variables to inject",
        )
        group.add_argument(
            "--stage0-free-beam",
            type=bool,
            default=False,
            help="Use for initial finetuning results generated with or without `--beam-context-block-ngram 0 --beam-block-ngram 0 --beam-min-length 1`",
        )
        group.add_argument(
            "--dir-annotations",
            type=str,
            default=f"{MCDIR}/annotations",
            help="Directory with MTurk annotated validation set",
        )
        group.add_argument(
            "--dir-runs",
            type=str,
            default=f"{DATADIR2}",
            help="Directory with MTurk annotated validation set",
        )


def _path(opt):
    build(opt)

    return (
        os.path.join(opt["datapath"], "TriviaQA", "qa"),
        os.path.join(opt["datapath"], "TriviaQA", "evidence"),
    )


class NoEvidenceUnionControlledTeacher(DialogTeacher):
    def __init__(self, opt, shared=None, control=None):
        if not hasattr(self, "prefix"):
            self.prefix = ""
            self.suffix = "train" if opt["datatype"].startswith("train") else "dev"

        qa_dir, self.evidence_dir = _path(opt)
        opt["datafile"] = os.path.join(
            qa_dir, self.prefix + "noevidence-union-" + self.suffix + ".json"
        )
        self.id = "triviaqa"
        self.control = control
        self.force_same = opt["force_same"]
        super().__init__(opt, shared)

    def setup_data(self, path):
        assert self.suffix is not None
        if self.force_same:
            q2vanillap = {
                s.question: s.prediction
                for s in itertools.chain.from_iterable(
                    TriviaQARun.get_run(
                        f"{DATADIR2}/NoEvidenceUnion_blender_3B_default_{t}withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl",
                        no_cache=True,
                    ).samples
                    for t in ("", "trainset_")
                )
            }
        with open(path) as data_file:
            data = json.load(data_file)["Data"]
        for d in data:
            question = d["Question"] + " " + self.control
            if self.force_same:
                question += " <SAME> " + q2vanillap[d["Question"]]
            answers = [d["Answer"]["Value"]] + sorted(list(set(d["Answer"]["Aliases"])))
            yield (question, answers), True

    @classmethod
    def add_cmdline_args(cls, argparser):
        group = argparser.add_argument_group("ControlForced Teacher Arguments")
        group.add_argument(
            "--force-same",
            type=bool,
            default=False,
            help="Add <SAME> control token and vanilla Blender answer to prompt.",
        )


class NoEvidenceUnionForcedCertainTeacher(NoEvidenceUnionControlledTeacher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, control="<CERTAIN>", **kwargs)


class NoEvidenceUnionForcedUncertainTeacher(NoEvidenceUnionControlledTeacher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, control="<UNCERTAIN>", **kwargs)


class NoEvidenceUnionUnforcedTeacher(NoEvidenceUnionControlledTeacher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, control="", **kwargs)


class NoEvidenceUnionForcedIDKTeacher(NoEvidenceUnionControlledTeacher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, control="<IDK>", **kwargs)


class NoEvidenceUnionForcedTRYTeacher(NoEvidenceUnionControlledTeacher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, control="<TRY>", **kwargs)


class NoEvidenceUnionForcedYEATeacher(NoEvidenceUnionControlledTeacher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, control="<YEA>", **kwargs)


class CertaintyClassificationTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        path = f"{MCDIR}/annotations/validset/"
        dataset = (
            "3x2000_blender3B_valid.majorities"
            if opt["datatype"].startswith("test")
            else "1x2000_blender3B_train.majorities"
        )
        opt["datafile"] = os.path.join(path, f"{dataset}.simplified_annotations.json")
        self.qp = opt["classify_with_prediction"]
        self.simplify = opt["simplify_certainty"]
        self.id = "triviaqa-certainty-classify"
        super().__init__(opt, shared)

    def setup_data(self, path):
        print("loading: " + path)
        with open(path) as data_file:
            data = json.load(data_file)["Data"]
        for d in data:
            yield {
                "text": d["question"] + "\n" + d["prediction"]
                if self.qp
                else d["question"],
                "label": {
                    "🤷": "<IDK>",
                    "💁": "<TRY>" if not self.simplify else "<IDK>",
                    "🙋": "<YEA>",
                    "🏃": "<EVA>",
                }[d["annotation"]["certainty"]],
                "label_candidates": ["<IDK>", "<TRY>", "<YEA>", "<EVA>"]
                if not self.simplify
                else ["<IDK>", "<YEA>", "<EVA>"],
            }, True

    @classmethod
    def add_cmdline_args(cls, argparser):
        group = argparser.add_argument_group("CertaintyClassificationTeacher Arguments")
        group.add_argument(
            "--classify-with-prediction",
            default=True,
            type=bool,
            help="Add the model generation to the to be classified string",
        )
        group.add_argument(
            "--simplify-certainty",
            default=True,
            type=bool,
            help="Turning <TRY> into <IDK>",
        )


class CertaintyOntoBSTTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt["datafile"] = os.path.join(
            opt["datapath"],
            "blended_skill_talk",
            opt["datatype"].split(":")[0] + ".json",
        )
        self.simplify = opt["simplify_certainty"]
        self.nturns = opt["nturns"]
        self.id = "certainty-onto-bst"
        super().__init__(opt, shared)

    def setup_data(self, path):
        nleft = self.nturns
        with open(path) as data_file:
            data = json.load(data_file)
        for d in data:
            for (_, h), (_, t) in zip(d["dialog"], d["dialog"][1:]):
                if h.endswith("?"):
                    yield {
                        "text": h + "\n" + t,
                        "label_candidates": ["<IDK>", "<TRY>", "<YEA>", "<EVA>"]
                        if not self.simplify
                        else ["<IDK>", "<YEA>", "<EVA>"],
                    }, True
                    nleft -= 1
                    if nleft == 0:
                        return

    @classmethod
    def add_cmdline_args(cls, argparser):
        group = argparser.add_argument_group("CertaintyClassificationTeacher Arguments")
        group.add_argument(
            "--nturns", default=-1, type=int, help="How many samples to show"
        )
        group.add_argument(
            "--simplify-certainty",
            default=True,
            type=bool,
            help="Turning <TRY> into <IDK>",
        )


class CertaintyOntoTextfileTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt["datafile"] = opt["textfile"]
        self.simplify = opt["simplify_certainty"]
        self.nturns = opt["nturns"]
        self.id = "certainty-onto-textfile"
        super().__init__(opt, shared)

    def setup_data(self, path):
        nleft = self.nturns
        with open(path) as data_file:
            for line in data_file:
                if line:
                    line = line.replace("<br>", "\n")
                    if len(line) >= 510:
                        line = line[-510:]
                    yield {
                        "text": line,
                        "label_candidates": ["<IDK>", "<TRY>", "<YEA>", "<EVA>"]
                        if not self.simplify
                        else ["<IDK>", "<YEA>", "<EVA>"],
                    }, True
                    nleft -= 1
                    if nleft == 0:
                        return

    @classmethod
    def add_cmdline_args(cls, argparser):
        group = argparser.add_argument_group("CertaintyOntoTextfileTeacher Arguments")
        group.add_argument(
            "--nturns", default=-1, type=int, help="How many samples to show"
        )
        group.add_argument(
            "--simplify-certainty",
            default=True,
            type=bool,
            help="Turning <TRY> into <IDK>",
        )
        group.add_argument(
            "--textfile",
            default=f"{DATADIR0}/NewFinetuningSweepStats/yes_qps.txt",
            type=str,
            help="Text file",
        )


class CertaintyOntoTriviaQATeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        if not hasattr(self, "prefix"):
            self.prefix = ""
            self.suffix = "train" if opt["datatype"].startswith("train") else "dev"

        qa_dir, self.evidence_dir = _path(opt)
        assert opt["datatype"].startswith("train")
        opt["datafile"] = opt["triviaqa_run_to_be_annotated"]
        self.id = "triviaqa"
        self.simplify = opt["simplify_certainty"]
        self.nsamples = opt["nsamples"]
        self.id = "certainty-onto-triviaqa"
        super().__init__(opt, shared)

    def setup_data(self, path):
        nleft = self.nsamples
        for s in TriviaQARun.get_run(path, no_cache=True).samples:
            yield {
                "text": s.question + "\n" + s.prediction,
                "label_candidates": ["<IDK>", "<TRY>", "<YEA>", "<EVA>"]
                if not self.simplify
                else ["<IDK>", "<YEA>", "<EVA>"],
            }, True
            nleft -= 1
            if nleft == 0:
                return

    @classmethod
    def add_cmdline_args(cls, argparser):
        group = argparser.add_argument_group("CertaintyClassificationTeacher Arguments")
        group.add_argument(
            "--nsamples", default=-1, type=int, help="How many samples to show"
        )
        group.add_argument(
            "--simplify-certainty",
            default=True,
            type=bool,
            help="Turning <TRY> into <IDK>",
        )
        group.add_argument(
            "--triviaqa-run-to-be-annotated",
            default=f"{DATADIR2}/NoEvidenceUnion_blender_3B_default_trainset_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl",
            type=str,
            help="replies.jsonl of the TriviaQA run that is to be annotated",
        )


class CorrectnessProbingTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.claimed_triviaqa = opt["claimed_data"]
        self.is_training = opt["datatype"].startswith("train")
        self.balance_correctness = opt["balance_correctness"]
        self.certainty_distribution = opt["certainty_distribution"]
        self.with_eva = opt["with_eva"]
        self.classes = opt["classes"]
        self.simplify = opt["simplify_correctness"]
        self.correctness_prediction_mode = opt["correctness_prediction_mode"]
        if self.is_training:
            opt["datafile"] = f"{MCDIR}/tmp{random.randint(0, 1000)}"
        else:
            partition = (
                "3x5000_blender3B_test"
                if opt["datatype"].startswith("test")
                else "1x2000_blender3B_train"
            )
            opt[
                "datafile"
            ] = f"{MCDIR}/annotations/validset/{partition}.majorities.simplified_annotations.json"
        self.path = opt["datafile"]
        self.id = "triviaqa-correctness-probe"
        self.examples = []
        super().__init__(opt, shared)

    def _lazy_build_samples(self):
        if len(self.examples) > 0:
            return
        path = self.path
        rights = 0
        totals = 0
        if self.is_training:
            for d in claim_from_training_data(
                balance_correctness=self.balance_correctness,
                certainty_distribution=self.certainty_distribution,
                with_eva=self.with_eva,
                from_the_back=True,
                training_samples=TriviaQARun.get_run(
                    f"{DATADIR2}/NoEvidenceUnion_blender_3B_default_trainset_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl",
                    no_cache=True,
                ).samples[: self.claimed_triviaqa],
                training_certainty_samples=TriviaQARun.get_run(
                    f"{DATADIR2}/triviaqa_full_166_parlai.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl",
                    no_cache=True,
                ).samples[: self.claimed_triviaqa],
            ):
                if d["regex_correctness"]:
                    rights += 1
                totals += 1
                desired_class = "RIGHT" if d["regex_correctness"] else "WRONG"
                if self.correctness_prediction_mode == "probing":
                    self.examples.append(
                        (
                            {
                                "text": d["text"],
                                "label": d["label"],
                                "desired_class": desired_class,
                                "certainty": d["bert_certainty"][1:-1],
                            },
                            True,
                        )
                    )
                elif self.correctness_prediction_mode == "bert-q":
                    self.examples.append(
                        (
                            {
                                "text": d["text"],
                                "label": desired_class,
                                "label_candidates": ["WRONG", "RIGHT"],
                            },
                            True,
                        )
                    )
                elif self.correctness_prediction_mode == "bert-qp":
                    self.examples.append(
                        (
                            {
                                "text": d["text"] + "\n" + d["label"],
                                "label": desired_class,
                                "label_candidates": ["WRONG", "RIGHT"],
                            },
                            True,
                        )
                    )
                elif self.correctness_prediction_mode == "bert-p":
                    self.examples.append(
                        (
                            {
                                "text": d["label"],
                                "label": desired_class,
                                "label_candidates": ["WRONG", "RIGHT"],
                            },
                            True,
                        )
                    )
                else:
                    raise Exception()
        else:
            with open(path) as data_file:
                for d in json.load(data_file)["Data"]:
                    correctness = {
                        "🔇": "OTHER" if not self.simplify else "WRONG",
                        "❌": "WRONG",
                        "🧶": "EXTRA" if not self.simplify else "RIGHT",
                        "💯": "RIGHT",
                        "✗": "WRONG",
                        "✔": "RIGHT",
                    }[d["annotation"]["correctness"]]
                    certainty = {
                        "🤷": "IDK",
                        "💁": "TRY" if not self.simplify else "IDK",
                        "🙋": "YEA",
                        "🏃": "EVA",
                    }[d["annotation"]["certainty"]]
                    if correctness == "RIGHT":
                        rights += 1
                    totals += 1
                    if self.correctness_prediction_mode == "probing":
                        self.examples.append(
                            (
                                {
                                    "text": d["question"],
                                    "label": d["prediction"],
                                    "desired_class": correctness,
                                    "certainty": certainty,
                                },
                                True,
                            )
                        )
                    elif self.correctness_prediction_mode == "bert-q":
                        self.examples.append(
                            (
                                {
                                    "text": d["question"],
                                    "label": correctness,
                                    "label_candidates": ["WRONG", "RIGHT"],
                                },
                                True,
                            )
                        )
                    elif self.correctness_prediction_mode == "bert-qp":
                        self.examples.append(
                            (
                                {
                                    "text": d["question"] + "\n" + d["prediction"],
                                    "label": correctness,
                                    "label_candidates": ["WRONG", "RIGHT"],
                                },
                                True,
                            )
                        )
                    elif self.correctness_prediction_mode == "bert-p":
                        self.examples.append(
                            (
                                {
                                    "text": d["prediction"],
                                    "label": correctness,
                                    "label_candidates": ["WRONG", "RIGHT"],
                                },
                                True,
                            )
                        )
                    else:
                        raise Exception()
        print(f"\n{rights}/{totals} = {100*rights/totals:.3f}% right!\n")

    def num_examples(self):
        self._lazy_build_samples()
        return len(self.examples)

    def setup_data(self, path):
        self._lazy_build_samples()
        yield from self.examples

    @classmethod
    def add_cmdline_args(cls, argparser):
        group = argparser.add_argument_group("CorrectnessProbeTeacher Arguments")
        group.add_argument(
            "--simplify-correctness",
            default=True,
            type=bool,
            help="Turning OTHER into WRONG and EXTRA into RIGHT",
        )
        group.add_argument(
            "--claimed-data",
            default=999999,
            type=int,
            help="Only consider this many training examples from TriviaQA (~75k).",
        )
        group.add_argument(
            "--balance-correctness",
            type=str,
            help="Balance correctness in the training data? (one of: anycorrectness / balancedcorrectness / onlycorrect)",
        )
        group.add_argument(
            "--certainty-distribution",
            type=str,
            help="How much should each certainty be represented in training data? (one of: everything / {uniform,natural}-{over,under}sample)",
        )
        group.add_argument(
            "--with-eva",
            default=False,
            type=bool,
            help="Include <EVA> as a controllable category",
        )
        group.add_argument(
            "--correctness-prediction-mode",
            default="probing",
            type=str,
            help="Use 'probing' mode (standard), 'bert-q', 'bert-p', or 'bert-qp'",
        )


class ClassifierOnGeneratorModel(TransformerGeneratorModel):
    def __init__(
        self,
        opt,
        dictionary,
        *,
        num_classes: int,
        nlayers: int,
        hidsize: int,
        pooling: str,
        prepooling: str,
        with_encode: bool,
        with_decode: bool,
    ):
        super().__init__(opt, dictionary)

        self.with_encode = with_encode
        self.with_decode = with_decode

        self.layermix = torch.nn.Parameter(torch.ones(25) / 25)

        assert prepooling in ["id", "linear", "linearGELU"]
        if prepooling == "id":
            headinputdim = opt["embedding_size"]
            self.prepooling = lambda t: t
        elif prepooling == "linear":
            headinputdim = hidsize
            self.prepooling = torch.nn.Linear(opt["embedding_size"], hidsize)
        elif prepooling == "linearGELU":
            headinputdim = hidsize
            self.prepooling = torch.nn.Sequential(
                torch.nn.Linear(opt["embedding_size"], hidsize), torch.nn.GELU()
            )

        assert pooling in ["mean", "max"]
        self.pooling = pooling

        assert nlayers >= 0
        self.nlayers = nlayers
        if nlayers < 2:
            self.classifier_head = torch.nn.Linear(headinputdim, num_classes)
        else:
            self.classifier_head = torch.nn.Sequential(
                torch.nn.Linear(headinputdim, hidsize),
                *itertools.chain.from_iterable(
                    [
                        [torch.nn.GELU(), torch.nn.Linear(hidsize, hidsize)]
                        for _ in range(nlayers - 2)
                    ]
                ),
                torch.nn.GELU(),
                torch.nn.Linear(hidsize, num_classes),
            )

    def deep_decode(self, input, encoder_state):
        """
        Does the same as calling `self.decoder`, but without incr_state and returning
        all layers' intermediate results.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)

        tensor = self.decoder.forward_embedding(input, positions)
        tensor = self.decoder.dropout(tensor)

        all_tensors = [tensor]
        if getattr(self.decoder.layers, "is_model_parallel", False):
            chunks = PipelineHelper.split((tensor, encoder_output, encoder_mask))
            work_items = PipelineHelper.schedule_work_items(self.decoder.layers, chunks)
            chunk2layer2x = {i: {} for i, _ in enumerate(chunks)}

            for chunk_idx, layer_nos, next_device in work_items:
                s_tensor, s_enc_out, s_enc_mask = chunks[chunk_idx]
                for layer_no in layer_nos:
                    s_tensor, _ = self.decoder.layers[layer_no](
                        x=s_tensor,
                        encoder_output=s_enc_out,
                        encoder_mask=s_enc_mask,
                        incr_state=None,
                    )
                    assert layer_no not in chunk2layer2x[chunk_idx]
                    chunk2layer2x[chunk_idx][layer_no] = s_tensor
                chunks[chunk_idx] = PipelineHelper.chunk_to(
                    (s_tensor, s_enc_out, s_enc_mask), next_device
                )

            d = PipelineHelper.join([chunk2layer2x[i] for i, _ in enumerate(chunks)])
            for l, _ in enumerate(self.decoder.layers):
                all_tensors.append(d[l])
        else:
            all_tensors = []
            for layer in self.layers:
                tensor, _ = layer(
                    x=tensor,
                    encoder_output=encoder_output,
                    encoder_mask=encoder_mask,
                    incr_state=None,
                )
                all_tensors.append(tensor)

        if self.decoder.variant == "prelayernorm":
            all_tensors = [
                _normalize(t, self.decoder.norm_embeddings) for t in all_tensors
            ]

        moved_tensors = [t.to(self.layermix) for t in all_tensors]
        mixed = torch.sum(torch.stack(moved_tensors, dim=-1) * self.layermix, dim=-1)
        return mixed

    def enrich_potential_blender_state_dict_(self, state_dict):
        """
        Override to add in the classifier head if it doesn't exist. (This is the case on
        the initial load where we init with the big Blender model)
        """
        state_dict["layermix"] = self.layermix

        if isinstance(self.prepooling, torch.nn.Linear):
            for tensor in ["weight", "bias"]:
                key = f"prepooling.{tensor}"
                if key not in state_dict:
                    state_dict[key] = getattr(self.prepooling, tensor)
        elif isinstance(self.prepooling, torch.nn.Sequential):
            for tensor in ["weight", "bias"]:
                key = f"prepooling.0.{tensor}"
                if key not in state_dict:
                    state_dict[key] = getattr(self.prepooling[0], tensor)

        if self.nlayers < 2:
            for tensor in ["weight", "bias"]:
                key = f"classifier_head.{tensor}"
                if key not in state_dict:
                    state_dict[key] = getattr(self.classifier_head, tensor)
        else:
            for i in range(self.nlayers):
                for tensor in ["weight", "bias"]:
                    key = f"classifier_head.{2*i}.{tensor}"
                    if key not in state_dict:
                        state_dict[key] = getattr(self.classifier_head[2 * i], tensor)

    def forward(self, text_vec, label_vec):
        if self.nlayers == 0:
            return self.classifier_head(
                self.classifier_head.weight.new_zeros(
                    (text_vec.shape[0], self.classifier_head.weight.shape[-1])
                )
            )

        what_to_take = []
        # [bsz, seqlen, emb_dim], where seqlen is 1 if we don't use the decode
        encoder_states = self.encoder(text_vec)
        if self.with_encode:
            numbers, mask = encoder_states
            masked = numbers * mask.unsqueeze(-1).expand(-1, -1, numbers.shape[-1])
            what_to_take.append(masked)
        # full decode or only present single START token for all batch elements
        deep_decoder_states = self.deep_decode(
            label_vec
            if self.with_decode
            else self.START.detach().expand(text_vec.size(0), 1),
            encoder_states,
        )
        what_to_take.append(deep_decoder_states)

        # take what we want
        all_states = torch.cat(what_to_take, dim=1)

        # transform before pooling
        all_states = self.prepooling(all_states)

        if self.pooling == "mean":
            pooled = all_states.mean(dim=1)
        elif self.pooling == "max":
            pooled, _ = all_states.max(dim=1)

        return self.classifier_head(pooled)


class ClassifierOnGeneratorAgent(TransformerGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        # This is how we get the classifier args without inheriting directly!
        TransformerGeneratorAgent.add_cmdline_args(argparser)
        TorchClassifierAgent.add_cmdline_args(argparser)
        argparser.add_argument(
            "--init-model",
            type=str,
            default=None,
            help="Initialize model with weights from this file.",
        )
        # And now our own!
        agent = argparser.add_argument_group("ClassifierOnGeneratorAgent Arguments")
        agent.add_argument(
            "--freeze-enc-dec-weights",
            type=bool,
            default=False,
            help="Only train the classifier head and not the encoder and decoder",
        )
        agent.add_argument("--n-classifier-layers", type=int)
        agent.add_argument("--classifier-hidsize", type=int, default=512)
        agent.add_argument("--classifier-state-pooling")
        agent.add_argument("--classifier-state-pre-pooling")
        agent.add_argument("--classifier-with-encode", type=bool, default=True)
        agent.add_argument("--classifier-with-decode", type=bool, default=True)
        return agent

    def __init__(self, opt, shared=None):
        assert opt.get("classes_from_file") is None

        if not shared:
            self.class_list = opt["classes"]
            self.class_dict = {val: i for i, val in enumerate(self.class_list)}
            if opt.get("class_weights") is not None:
                self.class_weights = opt["class_weights"]
            else:
                self.class_weights = [1.0 for _ in self.class_list]
        else:
            self.class_list = shared["class_list"]
            self.class_dict = shared["class_dict"]
            self.class_weights = shared["class_weights"]

        self.nlayers = opt["n_classifier_layers"]  # 3
        self.hidsize = opt["classifier_hidsize"]  # 512
        self.pooling = opt["classifier_state_pooling"]  # max
        self.prepooling = opt["classifier_state_pre_pooling"]  # linearGELU
        self.with_encode = opt["classifier_with_encode"]  # True
        self.with_decode = opt["classifier_with_decode"]  # True

        super().__init__(opt, shared)

        if opt["freeze_enc_dec_weights"]:
            for param in itertools.chain(
                self.model.encoder.parameters(), self.model.decoder.parameters()
            ):
                param.requires_grad = False

        if shared is None:
            self.reset_metrics()

    def _update_confusion_matrix(self, predictions, gold_singletons):
        raise AssertionError()

    def batch_act(self, observations):
        # clear local metrics before anything else
        self._local_metrics.clear()

        # initialize a list of replies with this agent's id
        batch_reply = [
            Message({"id": self.getID(), "episode_done": False}) for _ in observations
        ]

        # check if there are any labels available, if so we will train on them
        self.is_training = any("labels" in obs for obs in observations)

        # create a batch from the vectors
        batch = self.batchify(observations)
        self.global_metrics.add("exps", GlobalTimerMetric(batch.batchsize))

        if self.is_training:
            # register the start of updates for later counting when they occur
            self.global_metrics.add("ups", GlobalTimerMetric(0))
            output = self.train_step(batch)
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back gradients.
                output = self.eval_step(batch)

        if output is not None:
            # local metrics are automatically matched up
            self.match_batch(batch_reply, batch.valid_indices, output)

        # Add bookkeeping in our confusion matrix
        predictions = [reply.get("text") for reply in batch_reply]
        if not all(x is None for x in predictions):
            # The label is expected to be in a list like in the "labels" or "eval_labels" fields
            golds = [obs.get("desired_class") for obs in observations]
            # Get this batch's certainties
            certainties = [obs.get("certainty") for obs in observations]
            # Start setup: collect, predicted class, gold class, and annotated certainty
            all_ipgcs = []
            assert len([p for p in predictions if p is not None]) == len(
                batch.valid_indices
            )
            assert len(predictions) == len(golds)
            assert len(predictions) == len(certainties)
            for i, p, g, c in zip(batch.valid_indices, predictions, golds, certainties):
                if p is None and g is None:
                    continue
                assert p is not None and g is not None and c is not None
                all_ipgcs.append((i, p, g, c))
            # ALL and split by EVA / IDK / TRY / YEA
            # f1_dict = {}
            assert all(c in ("EVA", "IDK", "TRY", "YEA") for _, _, _, c in all_ipgcs), [
                c for _, _, _, c in all_ipgcs
            ]
            for certainty in ("ALL", "EVA", "IDK", "TRY", "YEA"):
                _is = [i for i, _, _, c in all_ipgcs if certainty in (c, "ALL")]
                _ps = [p for i, p, _, c in all_ipgcs if certainty in (c, "ALL")]
                _gs = [g for i, _, g, c in all_ipgcs if certainty in (c, "ALL")]

                def record_partial_metric(k, values):
                    for i, value in zip(_is, values):
                        if "metrics" not in batch_reply[i]:
                            batch_reply[i]["metrics"] = {}
                        batch_reply[i]["metrics"][k] = value

                # accuracies/F1s/raw counts
                prec, recall, f1 = ConfusionMatrixMetric.compute_metrics(
                    _ps, _gs, "RIGHT"
                )
                record_partial_metric(f"{certainty}_RIGHT_prec", prec)
                record_partial_metric(f"{certainty}_RIGHT_recall", recall)
                record_partial_metric(f"{certainty}_RIGHT_f1", f1)
                record_partial_metric(
                    f"{certainty}_accuracy",
                    [AverageMetric(int(p == g)) for p, g in zip(_ps, _gs)],
                )
                record_partial_metric(
                    f"{certainty}_RIGHT_TP",
                    [
                        SumMetric(int(p == "RIGHT" and g == "RIGHT"))
                        for p, g in zip(_ps, _gs)
                    ],
                )
                record_partial_metric(
                    f"{certainty}_RIGHT_FP",
                    [
                        SumMetric(int(p == "RIGHT" and g == "WRONG"))
                        for p, g in zip(_ps, _gs)
                    ],
                )
                record_partial_metric(
                    f"{certainty}_RIGHT_TN",
                    [
                        SumMetric(int(p == "WRONG" and g == "WRONG"))
                        for p, g in zip(_ps, _gs)
                    ],
                )
                record_partial_metric(
                    f"{certainty}_RIGHT_FN",
                    [
                        SumMetric(int(p == "WRONG" and g == "RIGHT"))
                        for p, g in zip(_ps, _gs)
                    ],
                )

        # broadcast the metrics back
        for k, values in self._local_metrics.items():
            if len(values) != len(batch.valid_indices):
                raise IndexError(
                    f"Batchsize mismatch on metric {k} (got {len(values)}, "
                    f"expected {len(batch.valid_indices)}"
                )
            for i, value in zip(batch.valid_indices, values):
                if "metrics" not in batch_reply[i]:
                    batch_reply[i]["metrics"] = {}
                batch_reply[i]["metrics"][k] = value

        # register the end of timers
        endtimer = GlobalTimerMetric(0)
        self.global_metrics.add("exps", endtimer)
        if (
            "label_vec" in batch
            and "text_vec" in batch
            and batch.label_vec is not None
            and batch.text_vec is not None
        ):
            self.global_metrics.add("ltps", GlobalTimerMetric(0))
            self.global_metrics.add("ctps", GlobalTimerMetric(0))
            self.global_metrics.add("tps", GlobalTimerMetric(0))

        return batch_reply

    def build_model(self, states=None):
        model = ClassifierOnGeneratorModel(
            self.opt,
            self.dict,
            num_classes=len(self.class_list),
            nlayers=self.nlayers,
            hidsize=self.hidsize,
            pooling=self.pooling,
            prepooling=self.prepooling,
            with_encode=self.with_encode,
            with_decode=self.with_decode,
        )

        self.enrich_potential_blender_state_dict_ = lambda sd: model.enrich_potential_blender_state_dict_(
            sd
        )

        return model

    def build_criterion(self):
        weight_tensor = torch.FloatTensor(self.class_weights)
        if not self.fp16:
            return torch.nn.CrossEntropyLoss(weight=weight_tensor, reduction="none")
        else:
            # FP16 safe cross entropy (softmax done in FP32)
            return FP16SafeCrossEntropy(weight=weight_tensor, reduction="none")

    def load_state_dict(self, state_dict):
        self.enrich_potential_blender_state_dict_(state_dict)
        super().load_state_dict(state_dict)

    def share(self):
        """
        Share model parameters.
        """
        shared = super().share()
        shared["class_dict"] = self.class_dict
        shared["class_list"] = self.class_list
        shared["class_weights"] = self.class_weights
        return shared

    def batchify(self, obs_batch, sort=False):
        # Sorting would make it hard to line up the observations within one batch
        assert sort is False
        base_batch = super().batchify(obs_batch, sort)
        desired_classes = [
            obs["desired_class"] for obs in obs_batch if self.is_valid(obs)
        ]
        assert len(desired_classes) == len(base_batch.text_vec)
        return AttrDict(desired_classes=desired_classes, **base_batch.__dict__)

    def _get_classes_tensor(self, batch):
        try:
            classes_tensor = torch.LongTensor(
                [self.class_dict[c] for c in batch.desired_classes]
            )
            if self.use_cuda:
                classes_tensor = classes_tensor.cuda()
            return classes_tensor
        except KeyError as e:
            warn_once("One of your classes is not in the class list.")
            raise e

    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        if batch.text_vec is None:
            raise AssertionError()
            return Output()
        self.model.train()
        self.zero_grad()

        # Calculate loss
        scores = self.score(batch)
        loss = self.criterion(scores, self._get_classes_tensor(batch))
        self.record_local_metric("loss", AverageMetric.many(loss))
        loss = loss.mean()
        self.backward(loss)
        self.update_params()

        # Get predictions
        _, prediction_id = torch.max(scores.float().cpu(), 1)
        preds = [self.class_list[idx] for idx in prediction_id]

        return Output(preds)

    def eval_step(self, batch):
        """
        Train on a single batch of examples.
        """
        if batch.text_vec is None:
            raise AssertionError()
            return

        self.model.eval()
        scores = self.score(batch)
        probs = torch.nn.functional.softmax(scores, dim=1)
        _, prediction_id = torch.max(probs.float().cpu(), 1)
        preds = [self.class_list[idx] for idx in prediction_id]

        if batch.labels is not None and not self.opt["ignore_labels"]:
            labels = self._get_classes_tensor(batch)
            loss = self.criterion(scores, labels)
            self.record_local_metric("loss", AverageMetric.many(loss))

        return Output(
            preds,
            maxprob=[x.item() for x in probs.max(dim=-1)[0]],
            posprob=[x.item() for x in probs[:, self.class_list.index("RIGHT")]],
        )

    def score(self, batch):
        return self.model(batch.text_vec, batch.label_vec)


class BertCalibrator(BertClassifierAgent):
    def _get_classes_tensor(self, batch):
        try:
            classes_tensor = torch.LongTensor(
                [self.class_dict[c] for c in batch.labels]
            )
            if self.use_cuda:
                classes_tensor = classes_tensor.cuda()
            return classes_tensor
        except KeyError as e:
            warn_once("One of your classes is not in the class list.")
            raise e

    def eval_step(self, batch):
        """
        Train on a single batch of examples.
        """
        if batch.text_vec is None:
            raise AssertionError()
            return

        self.model.eval()
        scores = self.score(batch)
        probs = torch.nn.functional.softmax(scores, dim=1)
        _, prediction_id = torch.max(probs.float().cpu(), 1)
        preds = [self.class_list[idx] for idx in prediction_id]

        if batch.labels is not None and not self.opt["ignore_labels"]:
            labels = self._get_classes_tensor(batch)
            loss = self.criterion(scores, labels)
            self.record_local_metric("loss", AverageMetric.many(loss))

        return Output(
            preds,
            maxprob=[x.item() for x in probs.max(dim=-1)[0]],
            posprob=[x.item() for x in probs[:, self.class_list.index("RIGHT")]],
        )


if __name__ == "__main__":
    with open(f"{MCDIR}/calibrator_training_answers.txt", 'w') as f:
        print(
            "\n".join(
                [
                    d["label"]
                    for d in claim_from_training_data(
                        balance_correctness="anycorrectness",
                        certainty_distribution="everything-oversample",
                        with_eva=False,
                        from_the_back=True,
                        training_samples=TriviaQARun.get_run(
                            f"{DATADIR2}/NoEvidenceUnion_blender_3B_default_trainset_withembeddings_cleanedanswers_triviaqa:NoEvidenceUnion_replies.jsonl",
                            no_cache=True,
                        ).samples[:50000],
                        training_certainty_samples=TriviaQARun.get_run(
                            f"{DATADIR2}/triviaqa_full_166_parlai.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl",
                            no_cache=True,
                        ).samples[:50000],
                    )
                ]
            ),
            file=f,
        )
