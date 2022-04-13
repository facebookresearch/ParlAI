#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.build import DownloadableFile, build_data
from parlai.utils import logging
from parlai.tasks.triviqa.build import build as triviaqa_build

import itertools
import json
import os


RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/metacognition/metacognition_data_public_0413.zip',
        'metacognition_data_public_0413.zip',
        'd0ae89defe2fd0b0a4221eaa642a457d7d40cef475f54798119c7f3b8dd9361d',
    )
]


def build(opt):
    # First check that TriviaQA data is built
    triviaqa_build(opt)

    # Download data
    version = "v1.0"
    datapath = os.path.join(opt["datapath"], "metacognition")

    if build_data.built(datapath, version):
        # Data is already built
        return

    logging.info("building data: " + datapath)
    if build_data.built(datapath):
        # An older version exists, so remove these outdated files.
        build_data.remove_dir(datapath)
    build_data.make_dir(datapath)

    # Download the data.
    for downloadable_file in RESOURCES:
        downloadable_file.download_file(datapath)

    # Mark the data as built.
    build_data.mark_done(datapath, version)

    # Next, format the data

    # Establish link with trivia QA

    with open(opt["datapath"] + "/TriviaQA/qa/noevidence-union-dev.json", "rt") as f:
        data = json.load(f)["Data"]

    with open(opt["datapath"] + "/TriviaQA/qa/noevidence-union-train.json", "rt") as f:
        data += json.load(f)["Data"]

    all_strings = set(
        itertools.chain(
            *[
                [d["Question"], d["Answer"]["Value"]] + d["Answer"]["Aliases"]
                for d in data
            ]
        )
    )

    index2string = sorted(list(all_strings))

    # Decide direction
    link = index2string

    # Perform conversion
    for fn in [
        "/NewParlAITriviaQA/bert_calibrator_0020_bert-qp_says_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl"
    ]:
        outlines = []
        with open(datapath + fn) as f:
            for l in f.read().split("\n"):
                if not l:
                    continue
                o = json.loads(l)
                if "text" not in o["dialog"][0][0]:
                    continue
                q, a = o["dialog"][0][0]["text"].split("\n")
                if link == index2string:
                    q = int(q)
                o["dialog"][0][0]["text"] = str(link[q]) + "\n" + a
                outlines.append(json.dumps(o))
        with open(datapath + fn, "w") as f:
            print("\n".join(outlines), file=f)

    for fn in [
        "/NewParlAITriviaQA/bert_calibrator_0020_bert-q_says_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
        "/NewParlAITriviaQA/probe_ametsub_446_says_3x4000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
        "/NewParlAITriviaQA/probe_ametsub_446_says_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
        "/NewParlAITriviaQA/probe_ametsub_446_says_no_both_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
        "/NewParlAITriviaQA/probe_ametsub_446_says_no_dec_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
        "/NewParlAITriviaQA/probe_ametsub_446_says_no_enc_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
        "/NewParlAITriviaQA/probe_luca_04b_says_no_enc_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
        "/NewParlAITriviaQA/probe_luca_5b2_says_no_dec_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
        "/NewParlAITriviaQA/probe_luca_76f_says_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
        "/NewParlAITriviaQA/probe_luca_bb4_says_no_both_3x5000_blender3B_test_parlai.projects.metacognition.agents:CorrectnessProbingTeacher_replies.jsonl",
    ]:
        outlines = []
        with open(datapath + fn) as f:
            for l in f.read().split("\n"):
                if not l:
                    continue
                o = json.loads(l)
                if "text" not in o["dialog"][0][0]:
                    continue
                o["dialog"][0][0]["text"] = link[o["dialog"][0][0]["text"]]
                outlines.append(json.dumps(o))
        with open(datapath + fn, "w") as f:
            print("\n".join(outlines), file=f)

    for fn in [
        "/NewParlAITriviaQA/triviaqa_full_166_valid_blender_3B_default_withembeddings_cleanedanswers_parlai.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl",
        "/NewParlAITriviaQA/triviaqa_full_166_valid_blender_3B_freebeam_withembeddings_cleanedanswers_parlai.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl",
    ]:
        outlines = []
        with open(datapath + fn) as f:
            for l in f.read().split("\n"):
                if not l:
                    continue
                o = json.loads(l)
                if "text" not in o["dialog"][0][0]:
                    continue
                q, a = o["dialog"][0][0]["text"].split("\n")
                if link == index2string:
                    q = int(q)
                o["dialog"][0][0]["text"] = str(link[q]) + "\n" + a
                outlines.append(json.dumps(o))
        with open(datapath + fn, "w") as f:
            print("\n".join(outlines), file=f)

    for fn in [
        "/NewParlAITriviaQA/triviaqa_full_166_valid_finetuned_sabrina_342_forced_IDK_parlai.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl",
        "/NewParlAITriviaQA/triviaqa_full_166_valid_finetuned_sabrina_342_forced_TRY_parlai.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl",
        "/NewParlAITriviaQA/triviaqa_full_166_valid_finetuned_sabrina_342_forced_YEA_parlai.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl",
    ]:
        outlines = []
        with open(datapath + fn) as f:
            for l in f.read().split("\n"):
                if not l:
                    continue
                o = json.loads(l)
                if "text" not in o["dialog"][0][0]:
                    continue
                s = o["dialog"][0][0]["text"]
                if " <IDK>" in s:
                    delim = " <IDK>"
                elif " <TRY>" in s:
                    delim = " <TRY>"
                elif " <YEA>" in s:
                    delim = " <YEA>"
                else:
                    assert False
                q, a = s.split(delim)
                if link == index2string:
                    q = int(q)
                o["dialog"][0][0]["text"] = str(link[q]) + delim + a
                outlines.append(json.dumps(o))
        with open(datapath + fn, "w") as f:
            print("\n".join(outlines), file=f)

    for fn in [
        "/NewParlAITriviaQA/triviaqa_full_166_valid_finetuned_sabrina_342_unforced_parlai.projects.metacognition.agents:CertaintyOntoTriviaQATeacher_replies.jsonl"
    ]:
        outlines = []
        with open(datapath + fn) as f:
            for l in f.read().split("\n"):
                if not l:
                    continue
                o = json.loads(l)
                if "text" not in o["dialog"][0][0]:
                    continue
                q, a = o["dialog"][0][0]["text"].split("  <SAME>")
                if link == index2string:
                    q = int(q)
                o["dialog"][0][0]["text"] = str(link[q]) + "  <SAME>" + a
                outlines.append(json.dumps(o))
        with open(datapath + fn, "w") as f:
            print("\n".join(outlines), file=f)

    for fn in [
        "/_copy_over_projects_metacognition/3x1000_blender3B_test_fourmodels.jsonl",
        "/_copy_over_projects_metacognition/3x4000_blender3B_test_fourmodels.jsonl",
        "/_copy_over_projects_metacognition/3x5000_blender3B_test_fourmodels.jsonl",
        "/_copy_over_projects_metacognition/3x5000_blender3B_test_fourmodels.non_simplified.jsonl",
    ]:
        outlines = []
        with open(datapath + fn) as f:
            for l in f.read().split("\n"):
                if not l:
                    continue
                o = json.loads(l)
                o["question"] = link[o["question"]]
                outlines.append(json.dumps(o))
        with open(datapath + fn, "w") as f:
            print("\n".join(outlines), file=f)

    for fn in [
        "/_copy_over_projects_metacognition/annotations/validset/1x2000_blender3B_train.majorities.simplified_annotations.json",
        "/_copy_over_projects_metacognition/annotations/validset/3x2000_blender3B_valid.majorities.simplified_annotations.json",
        "/_copy_over_projects_metacognition/annotations/validset/3x5000_blender3B_test.majorities.non_simplified_annotations.json",
        "/_copy_over_projects_metacognition/annotations/validset/3x5000_blender3B_test.majorities.simplified_annotations.json",
    ]:
        with open(datapath + fn) as f:
            o = json.load(f)
            for d in o["Data"]:
                d["question"] = link[d["question"]]
        with open(datapath + fn, "w") as f:
            json.dump(o, f)

    for fn in [
        "/_copy_over_projects_metacognition/annotations/validset/3x2000_blender3B_valid.json"
    ]:
        with open(datapath + fn) as f:
            o = json.load(f)
            for worker in o:
                nd = {}
                for k, v in o[worker].items():
                    q, a = k.split("#=%=#")
                    if link == index2string:
                        q = int(q)
                    nd[str(link[q]) + "#=%=#" + a] = v
                o[worker] = nd
        with open(datapath + fn, "w") as f:
            json.dump(o, f)

    for fn in [
        "/_copy_over_projects_metacognition/webapp/src/static/blender3B_all_four_dedup_test.jsonl",
        "/_copy_over_projects_metacognition/webapp/src/static/blender3B_forced_IDK_test.jsonl",
        "/_copy_over_projects_metacognition/webapp/src/static/blender3B_forced_TRY_test.jsonl",
        "/_copy_over_projects_metacognition/webapp/src/static/blender3B_forced_YEA_test.jsonl",
        "/_copy_over_projects_metacognition/webapp/src/static/blender3B_train.jsonl",
        "/_copy_over_projects_metacognition/webapp/src/static/blender3B_valid.jsonl",
    ]:
        outlines = []
        with open(datapath + fn) as f:
            for l in f.read().split("\n"):
                if not l:
                    continue
                o = json.loads(l)
                o["question"] = link[o["question"]]
                o["golds"] = [link[s] for s in o["golds"]]
                outlines.append(json.dumps(o))
        with open(datapath + fn, "w") as f:
            print("\n".join(outlines), file=f)
