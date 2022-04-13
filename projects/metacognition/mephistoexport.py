#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import defaultdict

from mephisto.core.local_database import LocalMephistoDB
from mephisto.core.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.data_model.worker import Worker


INFILE = "parlai/projects/metacognition/webapp/src/static/blender3B_all_four_dedup_test.jsonl"
OUTFILE = "parlai/projects/metacognition/file_you_can_send_me_need3.json"

# GIMME = "approved"
GIMME = "completed"


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
            print('Assertion error')
            continue
        if data["status"] != GIMME:
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


qp2as = {}

INFILE = "/some/storage/ParlAI/parlai/projects/metacognition/webapp/src/static/blender3B_all_four_dedup_test.jsonl"
OUTFILE = (
    "/some/storage/ParlAI/parlai/projects/metacognition/mephisto_intermediate.json"
)

with open(INFILE) as f:
    for d in f.read().splitlines():
        d = json.loads(d)
        qp = d["question"] + "#=%=#" + d["prediction"]
        qp2as[qp] = []

i = 0
for worker, qp2a in all_mturk_worker_annotations(
    "metacognition_blender3b_test_4dedupx3x5000_need3"
).items():
    i += 1
    for qp, a in qp2a.items():
        if qp in qp2as:
            qp2as[qp].append([worker, a])

print(f"Total workers: {i}")

with open(OUTFILE, 'w') as f:
    json.dump(
        all_mturk_worker_annotations(
            "metacognition_blender3b_test_4dedupx3x5000_need3"
        ),
        f,
    )
    # json.dump({"qp2as": qp2as}, f)
