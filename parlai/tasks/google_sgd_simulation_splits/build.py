#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.tasks.google_sgd.build as original_google_sgd_build
import parlai.core.build_data as build_data

import os
import json
import random
from shutil import copyfile

DATA_LEN = original_google_sgd_build.DATA_LEN

MODEL_MODEL_HOLDOUT_DOMAINS = [
    "Homes_1",
    "Homes_2",
    "FindHomeByArea",
    "RentalCars_1",
    "RentalCars_2",
    "RentalCars_3",
    "Messaging_1",
    "Payment_1",
]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "google_sgd_rl_splits")
    # define version if any
    version = "1.0"

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print("[building data: " + dpath + "]")

        # Grab things from the original Google SGD
        original_google_sgd_build.build(opt)

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        model_model_convos = []
        model_model_schemas = {}
        tot_count = 0
        for split_type in ["train", "dev", "test"]:
            outpath = os.path.join(dpath, split_type)
            os.makedirs(outpath, exist_ok=True)
            original_path = os.path.join(opt["datapath"], "google_sgd", split_type)

            copyfile(
                os.path.join(original_path, "schema.json"),
                os.path.join(outpath, "schema.json"),
            )
            with open(os.path.join(original_path, "schema.json")) as f:
                schemas = json.load(f)
            for schema in schemas:
                model_model_schemas[schema["service_name"]] = schema

            for file_id in range(1, DATA_LEN[split_type] + 1):
                filename = f"dialogues_{file_id:03d}.json"
                original_file = os.path.join(original_path, filename)
                with open(original_file) as f:
                    blobs = json.load(f)
                save_data = []
                for blob in blobs:
                    blob[
                        "dialogue_id"
                    ] = f"{blob['dialogue_id']}_from_{split_type}_{file_id:03d}"
                    in_model_model = False
                    for service in blob["services"]:
                        if service in MODEL_MODEL_HOLDOUT_DOMAINS:
                            in_model_model = True
                            tot_count += 1
                    if in_model_model:
                        model_model_convos.append(blob)
                    else:
                        save_data.append(blob)
                with open(os.path.join(outpath, filename), "w+") as f:
                    json.dump(save_data, f, indent=4)
                print(split_type, filename)

        print("done processing train + dev + test ")
        # deal with custom splits
        print(
            "number of samples in homes + rental cars + messaging + payments",
            len(model_model_convos),
        )
        print("service usage count of above domains", tot_count)
        model_model_path = os.path.join(dpath, "model_model_splits")
        os.makedirs(model_model_path, exist_ok=True)
        random.Random(42).shuffle(model_model_convos)

        def save_model_model(convos, split_type, model_model_path, schema):
            os.makedirs(os.path.join(model_model_path, split_type), exist_ok=True)
            for i in range(int(len(convos) / 64) + 1):
                with open(
                    os.path.join(
                        model_model_path, split_type, f"dialogues_{i:03d}.json"
                    ),
                    "w+",
                ) as f:
                    json.dump(convos[i * 64 : (i + 1) * 64], f, indent=4)
            with open(
                os.path.join(model_model_path, split_type, "schema.json"), "w+"
            ) as f:
                json.dump(list(schema.values()), f, indent=4)

        save_model_model(
            model_model_convos[: int(0.6 * len(model_model_convos))],
            "train",
            model_model_path,
            model_model_schemas,
        )
        save_model_model(
            model_model_convos[
                int(0.6 * len(model_model_convos)) : int(0.8 * len(model_model_convos))
            ],
            "dev",
            model_model_path,
            model_model_schemas,
        )
        save_model_model(
            model_model_convos[int(0.8 * len(model_model_convos)) :],
            "test",
            model_model_path,
            model_model_schemas,
        )

        print("done processing test")

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
