#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile

import csv
from itertools import islice
from pathlib import Path
import os
import json
import re
import tqdm

DEBUG_MISSING_RAW_CONVERSATIONS = False  # Unnecessary once Amazon fixes multidogo

RESOURCE = DownloadableFile(
    "https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset/archive/master.zip",
    "raw_data.zip",
    "fb59c7261da2d30d9d24b9af309ebb4bf0e5b39f97d718201a7160e591e76a3c",
    zipped=True,
)

RAW_DATA_PREFIX = "multi-domain-goal-oriented-dialogues-dataset-master/data/"

RAW_DATA_ANNOTATED_DATA_PATH = "paper_splits"
RAW_DATA_UNANNOTATED_DATA_PATH = "unannotated"

TURN_INTENT = "turn"
SENTENCE_INTENT = "sentence"
TURN_AND_SENTENCE_INTENT = "both"

RAW_DATA_SENTENCE_INTENT_PATH = "splits_annotated_at_sentence_level"
RAW_DATA_TURN_INTENT_PATH = "splits_annotated_at_turn_level"

RAW_DATA_INTENT_BY_TYPE_PATH = {
    TURN_INTENT: RAW_DATA_TURN_INTENT_PATH,
    SENTENCE_INTENT: RAW_DATA_SENTENCE_INTENT_PATH,
}

DOMAINS = ["airline", "fastfood", "finance", "insurance", "media", "software"]

DATATYPE_TO_RAW_DATA_FILE_NAME = {
    "test": "test.tsv",
    "train": "train.tsv",
    "valid": "dev.tsv",
}

PROCESSED = "processed/"


def _preprocess(opt, datapath, datatype, version):
    """
    MultiDoGo conversations take place between an "agent" and a customer". Labeled
    customer data is stored in one set of files while the agent data is in another.
    There is a common conversation ID between the two, but the conversations are not
    listed in a consistent way between the documents. Since we'll have to do work to
    associate the data between the files anyway, we might as well process the data into
    a new file that'll be easier to deal with.

    Stores the data as <multidogo_data_path>/processed/<domain>/<datatype>.txt.
    Will skip preprocessing if this file already exists.
    """
    domains = opt.get("domains", DOMAINS)
    intent_type = opt.get("intent_type", TURN_INTENT)

    for domain in domains:
        out_dir = get_processed_multidogo_folder(
            datapath, domain, datatype, intent_type
        )
        if build_data.built(out_dir, version):
            continue
        print(
            f"    Preprocessing '{domain}' data for '{datatype}' with '{intent_type}' intent labels."
        )

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # The agent responses for *all* datatypes are in one file.
        # We need to iterate through the datatype file to know which lines
        # we'll actually need... so build a quick lookup table to know which
        # lines in the tsv file we'll need to care about so we're not scanning
        # through the whole thing a bunch
        unannotated_id_map = _build_conversation_span_map(
            _get_unannotated_tsv_data(datapath, domain)
        )

        # Actually do the work of collating all of the conversations + annotations
        # For turn + sentence intent labels, we do two passes, one for sentence
        # then one for turn so that we do not add two sets of labels for the
        # same conversation ID. We can use this forced structure to do the
        # separate categories of turn intent and sentence intent labels.  We
        # also do a bit of chuking
        file_idx = 0
        seen_conversations_set = set()
        if intent_type == TURN_AND_SENTENCE_INTENT or intent_type == SENTENCE_INTENT:
            file_idx, seen_conversations_set = _aggregate_and_write_conversations(
                intent_type,
                SENTENCE_INTENT,
                datapath,
                domain,
                datatype,
                unannotated_id_map,
                start_file_idx=file_idx,
                skip_ids=set(),
            )

        if intent_type == TURN_AND_SENTENCE_INTENT or intent_type == TURN_INTENT:
            _, _ = _aggregate_and_write_conversations(
                intent_type,
                TURN_INTENT,
                datapath,
                domain,
                datatype,
                unannotated_id_map,
                start_file_idx=file_idx,
                skip_ids=seen_conversations_set,
            )

        # mark that we've built this combinations
        build_data.mark_done(out_dir, version_string=version)


def get_processed_multidogo_folder(datapath, domain, datatype, intent_type):
    return os.path.join(datapath, PROCESSED, domain, intent_type, datatype)


# unannotated data is UNANNOTATED_DATA_PROFIX + <domain> + '.tsv'
# annotated data is ANNOTATED_DATA_PATH + <annotations type> + <domain> + '/' + <datatype> + '.tsv'
def _get_unannotated_tsv_data(datapath, domain):
    file_name = os.path.join(
        datapath, RAW_DATA_PREFIX, RAW_DATA_UNANNOTATED_DATA_PATH, domain + ".tsv"
    )
    return csv.reader(open(file_name, "r"), delimiter=",")  # comma-separated tsv, lol


def _get_annotated_tsv_data(datapath, domain, datatype, annotation_type):
    file_name = os.path.join(
        datapath,
        RAW_DATA_PREFIX,
        RAW_DATA_ANNOTATED_DATA_PATH,
        RAW_DATA_INTENT_BY_TYPE_PATH[annotation_type],
        domain,
        DATATYPE_TO_RAW_DATA_FILE_NAME[datatype],
    )
    return csv.reader(open(file_name, "r"), delimiter="\t")


def _get_annotated_tsv_data_size(datapath, domain, datatype, annotation_type):
    file_name = os.path.join(
        datapath,
        RAW_DATA_PREFIX,
        RAW_DATA_ANNOTATED_DATA_PATH,
        RAW_DATA_INTENT_BY_TYPE_PATH[annotation_type],
        domain,
        DATATYPE_TO_RAW_DATA_FILE_NAME[datatype],
    )
    return sum(1 for line in open(file_name, 'r'))


def _build_conversation_span_map(unannotated_tsv_object):
    result = {}  # conversationId to (start line, length) map
    start = 0
    prev_conversation_id = ""
    length = 0
    for i, row in enumerate(unannotated_tsv_object):
        conversation_id = row[0][
            4:-2
        ]  # do substring cause conversationId has extra filler in unannotated
        if conversation_id != prev_conversation_id:
            result[prev_conversation_id] = (start, length)
            start = i
            prev_conversation_id = conversation_id
            length = 0
        length += 1
    result[conversation_id] = (start, length)
    return result


def _get_slots_map(utterance, slot_string):
    values = slot_string.split(" ")
    cleaned = re.sub(r"[^\w\s]", "", utterance)
    words = cleaned.split(" ")
    result = {}
    for i in range(len(words)):
        if values[i] != "O":
            result[values[i]] = words[i]
    return result


def _aggregate_and_write_conversations(
    raw_intent_type,
    fetch_intent_type,
    datapath,
    domain,
    datatype,
    unannotated_id_map,
    skip_ids,
    start_file_idx=0,
):
    conversations_to_write = {}  # conversationId -> list of turns
    seen_conversations = set()
    out_dir = get_processed_multidogo_folder(
        datapath, domain, datatype, raw_intent_type
    )
    file_idx = start_file_idx
    intent_tsv = _get_annotated_tsv_data(datapath, domain, datatype, fetch_intent_type)
    next(intent_tsv)  # don't need the header in the first line
    print(f"Processing for {domain}, {fetch_intent_type}, {datatype}")
    for labeled_line in tqdm.tqdm(
        intent_tsv,
        total=_get_annotated_tsv_data_size(
            datapath, domain, datatype, fetch_intent_type
        )
        - 1,
    ):
        conversation_id = labeled_line[0]
        if conversation_id in skip_ids:
            continue
        if conversation_id not in seen_conversations:
            # new conversation, add text of conversation to conversations_to_write
            conversations_to_write[conversation_id] = {}
            found_raw_conversation = _add_utterances(
                unannotated_id_map,
                conversation_id,
                conversations_to_write,
                datapath,
                domain,
            )
            seen_conversations.add(conversation_id)
            if not found_raw_conversation:
                if DEBUG_MISSING_RAW_CONVERSATIONS:
                    print(f"Could not find raw conversations for {conversation_id}")
                skip_ids.add(conversation_id)
                conversations_to_write.pop(conversation_id, None)
                continue
        if fetch_intent_type == SENTENCE_INTENT:
            _get_sentence_labels_and_slots_map(labeled_line, conversations_to_write)
        elif fetch_intent_type == TURN_INTENT:
            _get_turn_labels_and_slots_map(labeled_line, conversations_to_write)
        else:
            raise KeyError(
                "Invalid `fetch_intent_type`. This case should never be hit. Something is broken in the `build.py` file."
            )
    # Don't forget to dump out last file
    with open(f"{out_dir}/{file_idx}.json", "w+") as out_file:
        json.dump(conversations_to_write, out_file, indent=4)
        file_idx += 1
    # Return necessary outputs for next pass
    return file_idx, seen_conversations


def _add_utterances(
    unannotated_id_map, conversation_id, conversations_to_write, datapath, domain
):
    try:
        start, length = unannotated_id_map[conversation_id]
    except KeyError:
        return False
    conversation_text = islice(
        _get_unannotated_tsv_data(datapath, domain), start, start + length
    )

    for line in conversation_text:
        # Format of unannotated: conversationId,turnNumber,utteranceId,utterance,authorRole
        conversations_to_write[conversation_id] = {
            **conversations_to_write[conversation_id],
            int(line[1]): {"text": line[3], "role": line[4]},
        }
    return True


def _get_sentence_labels_and_slots_map(labeled_line, output):
    # Sentence tsv format: conversationId   turnNumber  sentenceNumber  utteranceId utterance   slot-labels intent
    conversation_id = labeled_line[0]
    turn_number = int(float(labeled_line[1]))  # cause a few got saved as float.
    if conversation_id not in output:
        raise RuntimeError("Should never happen; raw conversation text should be here")
    if turn_number not in output[conversation_id]:
        output[conversation_id][turn_number] = {}
    output[conversation_id][turn_number] = {
        **output[conversation_id][turn_number],
        "slots": _get_slots_map(labeled_line[4], labeled_line[5]),
    }
    if "intents" not in output[conversation_id][turn_number]:
        output[conversation_id][turn_number]["intents"] = []
    output[conversation_id][turn_number]["intents"].append(labeled_line[6])


def _get_turn_labels_and_slots_map(labeled_line, output):
    # Turn tsv format: conversationId  turnNumber  utteranceId utterance   slot-labels intent
    conversation_id = labeled_line[0]
    turn_number = int(float(labeled_line[1]))  # cause a few got saved as float
    if conversation_id not in output:
        raise RuntimeError("Should never happen; raw conversation text should be here")
    if turn_number not in output[conversation_id]:
        output[conversation_id][turn_number] = {}
    output[conversation_id][turn_number] = {
        **output[conversation_id][turn_number],
        "slots": _get_slots_map(labeled_line[3], labeled_line[4]),
        "intents": [labeled_line[5]],
    }


def build(opt):
    # get path to data directory
    datapath = os.path.join(opt["datapath"], "multidogo")
    # define version if any
    version = "v1.1"

    # check if data had been previously downloaded
    if not build_data.built(datapath, version_string=version):
        print("[building data: " + datapath + "]")

        # make a clean directory if needed
        if build_data.built(datapath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(datapath)
        build_data.make_dir(datapath)

        # Download the data.
        RESOURCE.download_file(datapath)

        # mark the data as built
        build_data.mark_done(datapath, version_string=version)

    # do preprocessing on the data to put it into FBDialogueData format. There's a lot so check to make sure it's okay
    for fold in ["train", "valid", "test"]:
        _preprocess(opt, datapath, fold, version)
