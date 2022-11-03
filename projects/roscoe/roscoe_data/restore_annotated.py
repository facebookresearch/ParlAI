#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Restore data with reasoning chains.
"""
import json
import csv
import os
import re
import xmltodict

from parlai.core.params import ParlaiParser

PATH_TO_DATA = "./projects/roscoe/roscoe_data/raw/"
PATH_TO_GENERATIONS = "./projects/roscoe/roscoe_data/generated/"

DATASETS = ["drop", "esnli", "cosmos", "semevalcommonsense", "gsm8k"]


def write_to_file(list_of_json_dict, output_path):
    with open(output_path, 'w') as outfile:
        for line in list_of_json_dict:
            json.dump(line, outfile)
            outfile.write('\n')


def parse_drop(fn, reasoning, savefile):
    json_lines = []
    reasonings = {}
    with open(reasoning, "r") as f:
        json_lines = json.load(f)
        print(len(json_lines))

    for json_line in json_lines:
        reasonings[json_line["key"]] = json_line

    structs = []

    with open(fn, "r") as json_file:

        alldata = json.load(json_file)

        for k, v in alldata.items():

            questions = v['qa_pairs']
            count = 0
            for q in questions:
                struct = {}
                count += 1
                question = q['question']
                answer = q['answer']
                number = answer['number']
                span = answer['spans']
                if len(number) > 0:
                    struct["key"] = k + "\t" + q["query_id"]
                    if len(v["passage"].split(' ')) < 200:
                        struct["premise"] = v["passage"]
                        struct["premise"] = v["passage"] + " " + question
                        struct["hypothesis"] = number
                        struct["answer"] = "yes"
                elif len(span) > 0:
                    struct["key"] = k + "\t" + q["query_id"]
                    if len(v["passage"].split(' ')) < 200:
                        struct["premise"] = v["passage"]
                        struct["premise"] = v["passage"] + " " + question
                        struct["hypothesis"] = span[0]
                        struct["answer"] = "yes"

                if "premise" in struct:
                    if struct["key"] in reasonings:
                        reasoning = reasonings[struct["key"]]["reasoning"]
                        struct["gpt-3"] = reasoning
                        structs.append(struct)
                    else:
                        val = struct["key"]
                        print(f"No reasoning found for key {val}")

                if count > 2:
                    break

            ## we used the first 250 of these
            if len(structs) > 250:
                break

    write_to_file(structs, savefile)


def parse_esnli(fn, reasoning, savefile):

    json_lines = []
    reasonings = {}
    with open(reasoning, "r") as f:
        json_lines = json.load(f)

    for json_line in json_lines:
        reasonings[json_line["key"]] = json_line

    with open(fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')

        c = 0
        structs = []
        for line in csv_reader:
            if c == 0:
                c += 1
                continue

            struct = {}
            answer = line[1]
            if answer == "entailment":
                struct['answer'] = "yes"
            elif answer == "contradiction":
                struct['answer'] = "no"
            else:
                continue
            struct["key"] = line[0]
            struct["premise"] = line[2]
            struct["hypothesis"] = line[3]
            struct['explanation_1'] = line[4]
            struct['explanation_2'] = line[9]
            struct['explanation_3'] = line[14]

            if struct["key"] in reasonings:
                reasoning = reasonings[struct["key"]]["reasoning"]
                struct["gpt-3"] = reasoning
                structs.append(struct)
            else:
                val = struct["key"]
                print(f"No reasoning found for key {val}")

            if len(structs) > 150:
                break

    write_to_file(structs, savefile)


def parse_cosmos(fn, reasoning, savefile):
    json_lines = []
    reasonings = {}
    with open(reasoning, "r") as f:
        json_lines = json.load(f)

    for json_line in json_lines:
        reasonings[json_line["key"]] = json_line

    with open(fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')

        c = 0
        corrects = 0
        structs = []
        for line in csv_reader:
            if c == 0:
                c += 1
                continue

            struct = {}
            struct["key"] = line[0]
            struct["premise"] = line[1] + " " + line[2]
            correct_answer = int(line[7])
            if corrects > 100:
                for t in range(4):
                    if t == correct_answer:
                        continue
                    struct["hypothesis"] = line[3 + t]
                    struct['answer'] = "no"

            else:
                struct["hypothesis"] = line[3 + correct_answer]
                struct['answer'] = "yes"
                corrects += 1

            if 'one of the above' in struct["hypothesis"]:
                continue

            if struct["key"] in reasonings:
                reasoning = reasonings[struct["key"]]["reasoning"]
                struct["gpt-3"] = reasoning
                structs.append(struct)
            else:
                val = struct["key"]
                print(f"No reasoning found for key {val}")

            if len(structs) > 200:
                break

    write_to_file(structs, savefile)


def parse_semeval(fn, reasoning, savefile):
    json_lines = []
    reasonings = {}
    with open(reasoning, "r") as f:
        json_lines = json.load(f)

    for json_line in json_lines:
        reasonings[json_line["key"]] = json_line

    with open(fn) as fd:
        doc = xmltodict.parse(fd.read())

    structs = []
    data = doc['data']  # == u'an attribute'
    instances = data['instance']  # == [u'elements', u'more elements']
    cnt = 0
    for instance in instances:
        ins = {}
        context = re.sub(r"\s+", ' ', instance["text"])
        id = instance["@id"]

        ins["id"] = id
        ins["premise"] = context

        ins["questions"] = []

        for question in instance['questions']["question"]:
            quu1 = {
                "premise": "",
                "hypothesis": "",
                "answer": "",
                "gpt-3": "",
                "dataset": "",
            }
            quu2 = {
                "premise": "",
                "hypothesis": "",
                "answer": "",
                "gpt-3": "",
                "dataset": "",
            }

            q = question["@text"]
            quu1["key"] = (
                instance["@id"]
                + "\t"
                + question["@id"]
                + "\t"
                + question["answer"][0]["@id"]
            )
            quu2["key"] = (
                instance["@id"]
                + "\t"
                + question["@id"]
                + "\t"
                + question["answer"][1]["@id"]
            )

            if question["answer"][0]["@correct"] == "False":
                quu1["hypothesis"] = question["answer"][0]["@text"]
                quu1["answer"] = "no"
            else:
                quu1["hypothesis"] = question["answer"][0]["@text"]
                quu1["answer"] = "yes"

            if question["answer"][1]["@correct"] == "False":
                quu2["answer"] = "no"
                quu2["hypothesis"] = question["answer"][1]["@text"]
            else:
                quu2["hypothesis"] = question["answer"][1]["@text"]
                quu2["answer"] = "yes"

            quu1["premise"] = context + " " + q
            quu2["premise"] = context + " " + q

            if quu1["key"] in reasonings:
                reasoning = reasonings[quu1["key"]]["reasoning"]
                quu1["gpt-3"] = reasoning
                structs.append(quu1)
            else:
                val = quu1["key"]
                print(f"No reasoning found for key {val}")

            if quu2["key"] in reasonings:
                reasoning = reasonings[quu2["key"]]["reasoning"]
                quu2["gpt-3"] = reasoning
                structs.append(quu2)
            else:
                val = quu2["key"]
                print(f"No reasoning found for key {val}")
        cnt += 1

    write_to_file(structs, savefile)


def parse_gsm8k(fn, savefile):
    with open(fn) as f:
        lines = f.readlines()

    structs = []
    for line in lines:
        data = json.loads(line.strip())
        blob = {}
        blob["premise"] = data["question"]
        blob["hypothesis"] = (
            "IGNORE THIS. Ground truth here for reference. " + data["ground_truth"]
        )
        blob["gpt-3"] = data["175b_verification"]["solution"]
        blob["answer"] = "yes" if data["175b_verification"]["is_correct"] else "no"
        blob["key"] = f"gsm8k_{len(structs)}"
        structs.append(blob)

    write_to_file(structs, savefile)


def main(opt):
    ### The line that constrcuts the context is:
    # context = "Premise: " + struct["premise"] + "\nHypothesis: " + struct["hypothesis"] + "\nExplanation: "
    ## The "Explanation" is followed by the GPT-3 generations
    datasets = opt["datasets"]
    path_to_data = opt["dataset_path"]
    path_to_generation = opt["generation_path"]
    output_path = opt["out_dir"]

    for dataset in datasets:
        input_file = os.path.join(path_to_data, dataset + ".txt")
        model_output_reasoning = os.path.join(
            path_to_generation, dataset + "_reasoning.txt"
        )
        save_file = os.path.join(output_path, dataset + ".json")

        if dataset == 'drop':
            parse_drop(input_file, model_output_reasoning, save_file)
            print(f"Saved DROP dataset in {save_file}")
        elif dataset == 'esnli':
            parse_esnli(input_file, model_output_reasoning, save_file)
            print(f"Saved E-SNLI dataset in {save_file}")
        elif dataset == 'cosmos':
            parse_cosmos(input_file, model_output_reasoning, save_file)
            print(f"Saved COSMOSQA dataset in {save_file}")
        elif dataset == 'semevalcommonsense':
            parse_semeval(input_file, model_output_reasoning, save_file)
            print(f"Saved SEMEVAL dataset in {save_file}")
        elif dataset == 'gsm8k':
            parse_gsm8k(input_file, save_file)
            print(f"Saved GSM8K dataset in {save_file}")
        else:
            raise NotImplementedError(f"Dataset {dataset} not recognized")


if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument(
        '--dataset-path',
        '-d',
        type=str,
        required=False,
        default=PATH_TO_DATA,
        help='Path to files with questions',
    )
    parser.add_argument(
        '--generation-path',
        '-g',
        type=str,
        required=False,
        default=PATH_TO_GENERATIONS,
        help='Path to files with generations',
    )
    parser.add_argument(
        '--datasets',
        '-s',
        type=str,
        default=DATASETS,
        choices=DATASETS,
        nargs="*",
        required=False,
        help='Dataset name',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        required=False,
        default=PATH_TO_GENERATIONS,
        help='Path where mixes will be saved.',
    )

    opt = parser.parse_args()

    main(opt)
