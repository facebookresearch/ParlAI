import json
from pathlib import Path
from pprint import pprint
import re
import shlex
import subprocess

from checkdst import (
    sort_by_steps,
    find_step,
    find_seed,
    find_all_checkpoints,
    find_epoch_for_checkpoint,
)
import sys

import os

CHECKDST_DIR = os.environ.get("CHECKDST_DIR", "~/CheckDST")

# PATH = f"{CHECKDST_DIR}/ParlAI/models/bart_scratch_multiwoz2.3"
# PATH = f"{CHECKDST_DIR}/ParlAI/models/bart_pft_multiwoz2.3"
# PATH = f"{CHECKDST_DIR}/ParlAI/models/bart_muppet_multiwoz2.3"
PATH = f"{CHECKDST_DIR}/ParlAI/models/bart_soloist_multiwoz2.3"

runs = Path(PATH).glob("*sd[0-9]")


MODEL = "bart"
ADDITIONAL = "--init-fairseq-model None"

BASE_SCRIPT_NAME = "bart_checkdst_eval.sh"
AUG_TYPES = ["orig", "NED", "SDI", "PI"]
TASK = "multiwoz_checkdst:augmentation_method={}"
DO_TEST = False
FEWSHOT = False
USEPROMPT = False
BATCH_SIZE = 128
EPOCHS = [0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]
# EPOCHS = [5, 10]


with open(BASE_SCRIPT_NAME, "r") as f:
    lines = f.readlines()

CMD = []
for run in sorted(list(runs)):

    seed = find_seed(run)

    # run for specific seed only
    # if seed != "3":
    #     continue

    print(run, seed)
    checkpoints = find_all_checkpoints(run)
    checkpoints = sort_by_steps(checkpoints)
    # pprint(list(checkpoints))
    for ckpt in checkpoints:

        epoch = find_epoch_for_checkpoint(ckpt)

        if epoch not in EPOCHS:
            continue

        print(epoch)
        CHECKPOINT = str(ckpt)
        model_config = Path(PATH).name
        # custom arguments needed for GPT-2 vs BART
        if "gpt2" in CHECKPOINT:
            MODEL = "hugging_face/gpt2"
            ADDITIONAL = "--add-special-tokens True"

        # evaluate for all checkdst augmentations and test set
        DATA_TYPE = "test"
        for aug in AUG_TYPES:
            task = TASK.format(aug)
            REPORT_FN = f"{CHECKPOINT}.{DATA_TYPE}_report_{task}"
            WORLDLOGS_FN = f"{CHECKPOINT}.{DATA_TYPE}_world_logs_{task}.jsonl"
            if not Path(WORLDLOGS_FN).is_file():
                CMD.append(
                    f"""
            
parlai eval_model \
    -m {MODEL} \
    -mf {CHECKPOINT} \
    -t {task} -bs {BATCH_SIZE} \
    -dt {DATA_TYPE} \
    --just_test {DO_TEST} \
    --report-filename {REPORT_FN} \
    --world-logs {WORLDLOGS_FN} \
    --skip-generation False \
    --few_shot {FEWSHOT} \
    --use_prompts {USEPROMPT} \
    {ADDITIONAL} |& tee {CHECKPOINT}_{DATA_TYPE}.log.txt

"""
                )

        # evaluate for validation loss
        DATA_TYPE = "valid"
        task = TASK.format("orig")
        REPORT_FN = f"{CHECKPOINT}.{DATA_TYPE}_report"
        WORLDLOGS_FN = f"{CHECKPOINT}.{DATA_TYPE}_world_logs.jsonl"
        if not Path(WORLDLOGS_FN).is_file():
            CMD.append(
                f"""
            
parlai eval_model \
    -m {MODEL} \
    -mf {CHECKPOINT} \
    -t {task} -bs {BATCH_SIZE} \
    -dt {DATA_TYPE} \
    --just_test {DO_TEST} \
    --report-filename {REPORT_FN} \
    --world-logs {WORLDLOGS_FN} \
    --skip-generation False \
    --few_shot {FEWSHOT} \
    --use_prompts {USEPROMPT} \
    {ADDITIONAL} |& tee {CHECKPOINT}_{DATA_TYPE}.log.txt 

"""
            )
    # break

# leverage as many jobs as allowed by splitting commands into chunks
total_num_jobs = int(sys.argv[1])


JOBNAME = f"checkdst_{model_config}"
print(JOBNAME)
print(lines)
if CMD:
    print(CMD[0])
    print(CMD[-1])

    jobs_per_chunk = len(CMD) // total_num_jobs
    chunks = []
    # split into chunks
    for idx in range(0, len(CMD), jobs_per_chunk):
        chunks.append(CMD[idx : idx + jobs_per_chunk])

    print(chunks)
    print(len(chunks))
    print(len(chunks[0]))
    print(len(CMD))
    print(jobs_per_chunk)

    if len(chunks) > total_num_jobs:
        new_chunks = chunks[:-1]
        new_chunks[-1] += chunks[-1]
        chunks = new_chunks
        print(len(chunks))
    assert len(chunks) == total_num_jobs, (len(chunks), total_num_jobs)

    for idx, chunk in enumerate(chunks):

        new_script_fn = Path(BASE_SCRIPT_NAME).with_suffix(f".{JOBNAME}_{idx}.temp")
        with new_script_fn.open("w") as f:
            f.writelines(lines)
            f.writelines(chunk)

        # run the command
        full_cmd = f"sbatch -J {JOBNAME} {str(new_script_fn)}"
        subprocess.run(shlex.split(full_cmd))
        # delete file
        # new_script_fn.unlink()
else:
    print("No jobs to run.")
