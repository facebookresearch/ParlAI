from argparse import ArgumentParser
import shlex
from subprocess import run
from pathlib import Path
import os
from loguru import logger
from glob import glob

parser = ArgumentParser()

# parser.add_argument("-mf", "--model_file", type=str, help="path to model checkpoint")
parser.add_argument(
    "-fs", "--fewshot", type=bool, default=False, help="Use few shot data"
)
parser.add_argument(
    "-f",
    "--force",
    type=bool,
    default=False,
    help="Overwrite previous invariance report results if report is already there",
)


args = parser.parse_args()

invariances = ["TP", "SD", "NEI"]
invariances = ["NEI"]

fps = []

# fps += glob("/data/home/justincho/ParlAI/models/gpt2_scratch_multiwoz2.3/*")
# fps += glob("/data/home/justincho/ParlAI/models/gpt2_sgd_ft_multiwoz2.3/*")
# fps += glob("/data/home/justincho/ParlAI/models/gpt2_para_ft_multiwoz2.3/*")
# fps += glob("/data/home/justincho/ParlAI/models/gpt2_scratch_LAUG_TP_multiwoz2.3/*")
# fps += glob("/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/*")
fps += glob("/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.3/*")
# fps += glob("/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/*")

# fps = ["/data/home/justincho/ParlAI/models/gpt2_scratch_LAUG_TP_multiwoz2.3/2021-10-15_07:34:50_ngpu1_bs8_lr5e-4_eps20_fewshot_False_sd0"]

run_count = 0
for fp in fps:
    temp_fewshot = args.fewshot
    filename = Path(fp).name

    if "model" not in filename and os.path.isfile(os.path.join(fp, "model")):
        logger.warning(
            "Make sure to add the full path to the model file, not the path. Automatically adding 'model' to the model path..."
        )
        fp = os.path.join(fp, "model")

    if "fewshot_True" in fp:
        logger.warning(
            "Found 'fewshot_True' in model path. Setting to use fewshot test set"
        )
        temp_fewshot = True

    # if there is no test results, there was a problem during training or the training is incomplete. skip
    if not os.path.isfile(fp + ".test"):
        continue

    for inv in invariances:

        report_fn = fp + f".{inv}_report_fs_{temp_fewshot}.json"
        # skip if we already have the invariance results
        if not args.force and os.path.isfile(report_fn):
            logger.info(
                f"Invariance report already found. skipping for {fp} for invariance {inv}"
            )
            continue

        logger.info(f"Saving report to {report_fn}...")

        cmd = "sbatch "

        # this doesn't seem to work
        # cmd += "--partition=a100"
        # cmd += "--gres=gpu:1"
        # cmd += "--time=4:00:00"
        # cmd += "--cpus-per-task=10"
        # cmd += f"--output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/eval_laug_inv_{inv}-%j.log"
        # cmd += f"--job-name={inv}_inv_eval"

        if "bart" in fp:
            model_type = "bart"
        else:
            model_type = "hugging_face/gpt2"

        cmd += f"evaluate_laug_invariance.sh -m {fp} -i {inv} -f {temp_fewshot} -d {model_type}"

        print(f"Executing command: \n\t{cmd}")

        run(shlex.split(cmd))

        run_count += 1

    # break

logger.info(f"Total number of jobs submitted: {run_count}")
