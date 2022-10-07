import subprocess
import shlex
from loguru import logger
from datetime import datetime
import time
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-m", "--model", type=str, default="bart", help="bart or gpt2")
parser.add_argument(
    "-lr", "--learning_rate", type=float, default=-1, help="bart or gpt2"
)

args = parser.parse_args()

tasks = {
    "all": "wsc,wnli,wic,squad2,google_sgd_dst,paraphrase_classification,wikisql,coqa",
    # "-google_sgd": "wsc,wnli,wic,squad2,paraphrase_classification,wikisql,coqa",
    # "-qa": "wsc,wnli,wic,google_sgd_dst,paraphrase_classification,wikisql",
    # "-exp_coref": "squad2,google_sgd_dst,paraphrase_classification,wikisql,coqa",
    # "-wikisql": "wsc,wnli,wic,squad2,google_sgd_dst,paraphrase_classification,wikisql,coqa",
    # "-paraphrase": "wsc,wnli,wic,squad2,google_sgd_dst,wikisql,coqa",
    # "-copy": "wsc,wnli,wic,paraphrase_classification",
    # "-all_coref": "squad2,google_sgd_dst,paraphrase_classification,wikisql",
    # "+qa": "squad2,coqa",
    # "+paraphrase": "paraphrase_classification",
    # "+copy": "google_sgd_dst,wikisql,squad2,coqa",
    # "+sgd": "google_sgd_dst",
    # "+exp_coref": "wsc,wnli,wic",
    # "+all_coref": "wsc,wnli,wic,coqa",
    # "+wikisql": "wikisql",
    # "g_sgd_reversed": "wsc,wnli,wic,squad2,multiwoz_dst:version=2.3,paraphrase_classification,wikisql,coqa",
    # "all_multitask": "wsc,wnli,wic,squad2,multiwoz_dst:data_version=2.3,google_sgd_dst,paraphrase_classification,wikisql,coqa"
}

if args.learning_rate == -1:
    lrs = [
        # "5e-5",
        # "1e-5",
        "5e-6",
    ]
else:
    lrs = [args.learning_rate]

epochs = 10

val_settings, run_eval = (
    "--validation-metric loss --validation-metric-mode min --validation_cutoff 0",
    False,
)
# val_settings, run_eval = (
#     "--validation-metric 'joint goal acc' --validation-metric-mode max --validation_cutoff 100",
#     True,
# )

count = 0
for alias, tasks in tasks.items():
    for lr in lrs:
        # time.sleep(1.1)
        now = datetime.now()
        now = datetime.strftime(now, "%Y-%m-%d_%T")

        cmd = f"sbatch --job-name {alias}_pft "
        cmd += f"{args.model}_prefinetune.sh -t {tasks} -l {lr} -e {epochs} -n {alias} -k {now} -m {val_settings} -v '{run_eval}'"
        logger.info(f"Executing command: {cmd}")
        subprocess.run(shlex.split(cmd))
        count += 1

logger.info(f"Total number of jobs submitted: {count}")
