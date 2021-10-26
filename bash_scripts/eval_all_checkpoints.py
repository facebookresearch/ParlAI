# Usage: python eval_all_checkpoints <dir with model checkpoints>

from subprocess import run
import sys
import glob
import os
from pathlib import Path
import shlex


parlai_dir = "/data/home/justincho/ParlAI/"  # to change
main_dir = sys.argv[1]  # pass main directory

command = "sbatch eval_checkpoint.sh {} {} {}"

# start evaluation only from epoch 5 and onwards
fn_list = glob.glob(f"{main_dir}/**checkpoint_ep[5-9]") + glob.glob(
    f"{main_dir}/**checkpoint_ep[0-9]?"
)

for f in fn_list:
    print(f)

for ckpoint in fn_list:
    path_obj = Path(ckpoint)
    par = path_obj.parent.parent.name
    subpar = path_obj.parent.name
    name = path_obj.name
    exp_fn = name[: name.index("checkpoint")] + "report" + name[name.index("_") :]
    log_fn = (
        name[: name.index("checkpoint")] + "logs" + name[name.index("_") :] + ".jsonl"
    )
    exp_path = f"experiment/{par}/{subpar}/{exp_fn}"
    exp_path = os.path.join(parlai_dir, exp_path)
    worlds_log_path = f"experiment/{par}/{subpar}/{log_fn}"

    cmd = command.format(ckpoint, exp_path, worlds_log_path)
    # print(cmd)
    # break

    run(shlex.split(cmd))


# organize all outputs and store it in a single file easy to copy to an excel spreadsheet
