# organize results into CSV file that is easy to copy over to spread sheet
# Usage: python organize_results.py

from argparse import ArgumentParser
from glob import glob
import os
import pandas as pd
import json
from collections import OrderedDict
from loguru import logger


fps = []

# provide list of directories to organize results for
# fps += glob("/data/home/justincho/ParlAI/models/gpt2_scratch_multiwoz2.3/*")
# fps += glob("/data/home/justincho/ParlAI/models/gpt2_sgd_ft_multiwoz2.3/*")
# fps += glob("/data/home/justincho/ParlAI/models/gpt2_para_ft_multiwoz2.3/*")
# fps += glob("/data/home/justincho/ParlAI/models/gpt2_scratch_LAUG_TP_multiwoz2.3/*")
# fps += glob("/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.3/*")
fps += glob("/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/*")

print(fps)

# desired_metrics = ["jga_original", "jga_perturbed", "jga_conditional", "consistency", "all_ne/hallucination", "jga_all", "slot_p", "slot_r"]
desired_types = ["original", "perturbed", "conditional"]
desired_metrics = ["jga", "slot_p", "slot_r", "all_ne/hallucination"]

target_metrics = []
for dm in desired_metrics:
    for dt in desired_types:
        target_metrics.append(f"{dm}_{dt}")
others = ["consistency"]
target_metrics += others

all_results = []
problematic_dirs = []
for dir in sorted(fps):
    reports = glob(os.path.join(dir, "*report*.json"))
    reports = sorted(reports)

    if len(reports) != 3:
        reports_str = '\n\t'.join(reports)
        logger.info(
            f"There are more or fewer than 3 reports in this dir: {dir}\n\t {reports_str}\n Passing this directory."
        )
        problematic_dirs.append(dir)
        # continue

    results_dict = OrderedDict()
    results_dict["fn"] = dir.replace("/data/home/justincho/ParlAI/models/", "")

    trainstats_fn = os.path.join(dir, "model.trainstats")
    with open(trainstats_fn, "r") as f:
        trainstats = json.load(f)

    if "final_test_report" in trainstats:
        results_dict["JGA"] = trainstats["final_test_report"].get("joint goal acc", -1)
        results_dict["PPL"] = trainstats["final_test_report"].get("ppl", -1)

    for report in reports:
        # make sure none of the faulty ones go in.
        if "fewshot_True" not in dir and "fs_True" in report:
            continue
        with open(report, "r") as f:
            data = json.load(f)

        invariance = report.split("model.")[1].split("_")[0]
        # print(f"Processing {invariance} invariance metrics...")

        reported_metrics = data["report"]

        sorted_metrics = sorted(
            [(k, v) for k, v in reported_metrics.items()], key=lambda x: x[0]
        )

        for k, v in sorted_metrics:
            # for metric in desired_metrics:
            #     if metric in k:
            #         results_dict[f"{invariance}_{k}"] = [v]
            if k in target_metrics:
                metric_name = f"{invariance}_{k}"
                if metric_name not in results_dict:
                    results_dict[metric_name] = v
                else:
                    logger.warning(f"duplicate metric name found: {metric_name}")

    all_results.append(results_dict)

df = pd.DataFrame(all_results)

df.to_csv("all_summary.csv")

print()
print("\n".join(problematic_dirs))
print(len(problematic_dirs))
print(len(fps))
