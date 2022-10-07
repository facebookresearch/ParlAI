# organize results into CSV file that is easy to copy over to spread sheet
# Usage: python organize_results.py

from argparse import ArgumentParser
from glob import glob
import os
import pandas as pd
import json
from collections import OrderedDict
from loguru import logger
from pathlib import Path

HOME = os.environ.get("HOME", "~")


def organize_results(fps):

    # desired_metrics = ["jga_original", "jga_perturbed", "jga_conditional", "consistency", "all_ne/hallucination", "jga_all", "slot_p", "slot_r"]
    desired_types = ["original", "perturbed", "conditional", "new_conditional"]
    desired_metrics = ["jga", "slot_p", "slot_r", "all_ne/hallucination"]

    target_metrics = []
    for dm in desired_metrics:
        for dt in desired_types:
            target_metrics.append(f"{dm}_{dt}")
    others = ["consistency"]
    target_metrics += others

    all_results = []
    problematic_dirs = []
    for dir in sorted(fps, reverse=True):
        reports = glob(os.path.join(dir, "*_report*.json"))
        if reports and len(reports) == 3:
            reports = sorted(reports)
            print("\n".join(reports))
        #     break

        if len(reports) != 3:
            reports_str = "\n\t".join(reports)
            logger.info(
                f"There are more or fewer than 3 reports in this dir: {dir}\n\t {reports_str}\n Passing this directory."
            )
            problematic_dirs.append(dir)
            # continue

        results_dict = OrderedDict()
        results_dict["fn"] = dir.replace(f"{HOME}/ParlAI/models/", "")

        trainstats_fn = os.path.join(dir, "model.trainstats")
        if not os.path.isfile(trainstats_fn):
            logger.info(f"No {trainstats_fn} found.")
            continue
        with open(trainstats_fn, "r") as f:
            trainstats = json.load(f)

        if "final_test_report" in trainstats:
            jga = trainstats["final_test_report"].get("joint goal acc", -1)
            coref_jga = trainstats["final_test_report"].get("coref_jga", -1)
            coref_ct = trainstats["final_test_report"].get("coref_ct", -1)
            if jga == -1:
                continue
            results_dict["JGA"] = jga
            results_dict["coref_jga"] = coref_jga
            results_dict["coref_ct"] = coref_ct
            results_dict["PPL"] = trainstats["final_test_report"].get("ppl", -1)

        for report in reports:
            # make sure none of the faulty evaluation results go in.
            if (
                "fewshot_True" not in dir and "fs_True" not in dir
            ) and "fs_True" in report:
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
                        logger.warning(
                            f"duplicate metric name found from {report}: {metric_name}"
                        )

        all_results.append(results_dict)
        # break

    df = pd.DataFrame(all_results)
    df = df.reindex(
        [
            "fn",
            "JGA",
            "coref_jga",
            "coref_ct",
            "PPL",
            "NEI_all_ne/hallucination_conditional",
            "NEI_all_ne/hallucination_original",
            "NEI_all_ne/hallucination_perturbed",
            "NEI_consistency",
            "NEI_jga_conditional",
            "NEI_jga_new_conditional",
            "NEI_jga_original",
            "NEI_jga_perturbed",
            "NEI_slot_p_conditional",
            "NEI_slot_p_original",
            "NEI_slot_p_perturbed",
            "NEI_slot_r_conditional",
            "NEI_slot_r_original",
            "NEI_slot_r_perturbed",
            "SD_all_ne/hallucination_conditional",
            "SD_all_ne/hallucination_original",
            "SD_all_ne/hallucination_perturbed",
            "SD_consistency",
            "SD_jga_conditional",
            "SD_jga_new_conditional",
            "SD_jga_original",
            "SD_jga_perturbed",
            "SD_slot_p_conditional",
            "SD_slot_p_original",
            "SD_slot_p_perturbed",
            "SD_slot_r_conditional",
            "SD_slot_r_original",
            "SD_slot_r_perturbed",
            "TP_all_ne/hallucination_conditional",
            "TP_all_ne/hallucination_original",
            "TP_all_ne/hallucination_perturbed",
            "TP_consistency",
            "TP_jga_conditional",
            "TP_jga_new_conditional",
            "TP_jga_original",
            "TP_jga_perturbed",
            "TP_slot_p_conditional",
            "TP_slot_p_original",
            "TP_slot_p_perturbed",
            "TP_slot_r_conditional",
            "TP_slot_r_original",
            "TP_slot_r_perturbed",
        ],
        axis=1,
    )

    df.to_csv(f"{HOME}/CheckDST/ParlAI/bash_scripts/all_summary.csv")

    print()
    print("\n".join(problematic_dirs))
    print(len(problematic_dirs))
    print(len(fps))

    return df


if __name__ == "__main__":

    fps = []
    # from evaluate_all_invariance_metrics.py
    fps += glob(
        f"{HOME}/ParlAI/models/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_scratch_multiwoz2.3/fs_True_prompts_True_lr5e-05_bs4_uf1*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.3/fs_True_prompts_True_lr1e-05_bs4_uf2*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_muppet_multiwoz2.3/fs_True_prompts_True_lr5e-05_bs4_uf1*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_muppet_multiwoz2.3/fs_False_prompts_True_lr1e-04_bs4_uf1*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_scratch_multiwoz2.1/fs_False_prompts_True_lr5e-05_bs8_uf1*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_scratch_multiwoz2.1/fs_True_prompts_True_lr5e-05_bs4_uf1*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.1/fs_False_prompts_True_lr1e-05_bs8_uf2*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.1/fs_True_prompts_True_lr5e-05_bs4_uf2*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_muppet_multiwoz2.1/fs_False_prompts_True_lr1e-04_bs8_uf1*"
    )
    fps += glob(
        f"{HOME}/ParlAI/models/bart_muppet_multiwoz2.1/fs_True_prompts_True_lr1e-04_bs4_uf1*"
    )
    organize_results(fps)
