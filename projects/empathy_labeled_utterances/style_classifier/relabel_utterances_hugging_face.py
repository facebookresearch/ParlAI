import configargparse
import json
from os import path
from typing import List, Optional, Union
import torch
from torch.utils.data import DataLoader, upsample_examples
from parlai_internal.projects.empathy_generation.data_utils.data_utils_globals import (
    BINARY_LABELS,
    NUMERIC_LABELS,
)
from parlai_internal.projects.empathy_generation.huggingface.utils import (
    load_pipeline,
    CustomBertForSequenceClassification,
    EmpathyDataset,
)
from parlai_internal.projects.empathy_generation.huggingface.train_classifier import (
    train,
)
from parlai_internal.projects.empathy_generation.huggingface.evaluate_classifier import (
    evaluate,
)


# this is for the BERT-based HuggingFace classifier


@torch.no_grad()
def relabel_using_classifier(
    model: CustomBertForSequenceClassification,
    dataloader: DataLoader,
    data_fpath: str,
    results_fpath: str,
    relabeling_threshold: List[float],
    upsample: bool = False,
):
    with open(data_fpath, "w") as f_data, open(results_fpath, "w") as f_results:
        # keep all predictions that differ from ground-truth and are above certain threshold
        retained_new_labels = []
        pred_label_indices, probs = evaluate(model, dataloader, model, verbose=False)

        turns = []
        none_idx = (
            dataloader.dataset.style_candidates.index(None)
            if dataloader.dataset.retain_none_styles
            else -1
        )

        for idx, (pred_idx, prob) in enumerate(zip(pred_label_indices, probs)):
            turn = dataloader.dataset.data[idx]
            current_label_idx = turn["style_idx"]
            assert pred_idx != 2
            if (
                pred_idx != current_label_idx and prob >= relabeling_threshold[pred_idx]
            ) or current_label_idx == none_idx:
                # new label above threshold or old label is None
                turn["style_idx"] = dataloader.dataset.get_index(turn["style"])
                turn["style"] = dataloader.dataset.style_candidates[pred_idx]
                input_str = turn["context"] + " [SEP] " + turn["listener"]
                retained_new_labels.append(
                    (pred_idx, current_label_idx, prob, input_str)
                )

                # specific for HF; remove before writing to file
                turn.pop("history_str", None)
                turn.pop("context", None)
                turn.pop("style_idx", None)

            turns.append(turn)

        if upsample:
            turns, _ = upsample_examples(turns, dataloader.dataset.style_candidates)

        for turn in turns:
            json.dump(turn, f_data)
            f_data.write("\n")

        # sort new labels by probability and write to file
        changes_by_label = {
            idx: 0 for idx in range(len(dataloader.dataset.style_candidates))
        }
        for pred, original, prob, input_str in sorted(
            retained_new_labels, key=lambda x: x[2], reverse=True
        ):
            f_results.write(f'{pred} {format(prob, ".3f")} {input_str}\n')
            changes_by_label[original] += 1

        for cls_idx, num_changes in changes_by_label.items():
            cls_label = dataloader.dataset.style_candidates[cls_idx]
            f_results.write(
                f"{cls_label}: {num_changes} new labels, originally {dataloader.dataset.total_by_class[cls_label]}\n"
            )
        f_results.write("\n\n")


def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        type=str,
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="path to training data in ParlAI format, stored at exchange-level",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        required=True,
        help="path to validation data in ParlAI format, stored at exchange-level",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="path to test data in ParlAI format, stored at exchange-level",
    )
    parser.add_argument(
        "--output_file_prefix",
        type=str,
        required=True,
        help="file prefix for relabeled splits",
    )
    # for formatting for classifier
    parser.add_argument(
        "--label_type",
        choices=["binary", "numeric"],
        default="binary",
        help="use either binary or numeric labels for data for classifier",
    )
    parser.add_argument(
        "--exclude_from_train",
        nargs="+",
        choices=[
            "bst",
            "style_gen",
            "ed_train",
            "ed_valid",
            "ed_test",
            "wow",
            "convai2",
        ],
        default=[],
        help="these datasets will be removed during training run",
    )
    parser.add_argument(
        "--exclude_from_eval",
        nargs="+",
        choices=[
            "bst",
            "style_gen",
            "ed_train",
            "ed_valid",
            "ed_test",
            "wow",
            "convai2",
        ],
        default=[],
        help="these datasets will be removed during evaluation run",
    )
    parser.add_argument(
        "-up",
        "--upsample",
        action="store_true",
        help="if True, will balance classes by upsampling",
    )
    parser.add_argument(
        "--num_relabeling",
        type=int,
        default=2,
        help="number of times to relabel examples",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="per device train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=128, help="per device eval batch size"
    )
    parser.add_argument(
        "--relabeling_threshold",
        nargs="+",
        type=float,
        default=0.6,
        help="probability threshold for relabeling mislabeled examples from classifier",
    )
    parser.add_argument(
        "--relabeling_threshold_increase",
        nargs="+",
        type=float,
        default=0.1,
        help="amt to increase threshold per classifier pass",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6, help="classifier learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="number of epochs per relabeling pass"
    )
    parser.add_argument(
        "-hstsz",
        "--history_size",
        type=int,
        default=2,
        help="number of prior utterances in conversation to retain",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="path to directory to write relabeled examples",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--num_eval_steps",
        type=int,
        default=5000,
        help="number of steps between each evaluation",
    )
    args = parser.parse_args()

    label_candidates = BINARY_LABELS if args.label_type == "binary" else NUMERIC_LABELS
    if len(args.relabeling_threshold) == 1:
        args.relabeling_threshold = args.relabeling_threshold * len(label_candidates)
    else:
        assert len(args.relabeling_threshold) == len(
            label_candidates
        ), "Either provide single value for relabeling threshold or a value for every class"

    model, collator = load_pipeline(args.label_type)
    relabel_train_path, relabel_valid_path, relabel_test_path = (
        args.train_path,
        args.valid_path,
        args.test_path,
    )
    f_get_eval_dataloader = lambda dataset: DataLoader(
        dataset, collate_fn=collator, batch_size=args.eval_batch_size, shuffle=False
    )

    # initial evaluation on all data
    for dpath in [args.train_path, args.valid_path]:
        print(f"Run initial eval on all data at {dpath}")
        dataset = EmpathyDataset.from_file(
            dpath,
            train=False,
            label_type=args.label_type,
            history_size=args.history_size,
        )
        evaluate(model, f_get_eval_dataloader(dataset), verbose=True)

    for pass_idx in range(args.num_relabeling):
        print(f"Begin train for relabeling {pass_idx+1} of {args.num_relabeling}")
        train_dataset = EmpathyDataset.from_file(
            relabel_train_path,
            train=True,
            label_type=args.label_type,
            history_size=args.history_size,
            exclude_src_datasets=args.exclude_from_train,
            upsample=True,
        )
        eval_dataset = EmpathyDataset.from_file(
            relabel_valid_path,
            train=False,
            label_type=args.label_type,
            history_size=args.history_size,
            exclude_src_datasets=args.exclude_from_eval,
        )
        train(
            model=model,
            collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            results_dir=args.results_dir,
            num_workers=args.num_workers,
            eval_steps=args.num_eval_steps,
        )
        # run evaluation before labeling
        print("Eval results before relabeling", args.exclude_from_eval)
        eval_dataset_from_train = EmpathyDataset.from_file(
            relabel_train_path,
            train=False,
            label_type=args.label_type,
            history_size=args.history_size,
            exclude_src_datasets=args.exclude_from_eval,
        )
        eval_dataset_from_valid = EmpathyDataset.from_file(
            relabel_valid_path,
            train=False,
            label_type=args.label_type,
            history_size=args.history_size,
            exclude_src_datasets=args.exclude_from_eval,
        )
        evaluate(
            model, f_get_eval_dataloader(eval_dataset_from_train), model, verbose=True
        )
        evaluate(
            model, f_get_eval_dataloader(eval_dataset_from_valid), model, verbose=True
        )

        # -- relabel
        print("Begin relabeling..")
        for split_name, data_path_in in [
            ("train", relabel_train_path),
            ("valid", relabel_valid_path),
            ("test", relabel_test_path),
        ]:
            # if final relabeling pass, upsample train data
            dataset = EmpathyDataset.from_file(
                data_path_in,
                train=False,
                label_type=args.label_type,
                history_size=args.history_size,
                retain_none_styles=True,
            )
            # where to write new examples
            data_fpath_out = args.output_file_prefix + f"{split_name}.jsonl"
            # where to write classifier output
            results_fpath = path.join(
                args.results_dir,
                f"relabled_{split_name}_{args.num_relabeling}_passes.txt",
            )
            final_pass_upsample = (
                split_name == "train" and (pass_idx + 1) == args.num_relabeling
            )
            relabel_using_classifier(
                model,
                f_get_eval_dataloader(dataset),
                data_fpath_out,
                results_fpath,
                args.relabeling_threshold,
                upsample=final_pass_upsample,
            )

        # run evaluation again after labeling
        print("Eval results after relabeling:")
        relabel_train_path = args.output_file_prefix + "train.jsonl"
        relabel_valid_path = args.output_file_prefix + "valid.jsonl"
        relabel_test_path = args.output_file_prefix + "test.jsonl"

        eval_dataset_from_train = EmpathyDataset.from_file(
            relabel_train_path,
            train=False,
            label_type=args.label_type,
            history_size=args.history_size,
            exclude_src_datasets=args.exclude_from_eval,
        )
        eval_dataset_from_valid = EmpathyDataset.from_file(
            relabel_valid_path,
            train=False,
            label_type=args.label_type,
            history_size=args.history_size,
            exclude_src_datasets=args.exclude_from_eval,
        )
        evaluate(model, f_get_eval_dataloader(eval_dataset_from_train), verbose=True)
        evaluate(model, f_get_eval_dataloader(eval_dataset_from_valid), verbose=True)

        # increase threshold for next pass
        args.relabeling_threshold = [
            thresh + args.relabeling_threshold_increase
            for thresh in args.relabeling_threshold
        ]

    # final evaluation on all data
    for dpath in [
        args.output_file_prefix + "train.jsonl",
        args.output_file_prefix + "valid.jsonl",
    ]:
        print(f"Run final eval on all data at {dpath}")
        dataset = EmpathyDataset.from_file(
            dpath,
            train=False,
            label_type=args.label_type,
            history_size=args.history_size,
        )
        evaluate(model, f_get_eval_dataloader(dataset), verbose=True)


if __name__ == "__main__":
    main()
