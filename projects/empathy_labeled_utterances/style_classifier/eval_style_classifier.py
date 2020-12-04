from collections import Counter
import configargparse
from copy import deepcopy
from itertools import permutations
import json
from os import path
from parlai.core.torch_classifier_agent import ConfusionMatrixMetric, WeightedF1Metric
from parlai_internal.projects.empathy_generation.style_classifier.model_utils import (
    BatchClassifier,
    load_classifier_agent,
    FUTURE_SEP_TOKEN,
)

HISTORY_DELIMITER = "\n"


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "config", is_config_file=True, type=str, help="path to config file"
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument(
        "--classes_file", type=str, required=True, help="path to style classes"
    )
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        type=str,
        choices=["train:evalmode", "valid", "test"],
        required=True,
        help="datasplits(s) to data for evaluation",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        required=True,
        help="path to pretrained classifier",
    )
    parser.add_argument(
        "--data_prefix",
        type=str,
        choices=[
            "original",
            "original_upsampled",
            "relabeled_prev_curr",
            "relabeled_prev_curr_upsampled",
            "relabeled_prev_curr_future",
            "relabeled_prev_curr_future_upsampled",
            "relabeled_second_pass_prev_curr_future",
            "relabeled_second_pass_prev_curr_future_upsampled",
        ],
        required=True,
    )
    parser.add_argument(
        "--generate_future",
        action="store_true",
        help="if true, hallucinate future utterance for classifier",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="location to store results"
    )
    parser.add_argument(
        "--task",
        choices=["prev_curr", "prev_curr_future"],
        default="prev_curr_future",
        help="specify which data format for classifier",
    )
    args = parser.parse_args()
    return args


def styles_from_stylegen():
    # TODO move this code elsewhere; used for getting style predictions from Eric's classifier
    # get counts and convert to percentages
    style_percentages = {}
    for label, by_label in styles_by_empathy_label.items():
        style_percentages[label] = {
            style: count / total_by_class[label]
            for style, count in Counter(by_label).items()
        }

    # compute deltas for emotions between classes
    f_counts.write(f"Difference for split {eval_split}\n")
    for label_A, label_B in permutations(empathy_labels):
        # deltas for speakers
        diffs = {}
        for emot in style_percentages[label_A].keys():
            diffs[emot] = style_percentages[label_A].get(emot, 0) - style_percentages[
                label_B
            ].get(emot, 0)

        f_counts.write(f"Deltas between {label_A} and {label_B}\n")
        for emotion, diff in sorted(diffs.items(), key=lambda x: x[1], reverse=True):
            fdiff = format(100 * diff, ".2f")
            f_counts.write(f"{emotion} {fdiff} \n")


def main():
    args = parse_args()

    # load classifier and possibly future generator
    batch_classifier = BatchClassifier(
        classifier_path=args.classifier_path,
        batch_size=args.batch_size,
        classes_file=args.classes_file,
    )
    batch_future_generator = None
    if args.generate_future:
        assert (
            args.task == "prev_curr_future"
        ), "Can only hallucinate future utterance when using future-looking classifier"
        from parlai_internal.projects.empathy_generation.style_controlled_generation.model_utils import (
            FutureGenerator,
        )

        batch_future_generator = FutureGenerator(args.batch_size)

    f_out = open("with_hall_gens_test.jsonl", "w")
    for eval_split in ["test"]:  # args.eval_splits:
        classifier_agent = load_classifier_agent(
            eval_split, args.data_prefix, args.task
        )
        all_ground_truths, all_predictions = [], []
        for batch_idx, start_example_idx in enumerate(
            range(0, len(classifier_agent.data), args.batch_size)
        ):
            # for batch_idx, start_example_idx in enumerate(range(0, 100, args.batch_size)):
            if batch_idx % 5 == 0:
                print(f"On batch {batch_idx}")
            unformatted_chunk = classifier_agent.data[
                start_example_idx : start_example_idx + args.batch_size
            ]
            if (
                args.generate_future
            ):  # will hallucinate a new future utterance to replace ground-truth
                future_classifier_input = []
                for i, episode in enumerate(unformatted_chunk):
                    input_text = HISTORY_DELIMITER.join(
                        [
                            *episode["history"],
                            episode["speaker_utt"],
                            episode["listener_utt"],
                        ]
                    )
                    future_classifier_input.append(input_text)

                gen_input = batch_future_generator.batch_generate(
                    future_classifier_input
                )
                for i in range(len(unformatted_chunk)):
                    unformatted_chunk[i]["future_utt"] = gen_input[i]
                    json.dump(unformatted_chunk[i], f_out)
                    f_out.write("\n")

            batch = []
            for episode in unformatted_chunk:
                context = (
                    (
                        episode["speaker_utt"]
                        + f" {FUTURE_SEP_TOKEN} "
                        + episode["future_utt"]
                    )
                    if args.task == "prev_curr_future"
                    else episode["speaker_utt"]
                )

                text = HISTORY_DELIMITER.join([context, episode["listener_utt"]])

                batch.append({"text": text, "style": episode["style"]})
            batch_ground_truths, batch_predictions = batch_classifier.batch_classify(
                batch
            )
            all_ground_truths.extend(batch_ground_truths)
            all_predictions.extend(batch_predictions)
        f_out.close()
        exit()

        # get metrics
        assert len(all_predictions) == len(all_ground_truths)
        f1_dict = {}
        class_list = set(all_predictions + all_ground_truths)
        for class_name in class_list:
            precision, recall, f1 = ConfusionMatrixMetric.compute_metrics(
                all_predictions, all_ground_truths, class_name
            )
            f1_dict[class_name] = f1
            filtered_precision, filtered_recall, filtered_f1 = [], [], []
            acc = []
            for i, gt in enumerate(all_ground_truths):
                if gt == class_name:
                    acc.append(gt == all_predictions[i])
                    filtered_precision.append(precision[i])
                    filtered_recall.append(recall[i])
                    filtered_f1.append(f1[i])

            print(sum(acc) / len(acc), "end acc")

            print(
                f"class {class_name} precision/recall/f1 -",
                sum([f.value() for f in filtered_precision]) / len(filtered_precision),
                sum([f.value() for f in filtered_recall]) / len(filtered_recall),
                sum([f.value() for f in filtered_f1]) / len(filtered_f1),
            )
        weighted_f1 = WeightedF1Metric.compute_many(f1_dict)
        print("weighted f1:", sum([w.value() for w in weighted_f1]) / len(weighted_f1))

        # for class_name in batch_classifier.class_list:
        #    # these values are a list of ints for each ground truth in the list that matches the class name
        #    list_precision, list_recall, list_f1 = ConfusionMatrixMetric.compute_metrics(
        #            all_predictions, all_ground_truths, class_name
        #        )
        #    f1_for_class = sum(WeightedF1Metric.compute_many({class_name : list_f1}), None)
        #    print(f'{class_name}_f1: {f1_for_class}')
        #    f1_dict[class_name] = list_f1
        # weighted_f1 = sum(WeightedF1Metric.compute_many(f1_dict), None)
        # print('weighted f1:', weighted_f1)


if __name__ == "__main__":
    main()
