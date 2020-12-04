import configargparse
from collections import Counter
from itertools import permutations
import json
import os
from os import path
import subprocess
from parlai_internal.tasks.empathy_labeled_utterances.agents import (
    _path as get_datapath,
)

# from parlai_internal.projects.style_gen.models.util import (
#    calculate_stats_per_label,
#    create_confusion_matrix,
# )


def format_ids(conv_turn_id: str):
    ids = conv_turn_id.split('_')
    conv_id = ids[0]
    turn_id = '_'.join(ids[1:])  # will possibly include bot id
    if 'bot' in conv_turn_id:
        print('c', conv_id, turn_id)
        exit()
    return conv_id, turn_id


def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        'config', is_config_file=True, type=str, help='path to config file'
    )
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument(
        '--data_prefix',
        type=str,
        choices=[
            'original',
            'original_upsampled',
            'relabeled_prev_curr',
            'relabeled_prev_curr_upsampled',
            'relabeled_prev_curr_future',
            'relabeled_prev_curr_future_upsampled',
            'relabeled_second_pass_prev_curr_future',
            'relabeled_second_pass_prev_curr_future_upsampled',
        ],
        required=True,
        help='TODO',
    )
    parser.add_argument(
        '--generator_path', type=str, required=True, help='path to trained generator'
    )
    parser.add_argument('--dict_path', type=str, required=True, help='path to dict')
    parser.add_argument(
        '--history_size',
        type=int,
        default=2,
        help='history size (in # of utterances) to retain for each turn',
    )
    parser.add_argument(
        '--generation_dir',
        type=str,
        required=True,
        help='directory to save generations',
    )
    parser.add_argument(
        '--report_dir',
        type=str,
        required=True,
        help='directory to save evaluation reports',
    )
    parser.add_argument(
        '--style_frac', type=float, default=1.0, help='amount of time to use style'
    )
    args = parser.parse_args()

    # TODO get classifier
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.generation_dir, exist_ok=True)
    print(args)

    for datatype in ['test']:  # ['train:evalmode', 'valid', 'test']:
        # get model generations
        cmd = f'''python -m parlai.scripts.eval_model \
                --data-prefix {args.data_prefix} \
                --batchsize {args.batch_size} \
                --datatype {datatype} \
                --history-size {args.history_size} \
                -mf {args.generator_path} \
                --report-filename {args.report_dir}/report \
                --save-world-logs True \
                --skip-generation False \
                --task parlai_internal.tasks.empathy_labeled_utterances.agents:ContextAndStyleAgent \
                --use_style_frac 1 \
                --exclude-bot-responses False \
                --model-parallel True \
                --inference beam \
                --beam-size 10 \
                --beam-min-length 20 \
                --beam-block-ngram 3 \
                --beam-context-block-ngram 3 \
                '''
        subprocess.run(cmd.split(), check=True)
        report_file = f'{args.report_dir}/report_parlai_internal.tasks.empathy_labeled_utterances.agents:ContextAndStyleAgent_replies.jsonl'
        print(f'Saved report to {report_file}')

        # load raw data
        opt_datapath = {
            'datapath': '/private/home/ccross/ParlAI/data',
            'datatype': datatype,
            'data_prefix': args.data_prefix,
        }
        datapath = get_datapath(opt_datapath)
        with open(datapath) as f_data:
            data_by_conv = {}
            for line in f_data:
                turn = json.loads(line)
                conv_id, turn_id = format_ids(turn['conv_turn_id'])
                key = (
                    turn['src_dataset'] + '_' + conv_id
                )  # store by dataset and conv num
                conv = data_by_conv.get(key)
                if conv is None:
                    data_by_conv[key] = {turn_id: turn}
                else:
                    conv[turn_id] = turn

        with open(report_file) as f_report:
            for line in f_report:
                input_data, pred_data = json.loads(line)['dialog'][0]
                if 'beam_texts' not in pred_data:
                    break
                conv_id, turn_id = format_ids(input_data['conv_turn_id'])
                key = (
                    input_data['src_dataset'] + '_' + conv_id
                )  # store by dataset and conv num
                conv = data_by_conv[key]

                pred_str, pred_prob = pred_data['beam_texts'][0]
                if 'pred_str' in conv[turn_id]:
                    raise Exception('duplicate key?')
                conv[turn_id].update({'pred_str': pred_str, 'pred_prob': pred_prob})

        datatype = datatype.split(":")[0]  # strip the evalmode portion, if present
        generation_path = path.join(args.generation_dir, f'generation_{datatype}.jsonl')
        with open(generation_path, 'w') as f_out:
            for conv_id, conv in data_by_conv.items():
                turns = [turn for _, turn in sorted(conv.items(), key=lambda x: x[0])]
                json.dump(turns, f_out)
                f_out.write('\n')
        print(f'Saved generations to {generation_path}')
    exit()

    for dpath in args.eval_paths:
        dataset = EmpathyParlAIStyleClassifierDataset.from_file(
            dpath,
            train=False,
            history_size=args.history_size,
            label_type=args.label_type,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        examples = dataset.get_examples_as_messages()

        empathy_labels = dataset.label_candidates
        file_tag, _ = path.splitext(path.basename(dpath))

        predicted_styles, prob_strings = batch_classifier.batch_classify(examples)

        assert len(examples) == len(predicted_styles) == len(prob_strings)
        with open(
            path.join(args.results_dir, f'tone_classifier_counts_{file_tag}.txt'), 'w'
        ) as f_counts, open(
            path.join(args.results_dir, f'tone_classifier_examples_{file_tag}.txt'), 'w'
        ) as f_examples:
            styles_by_empathy_label = {label: [] for label in empathy_labels}
            total_by_class = {label: 0 for label in empathy_labels}
            for (speaker, listener, label_idx), predicted_style, prob_string in zip(
                dataloader.dataset.examples, predicted_styles, prob_strings
            ):
                predicted_style = predicted_style.lower()
                label = empathy_labels[label_idx]
                styles_by_empathy_label[label].append(predicted_style)
                total_by_class[label] += 1
                f_examples.write(
                    f'{predicted_style}: {speaker} [SEP] {listener} ({prob_string})\n'
                )

            # get counts and convert to percentages
            style_percentages = {}
            for label, by_label in styles_by_empathy_label.items():
                style_percentages[label] = {
                    style: count / total_by_class[label]
                    for style, count in Counter(by_label).items()
                }

            # compute deltas for emotions between classes
            f_counts.write(f'Difference for split {file_tag}\n')
            for label_A, label_B in permutations(empathy_labels):
                # deltas for speakers
                diffs = {}
                for emot in style_percentages[label_A].keys():
                    diffs[emot] = style_percentages[label_A].get(
                        emot, 0
                    ) - style_percentages[label_B].get(emot, 0)

                f_counts.write(f'Deltas between {label_A} and {label_B}\n')
                for emotion, diff in sorted(
                    diffs.items(), key=lambda x: x[1], reverse=True
                ):
                    fdiff = format(100 * diff, '.2f')
                    f_counts.write(f'{emotion} {fdiff} \n')


if __name__ == '__main__':
    main()
