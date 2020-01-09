#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


    all_conv_data = {}
    conv_ids_by_model = {}
    internal_id = 0

def add_args(from_argv=False):
    """ Add arguments to parser and either parse from commandline or initialize
    to defaults (for overriding in scripts)
    """
    argparser = ParlaiParser(False, False)
    argparser.add_argument(
        '--dialogs_path',
        type=str,
        default=None,
        help='path to folder with conversation log files for evaluation',
    )
    argparser.add_argument(
        '--model_comparisons',
        type=str,
        help='list of model pairs to compare, comma separated. E.g. ["transformer,human_eval"] ',
    )
    argparser.add_argument(
        '--pairs_per_comparison',
        type=int,
        default=160,
        help='Number of conversation pairs to generate for the comparison',
    )

def create_pairs():
    # read in all conversation data
    for data_fn in os.listdir(data_folder):
        full_data_fn = os.path.join(data_folder, data_fn)
        with open(full_data_fn, 'r') as dialog_data_file:
            model_name = data_fn.split('.')[0]
            model_convs = []
            conv_ids_by_model[model_name] = model_convs

            for l in dialog_data_file:
                try:
                    single_task_json = json.loads(l)
                except:
                    print("ILLEGAL FORMATTING:", l)
                    raise RuntimeError('One or more files are not valid .jsonl')
                id = single_task_json.get('conversation_id')
                all_conv_data[id] = single_task_json
                model_convs.append(id)

    if not opt['model_comparisons']:
        raise Exception("--model_comparisons must be set.")

    print('Creating random tasks for model pairs in --model_comparisons')
    for model_0, model_1 in opt['model_comparisons']:
        if model_0 not in conv_ids_by_model:
            raise Error("{} is not a valid model name".format(model0))
        if model_1 not in conv_ids_by_model:
            raise Error("{} is not a valid model name".format(model1))
        num_pairs = opt['pairs_per_comparison']
        matchup_name = '{},{}'.format(model_0, model_1)
        conv_pairs = []
        all_model1_convs = [
            id for id in conv_ids_by_model[model_0] if id not in onboarding_conv_ids
        ]
        all_model2_convs = [
            id for id in conv_ids_by_model[model_1] if id not in onboarding_conv_ids
        ]
        while len(conv_pairs) < num_pairs:
            id1 = np.random.choice(all_model1_convs)
            id2 = np.random.choice(all_model2_convs)
            if (id1, id2) in conv_pairs:
                continue
            conv_pairs.append((id1, id2))

            make_task_from_ids(
                id1,
                id2,
                internal_id,
                all_conv_data,
                opt['s1_choice'],
                opt['s2_choice'],
                opt['question'],
                matchup=matchup_name,
            )
            internal_id += 1
