from parlai.mturk.tasks.pairwise_dialogue_eval.run import main as run_main, add_args

"""
Example script for running ACUTE-EVAL.
The only arguments that need to be modified for this to be run are:
'dialogs_path':  Path to folder containing logs of all model conversations.
    Each model should have one file named '<modelname>.jsonl'
'model_comparisons': List of tuples indicating pairs of models to be compared.
    The model names here should match the names of files in the folder
'onboarding_tasks': List of tuples in format (id1, id2, name of matchup) where
    id1, id2 are conversation ids where id2 is the correct conversation's id
Help strings for the other arguments can be found in run.py
"""

def set_args():
    args = add_args()
    args['dialogs_path'] = '/home/ParlAI/parlai/mturk/tasks/acute_eval/example'
    args['model_comparisons'] = [('modela', 'modelb')]
    args['onboarding_tasks'] = [('ZYX', 'AGHIJK', 'example_qual')]
    args['task_description'] = {
        'num_subtasks': 2,
        'question': args['question'],
        'get_task_feedback': True,
    }

    # Main ParlAI Mturk options
    args['num_conversations'] = int(
        len(args['model_comparisons'])
        * args['pairs_per_matchup']
        / (args['task_description']['num_subtasks'] - 1)
    ) # release enough hits to finish all annotations requested
    args['block_qualification'] = 'onboarding_qual_name'
    args['assignment_duration_in_seconds'] = 600
    args['reward'] = 0.5
    args['max_hits_per_worker'] = 1

    # Additional args that can be set - here we show the default values.
    # For a full list, refer to run.py & the parlai/params.py
    # args['is_sandbox'] = True
    # args['annotations_per_pair'] = 1
    # args['pairs_per_matchup'] = 160
    # args['seed'] = 42
    # args['s1_choice'] = 'I would prefer to talk to <Speaker 1>'
    # args['s2_choice'] = 'I would prefer to talk to <Speaker 2>'
    # args['question'] = 'Who would you prefer to talk to for a long conversation?'
    # args['block_on_onboarding'] = True

    return args


if __name__ == '__main__':
    args = set_args()
    run_main(args)
