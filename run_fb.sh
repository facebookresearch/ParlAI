#!/bin/bash
# TODO(ahm): remove this file, it's just for experimenting in fbcode

if [[ $# -eq 0 ]]; then
    echo "No arguments supplied: provide a parameter for which dataset to run (e.g. babi, wikiqa)"
    exit 1
fi

args="${*:2}"
remote_cmd="--remote-cmd /data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/projects/parlai/parlai/agents/memnn_fb/memnn_zmq.lex"
remote_args_prefix="--remote-args fbcode.deeplearning.projects.parlai.parlai.examples.memnn_fb.params"

if [[ $1 == 'babi' ]]; then
    /data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/projects/parlai/parlai/examples/memnn_fb/babi_train.par \
        $remote_cmd $remote_args_prefix.params_babi \
        $args
elif [[ $1 == 'hogbabi' ]]; then
    /data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/projects/parlai/parlai/examples/memnn_fb/babi_train_hogwild.par \
        $remote_cmd $remote_args_prefix.params_hogbabi \
        $args
elif [[ $1 == 'wikiqa' ]]; then
    /data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/projects/parlai/parlai/examples/memnn_fb/wikiqa_train.par \
        $remote_cmd $remote_args_prefix.params_wikiqa \
        $args
elif [[ $1 == 'hogwikiqa' ]]; then
    /data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/projects/parlai/parlai/examples/memnn_fb/wikiqa_train_hogwild.par \
        $remote_cmd $remote_args_prefix.params_wikiqa \
        $args
elif [[ $1 == 'display' ]]; then
    /data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/projects/parlai/parlai/examples/display_data.par \
        $args
elif [[ $1 == 'dict' ]]; then
    /data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/projects/parlai/parlai/examples/build_dict.par \
        $args
elif [[ $1 == 'full' ]]; then
    /data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/projects/parlai/parlai/examples/memnn_fb/full_task_train.par \
        --remote-cmd "/data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/projects/parlai/parlai/agents/memnn_fb/memnn_zmq_parsed.lex" \
        $remote_args_prefix.params_default \
        --dict-savepath /tmp/dict.txt \
        $args
else
    echo "$0: $1 is not a valid param"
fi
