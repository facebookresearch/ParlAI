#!/usr/bin/env python3

# for n in 50000; do CUDA_VISIBLE_DEVICES=`free-gpu -n 2` python -m parlai.scripts.train_model -t parlai.projects.metacognition.agents:CorrectnessProbingTeacher --model parlai.projects.metacognition.agents:ClassifierOnGeneratorAgent --freeze-enc-dec-weights True --classes WRONG RIGHT --simplify-correctness True --init-model zoo:blender/blender_3B/model --dict-file zoo:blender/blender_3B/model.dict --dict-tokenizer bytelevelbpe --optimizer adam -lr 1e-05 --lr-scheduler reduceonplateau --activation gelu --validation-metric ALL_RIGHT_f1 --validation-metric-mode max --validation-patience 20 --validation-max-exs -1 --validation-every-n-secs 120 --log-every-n-secs 10 --max-train-time 5400 --load-from-checkpoint false --save-after-valid true --update-freq 1 --text-truncate 128 --label-truncate 128 --truncate 128 --fp16 true --fp16-impl mem_efficient --model-parallel true -bs 128 --model-file /some/storage/post_mcprobe_station/claim${n}/model  --embedding-size 2560 --ffn-size 10240 --variant prelayernorm --n-heads 32 --n-positions 128 --n-encoder-layers 2 --n-decoder-layers 24  --warmup-updates 100 --claimed-data ${n} --n-classifier-layers 2 --classifier-hidsize 256 --classifier-state-pooling max --classifier-state-pre-pooling linearGELU --classifier_with_decode True --classifier_with_encode True --balance-correctness anycorrectness --certainty-distribution everything-oversample --with-eva False; done

# for n in 50000; do CUDA_VISIBLE_DEVICES=`free-gpu -n 2` parlai eval -dt valid -t parlai.projects.metacognition.agents:CorrectnessProbingTeacher --model parlai.projects.metacognition.agents:ClassifierOnGeneratorAgent --freeze-enc-dec-weights True --classes WRONG RIGHT --claimed-data ${n} --n-classifier-layers 2 --classifier-hidsize 256 --classifier-state-pooling max --classifier-state-pre-pooling linearGELU --simplify-correctness True --balance-correctness anycorrectness --certainty-distribution everything-oversample --classifier_with_decode True --classifier_with_encode True --with-eva False --dict-file zoo:blender/blender_3B/model.dict --dict-tokenizer bytelevelbpe --activation gelu --embedding-size 2560 --ffn-size 10240 --variant prelayernorm --n-heads 32 --n-positions 128 --n-encoder-layers 2 --n-decoder-layers 24 --text-truncate 128 --label-truncate 128 --truncate 128 --fp16 true --fp16-impl mem_efficient --model-parallel true -bs 128 --model-file /some/storage/post_mcprobe_station/claim${n}/model --save-world-logs True --report-filename /some/storage/probe_station_claim${n}_says --inference beam; done

HOURS = 12

grid = {
    '-t': ['parlai.projects.metacognition.agents:CorrectnessProbingTeacher'],
    '--model': [
        'parlai.projects.metacognition.agents:ClassifierOnGeneratorAgent'
    ],
    '--freeze-enc-dec-weights': [True],
    '--classes': ['WRONG RIGHT'],
    '--claimed-data': [50000],
    '--n-classifier-layers': [2],
    '--classifier-hidsize': [256],
    '--classifier-state-pooling': ["max"],
    '--classifier-state-pre-pooling': ["linearGELU"],
    "--classifier_with_decode": [True],
    "--classifier_with_encode": [True],
    '--simplify-correctness': [True],
    '--balance-correctness': ["anycorrectness"],
    '--certainty-distribution': ["everything-oversample"],
    '--with-eva': [False],
    '--init-model': ['zoo:blender/blender_3B/model'],
    '--dict-file': ['zoo:blender/blender_3B/model.dict'],
    '--dict-tokenizer': ['bytelevelbpe'],
    '--optimizer': ['adam'],
    '-lr': [1e-6],
    '--lr-scheduler': ['reduceonplateau'],
    '--activation': ['gelu'],
    '--embedding-size': ['2560'],
    '--ffn-size': ['10240'],
    '--variant': ['prelayernorm'],
    '--n-heads': ['32'],
    '--n-positions': ['128'],
    '--n-encoder-layers': ['2'],
    '--n-decoder-layers': ['24'],
    '--dropout': [0.0],
    '--relu-dropout': [0.0],
    '--gradient-clip': [0.1],
    '--validation-metric': ['ALL_RIGHT_f1'],
    '--validation-metric-mode': ['max'],
    '--validation-patience': [20],
    '--validation-max-exs': [-1],
    '--validation-every-n-secs': [600],
    '--log-every-n-secs': [10],
    '--max-train-time': [(HOURS * 60 - 30) * 60],
    '--save-after-valid': ['true'],
    '--update-freq': ['1'],
    '--text-truncate': ['128'],
    '--label-truncate': ['128'],
    '--truncate': ['128'],
    '--fp16': ['true'],
    '--fp16-impl': ['mem_efficient'],
    '--model-parallel': ['true'],
    '--warmup-updates': [500],
    '-bs': [128],
}

if __name__ == '__main__':
    run_grid(
        grid,
        {},
        'mcprobe_luca',
        gpus=2,
        data_parallel=True,
        nodes=1,
        jobtime='{}:00:00'.format(HOURS),
    )
