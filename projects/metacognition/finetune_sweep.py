#!/usr/bin/env python3

max_hours = 36

# best stage1: alcest_68c (freebeam: controlprob: 0.9, lr: 1e-5) / alcest_94c (default)
# best stage2: linkinpark_9d5 (from oathbreaker_448)
grid = {
    '--activation': ['gelu'],
    '--attention-dropout': [0.0],
    '--balance-correctness': ["onlycorrect", "balancedcorrectness"],  # alcest used onlycorrect
    '--batchsize': [128],
    '--certainty-distribution': ["natural-oversample"],
    '--claimed-data': [25000],
    '--controlprob': [0.8, 1.0], # both liked 1.0
    '--dict-file': [
        '/checkpoint/parlai/zoo/meena/20200319_meenav0data_tall_2.7B_adamoptimizer/20200319_13.3ppl_200kupdates/model.dict'
    ],
    '--dict-tokenizer': ['bytelevelbpe'],
    '--dir-annotations': ["/some/storage/parlai/projects/metacognition/annotations"],
    '--dir-runs': ["/some/storage"],
    '--dropout': [0.2],
    '--embedding-size': [2560],
    '--ffn-size': [10240],
    '--fp16': [True],
    '--fp16-impl': ['mem_efficient'],
    '--gradient-clip': [0.1],
    '--hf-skip-special-tokens': [False],
    '--history-add-global-end-token': ['end'],
    '--init-model': [
        '/checkpoint/parlai/zoo/meena/20200319_meenav0data_tall_2.7B_adamoptimizer/20200319_13.3ppl_200kupdates/model'
    ],
    '--label-truncate': [128],
    '--log-every-n-secs': [30],
    '-lr': [1e-6, 7e-6, 1e-5],  # both liked 7e-6
    '--lr-scheduler': ['reduceonplateau'],
    '--lr-scheduler-patience': [3],
    '--max-train-time': [0.95 * max_hours * 60 * 60],
    '--model': [
        'parlai.projects.metacognition.agents:ControllingTransformerGeneratorAgent'
    ],
    '--model-parallel': [True],
    '--multitask-weights': ['1,3,3,3,9', '1,3,3,3,3', '1,3,3,3,5'],  # both liked 5
    '--n-decoder-layers': [24],
    '--n-encoder-layers': [2],
    '--n-heads': [32],
    '--n-positions': [128],
    '--num-epochs': [4],
    '--optimizer': ['adam'],
    '--relu-dropout': [0.0],
    '--save-after-valid': [True],
    '--skip-generation': [True],
    '--stage1-results': ["alcest_94c"],
    # '--stage0-free-beam': [True, False],
    '-t': [
        'blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues,parlai.projects.metacognition.agents:CertaintyControlTeacher'
    ],
    '--text-truncate': [128],
    '--truncate': [128],
    '--update-freq': [2],
    '--variant': ['prelayernorm'],
    '--warmup_updates': [100],
    '--with-eva': [False],
    '--with-fished': [0.0],
    '-veps': [0.25],
    '-vmm': ['min'],
    '-vmt': ['ppl'],
    '-vp': [10],
}

if __name__ == '__main__':
    run_grid(
        grid=grid,  # {k: [v[-1]] for k, v in grid.items()},
        name_keys={},
        sweep_name="sweep_mccfts2_alcest",
        jobtime=f'{max_hours}:00:00',
        gpus=8,
        nodes=1,
    )
