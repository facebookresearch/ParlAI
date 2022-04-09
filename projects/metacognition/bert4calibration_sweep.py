#!/usr/bin/env python3

if __name__ == '__main__':
    run_grid(
        grid={
            '-bs': [32, 64, 128, 256],
            '--classes': ["'WRONG' 'RIGHT"],
            '--classify-with-prediction': [True],
            '--claimed-data': [50000],
            '--max-train-time': [8 * 60 * 60],
            '--model': ['bert_classifier'],
            '--optimizer': ['adam'],
            '--simplify-correctness': [True],
            '-t': [
                'parlai.projects.metacognition.agents:CorrectnessProbingTeacher'
            ],
            '--type-optimization': ['additional_layers', 'top_layer', 'top4_layers'],  # so far done: top_layer
            '--balance-correctness': ["anycorrectness", "balancedcorrectness"],  # so far done: anycorrectness
            '--certainty-distribution': ["everything-oversample"],
            '--with-eva': [False],
            '-veps': [1],
            '--fp16': ['true'],
            '--fp16-impl': ['mem_efficient'],
            '--learningrate': [1e-7, 1e-6, 1e-5],  # so far done: 1e-6
            '--correctness-prediction-mode': ["bert-qp"],
        },
        sweep_name="sweep_mccc_full_bigthief",
        jobtime=f'9:00:00',
        gpus=1,
        nodes=1,
    )