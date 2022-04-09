#!/usr/bin/env python3

# parlai eval -dt valid:stream -t parlai.projects.metacognition.agents:CertaintyOntoRedditTeacher --simplify-certainty False --classes '<IDK>' '<TRY>' '<YEA>' '<EVA>' --model bert_classifier --save-world-logs True --report-filename ~/reddit_full_166 -bs 100 --ncomments 12000 -mf /some/storage/202009*/sweep_mccc_full_bigthief/166/model; parlai eval -dt valid:stream -t parlai.projects.metacognition.agents:CertaintyOntoRedditTeacher --simplify-certainty True --classes '<IDK>' '<YEA>' '<EVA>' --model bert_classifier --save-world-logs True --report-filename ~/reddit_simp_b7f -bs 100 --ncomments 12000 -mf /some/storage/202009*/sweep_mccc_simp_bigthief/b7f/model

if __name__ == '__main__':
    # best: bigthief/166 .7289
    run_grid(
        grid={
            '-bs': [200],
            '--classes': ["'<IDK>' '<TRY>' '<YEA>' '<EVA>'"],
            '--classify-with-prediction': [True],
            '--learningrate': [1e-6],
            '--max-train-time': [2.5 * 60 * 60],
            '--model': ['bert_classifier'],
            '--optimizer': ['adam'],
            '--simplify-certainty': [False],
            '-t': [
                'parlai.projects.metacognition.agents:CertaintyClassificationTeacher'
            ],
            '--type-optimization': ['top_layer'],
        },
        sweep_name="sweep_mccc_full_bigthief",
        jobtime=f'3:00:00',
        gpus=1,
        nodes=1,
    )
    # best: bigthief/b7f .9571
    run_grid(
        grid={
            '-bs': [200],
            '--classes': ["'<IDK>' '<YEA>' '<EVA>'"],
            '--classify-with-prediction': [True],
            '--learningrate': [1e-4],
            '--max-train-time': [2.5 * 60 * 60],
            '--model': ['bert_classifier'],
            '--optimizer': ['sgd'],
            '--simplify-certainty': [True],
            '-t': [
                'parlai.projects.metacognition.agents:CertaintyClassificationTeacher'
            ],
            '--type-optimization': ['top_layer'],
        },
        sweep_name="sweep_mccc_simp_bigthief",
        jobtime=f'3:00:00',
        gpus=1,
        nodes=1,
    )
