# Mastering the Dungeon

This project was archived on 2020-08-01. The code had not been tested in a long time.
If you wish to view the old code, you may rewind to the
[`mastering_the_dungeon`](https://github.com/facebookresearch/ParlAI/tree/mastering_the_dungeon/projects/mastering_the_dungeon)
tag.

# Old README

This project contains the code we used in our paper:

[Mastering the Dungeon: Grounded Language Learning by Mechanical Turker Descent](https://arxiv.org/abs/1711.07950)

Zhilin Yang, Saizheng Zhang, Jack Urbanek, Will Feng, Alexander H. Miller, Arthur Szlam, Douwe Kiela, Jason Weston

## Requirements

Python 3.6, PyTorch 0.2, spacy

To install `spacy` and download related packages:
```
python -m pip install spacy
python -m spacy.en.download all
```

## Get the Data

Go to the ParlAI root directory. Create a data directory if it does not exist.
```
mkdir data
```

Then we can download the data
```
cd data
wget http://parl.ai/downloads/mastering_the_dungeon/mastering_the_dungeon.tgz
tar -xvzf mastering_the_dungeon.tgz
```

### Data Organization

The dataset is organized as follows. `data/graph_world2` contains the pilot study data, where each file ending with `.pkl` is a pickle file of an example. `data/graph_world2_v*_r*` contains a pickle file `filtered_data_list.pkl` storing the data collected in a specific round of a specific setting. The number after `r` indicates the round index. The number after `v` indicates the setting, where `v13` means `MTD limit`, `v14` means `MTD`, `v15` means `MTD limit w/o model`, and `BASELINE_2` means the baseline. For example, `data/graph_world2_v15_r2/filtered_data_list.pkl` contains the data collected in the second round of the setting `MTD limit w/o model`.

### Splitting the Data

We split the dataset before doing training and evaluation. Assuming you are under the ParlAI root directory,
```
cd projects/mastering_the_dungeon/mturk/tasks/MTD
python run.py --split
```

This will split the data from different rounds under different settings into training and test sets, while ensuring the number of training examples to be the same.

## Training and Evaluation

There are three steps: 1) start a GPU placeholder; 2) run the training and evaluation jobs; 3) terminate the GPU placeholder. We will illustrate the usage first on one single GPU, and later on a cluster with Slurm installed.

### Start GPU Placeholder

Open a new window (e.g. a screen session), and go to the ParlAI root directory.
```
cd projects/mastering_the_dungeon/projects/graph_world2
python train.py --job_num 0
```

This will start a GPU placeholder that accepts new training jobs on the fly.

### Run Training Jobs

Open another window (e.g. another screen session), and go to the ParlAI root directory.

#### Training
```
cd projects/mastering_the_dungeon/mturk/tasks/MTD
python run.py --train [--seq2seq]
```

This will train the models using AC-Seq2Seq (by default) or Seq2Seq (when seq2seq is specified).

#### Evaluation
```
cd projects/mastering_the_dungeon/mturk/tasks/MTD
python run.py --eval [--seq2seq] [--constrain]
```

This will evaluate the models just trained. The option `constrain` indicates using constrained decoding or not; turning it on is recommended.

#### Breakdown by Round
```
cd projects/mastering_the_dungeon/mturk/tasks/MTD
python run.py --rounds_breakdown [--seq2seq]
```

In this experiment, we will see the performance of agents in different rounds.

#### Breakdown by Dataset
```
cd projects/mastering_the_dungeon/mturk/tasks/MTD
python run.py --data_breakdown [--seq2seq]
```

In this experiment, we will see the performance of agents trained on one dataset and evaluated on another dataset.

#### Ablation Study
```
cd projects/mastering_the_dungeon/mturk/tasks/MTD
python run.py --ablation
```

This will do the ablation study by considering removing the counter feature and the room embeddings.

### Terminate GPU Placeholder

After all training and evaluation jobs are finished, we can now terminate the placeholders. Go back to the placeholder window, and press `Ctrl+C` to terminate the process.

### Running on GPU Cluster

There are a few changes to be made when running on a GPU cluster. In this section we assume [Slurm](https://slurm.schedmd.com/) is installed on the cluster.

When creating the GPU placeholder, we use the following commands:
```
cd projects/mastering_the_dungeon/projects/graph_world2
python gen_sbatch_script.py --num_gpus <num_gpus> --slurm
./batch_holder.sh
```
where `num_gpus` is the number of GPUs to use.

For training and evaluation, we need to add an additional option `--num_machines` to every command. For example, training now becomes:
```
cd projects/mastering_the_dungeon/mturk/tasks/MTD
python run.py --train --num_machines <num_gpus> [--seq2seq]
```

To terminate the GPU placeholder, simply cancel all jobs (this assumes you are not running other jobs using Slurm)
```
scancel -u <my_username_on_slurm>
```
