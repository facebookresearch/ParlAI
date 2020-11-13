# Controllable Dialogue

## Paper information

Abigail See, Stephen Roller, Douwe Kiela, Jason Weston. 
_[What makes a good conversation? How controllable attributes affect human
judgments](https://arxiv.org/abs/1902.08654)_. 
To appear in NAACL 2019.


## Abstract

A good conversation requires balance -- between simplicity and detail; staying
on topic and changing it; asking questions and answering them. Although
dialogue agents are commonly evaluated via human judgments of overall quality,
the relationship between quality and these individual factors is less
well-studied. In this work, we examine two controllable neural text generation
methods, conditional training and weighted decoding, in order to control four
important attributes for chitchat dialogue: repetition, specificity,
response-relatedness and question-asking. We conduct a large-scale human
evaluation to measure the effect of these control parameters on multi-turn
interactive conversations on the PersonaChat task. We provide a detailed
analysis of their relationship to high-level aspects of conversation, and show
that by controlling combinations of these variables our models obtain clear
improvements in human quality judgments.

## Citation

If you use the dataset or models in your own work, please cite with the
following BibTex entry:

    @inproceedings{see2019what,
      author={Abigail See and Stephen Roller and Douwe Kiela and Jason Weston},
      booktitle={North American Chapter of the Association for Computational Linguistics (NAACL)},
      title={What makes a good conversation? How controllable attributes affect human judgments},
      url={https://arxiv.org/abs/1902.08654},
      year={2019},
    }

# Code Instructions
Once you have [installed ParlAI](https://github.com/facebookresearch/ParlAI/#installing-parlai), 
follow the instructions below.

## Download the data

Running the commands to train or chat with the models will automatically download
the data for you. Alternatively, you can manually download the data by
running `python projects/controllable_dialogue/tasks/build.py`. This will download
the following files to `data/controllable_dialogue`:

- `train.txt`: This is Convai2 training data, with extra annotations for three
  CT controllable attributes (`question`, `lastuttsim`, `avg_nidf`). It is in
  parlai format.
- `valid.txt`: Similarly to train.txt.
- `arora.pkl`: This is a pickle file containing information necessary to
  compute Arora-style sentence embeddings, needed for the response-relatedness
  control methods.
- `word2count.pkl`: This is a pickle file containing information necessary to
  compute NIDF measures, needed for the specificity control methods.
- `personas_validation.txt`: This file contains all the ConvAI2 validation set
personas, provided for convenience (useful for talking to the model interactively).
- `ConvAI2_parlaiformat/`:
    - `train.txt`: This is the ConvAI2 training set
      (`data/ConvAI2/train_self_original_no_cands.txt`) converted to parlai
      format. 
    - `valid.txt`: This is the ConvAI2 validation set
      (`data/ConvAI2/valid_self_original_no_cands.txt`) converted to parlai
      format.
- `wordstat_files/`: This directory contains json files with generated output and
automatic metrics computed for the various pretrained models.
- `evaluation_logs/`: This directory contains logs and evaluations from the human
   evaluations.

### (Alternatively) Making the data yourself

For reproducibility, in this section we provide the commands to create
the data yourself.

_Note: Due to changes in ParlAI, there might be some small differences between
the generated files obtained in this section, and the downloaded files in the
previous section._

First, convert the ConvAI2 data to ParlAI format:

    mkdir -p data/controllable_dialogue/ConvAI2_parlaiformat

    python parlai/scripts/convert_data_to_parlai_format.py \
    --task convai2:SelfOriginal:no_cands \
    --datatype train:ordered \
    --outfile data/controllable_dialogue/ConvAI2_parlaiformat/train.txt

    python parlai/scripts/convert_data_to_parlai_format.py \
    --task convai2:SelfOriginal:no_cands \
    --datatype valid \
    --outfile data/controllable_dialogue/ConvAI2_parlaiformat/valid.txt

Next, create `word2count.pkl`:

    python projects/controllable_dialogue/controllable_seq2seq/nidf.py

This will create a file called `word2count.pkl` in your `data/controllable_dialogue`
directory. It might take a while, especially the part when it goes through the
Twitter dataset counting words.

Next, create `arora.pkl`:

    python projects/controllable_dialogue/controllable_seq2seq/arora.py

This will create a file called `arora.pkl` in your `data/controllable_dialogue`
directory. It might take a while - in particular, if necessary it will download
GloVe vectors and store them in `ParlAI/data/models/glove_vectors`.

Next, create `data/controllable_dialogue/train.txt` and `valid.txt`:

    python projects/controllable_dialogue/make_control_dataset.py \
    --fromfile_datapath data/controllable_dialogue/ConvAI2_parlaiformat/train.txt \
    --outfile data/controllable_dialogue/train.txt \
    --controls question,lastuttsim,avg_nidf

    python projects/controllable_dialogue/make_control_dataset.py \
    --fromfile_datapath data/controllable_dialogue/ConvAI2_parlaiformat/valid.txt \
    --outfile data/controllable_dialogue/valid.txt \
    --controls question,lastuttsim,avg_nidf

This will create files called `train.txt` and `valid.txt` in your
`data/controllable_dialogue` directory.

## The pretrained models

Running the commands in the next section to chat with the pretrained models will
automatically download them for you. In `data/models/controllable_dialogue` you will
find the following models, along with their `.opt` files:

- `twitter_pretrained_baseline`: A seq2seq model trained on the Twitter dataset.
- `convai2_finetuned_baseline`: The `twitter_pretrained_baseline` model, after
  fine-tuning on the ConvAI2 dataset.
- `control_avgnidf10b10e`: The `convai2_finetuned_baseline` model, after adding
  parameters for CT specificity control (10 buckets, embedding size 10), and
  fine-tuned on the ConvAI2 dataset with loss\_CT as described in Section 5.1 of
  the paper.
- `control_questionb11e10`: Similarly to `control_avgnidf10b10e`, except this
  is CT question-asking control (11 buckets, embedding size 10).

The directory also contains a dictionary file:

- `dict_twit30k_train_split`: This is the dictionary used for all models.

## Chat with the pretrained models

This section provides the commands to talk to the model configurations
described in the paper. You can refer to Table 5 in the paper to see how these
commands correspond to the configurations described there.

Running any of these commands will also download the pretrained models, if necessary.

**Talk to the greedy search baseline model:**

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline \
    --beam-size 1

**Talk to the beam search baseline model:**

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline

This setting uses beam size 20 by default.

**Talk to the repetition-controlled (WD) baseline:**

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20

You can change the weights for these three WD repetition features to be any
real number (positive or negative). Here `-1e20` represents -infinity.  In
addition, there are other repetition WD features you can use if you wish: see
the keys of `WDFEATURE2UPDATEFN` in `controllable_seq2seq/controls.py`.

**Talk to the question-controlled CT model (with WD repetition control):**

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/control_questionb11e10 \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 \
    --set-controls question:7

Here `question:7` means the '70% questions' bucket.
You can set this anywhere between 0 and 10.

To talk to the "z=10 (boost)" version mentioned in the paper:

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/control_questionb11e10 \
    -wd extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 \
    --set-controls question:10 --beam-reorder best_extrep2gram_qn

**Talk to the specificity-controlled CT model (with WD repetition control):**

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/control_avgnidf10b10e \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 \
    --set-controls avg_nidf:7

Here `avg_nidf:7` means the 7th specificity bucket (where higher is more specific).
You can set this anywhere between 0 and 9.

**Talk to the specificity-controlled WD model (with WD repetition control):**

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:4

Here `nidf:4` means using the NIDF WD feature with weight 4.
You can use any real number as a weight (positive or negative).

**Talk to the response-relatedness WD model (with WD repetition control):**

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_2gram:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,lastuttsim:5

Here `lastuttsim:5` means using the response-relatedness WD feature with weight 5.
You can use any real number as a weight (positive or negative).

Note that this this feature can take a while to load, especially the first time
you run it.  This is because we have to load the GloVe vectors from file.

**Giving the bot a persona**:
If you want the bot to have a persona when you talk to it, select one of the
lines in `data/controllable_dialogue/personas_validation.txt` and prepend it to
your first utterance. Alternatively you can write the persona yourself - but
make sure to use the same format.

**Viewing top 10 beam search candidates:**
If you want to see the top 10 candidates produced by beam search (rather
than just the top 1), add the flag `--verbose True`.

## Train a CT model

**To train a CT model from scratch:**

    python projects/controllable_dialogue/train_controllable_seq2seq.py \
    -mf /path/to/your/modelfile \
    --control-vars avg_nidf

Here we are training a specificity-controlled CT model.

**To change control embedding size:** The CT control embedding size will
default to 10, but you could include e.g. `--control-embeddingsize 15` if you
wanted to change it.

**To change number of buckets:** For `avg_nidf`, the number of buckets will
default to 10.  If you want to use a different number of buckets, first you
need to figure out what the NIDF lower bound should be for each bucket. Suppose
you want 8 buckets. First run:

    python projects/controllable_dialogue/get_bucket_lowerbounds.py \
    --num_buckets 8 \
    --control-vars avg_nidf

and then copy and paste the provided lower bounds into
`projects/controllable_dialogue/controllable_seq2seq/controls.py`, similarly to
the existing `AVG_NIDF_10BUCKET_LBS`. Then you can train a model with
`--control-num-buckets 8`.

**To train a CT model on _multiple_ controls:**

    python projects/controllable_dialogue/train_controllable_seq2seq.py \
    -mf /path/to/your/modelfile \
    --control-vars avg_nidf,question

Here we are training a model conditioned on specificity and question-asking.

**To take an existing non-CT model and finetune it as a CT model:**
First, run this command (in this example, taking the ConvAI2-finetuned
baseline and adding specificity control):

    python projects/controllable_dialogue/train_controllable_seq2seq.py \
    -mf /path/to/your/modelfile \
    --init-model models:controllable_dialogue/convai2_finetuned_baseline \
    --add-control True \
    --control-vars avg_nidf

This command will take the parameters saved in `--init-model`, load them in the
new model (which has randomly initialized weights for the new CT parameters),
and then save that model to the given modelfile (`-mf`). It should be quick.
Once that's done, run this command:

    python projects/controllable_dialogue/train_controllable_seq2seq.py \
    -mf /path/to/your/modelfile \
    --add-control False \
    --control-vars avg_nidf

You should see your new CT model training. Note: this is how the models in the
paper were trained.

## Look at generated output and automatic metrics

Once you have downloaded the data, you will find a
directory `wordstat_files` in `data/controllable_dialogue`.
The json files in this directory contain the generated output computed on the
ConvAI2 validation set, plus the corresponding automatic metrics.
Each json file corresponds to a different model configuration.

Run the following:

    cd projects/controllable_dialogue
    jupyter notebook

and then open up `inspect_wordstats.ipynb`. Where it says `models_dir`, enter
the path to your `wordstat_files` directory. You will be able to recreate the
table of automatic metrics from the paper (Table 6), and explore the models'
generated output.

## Save generated output and automatic metrics to file

If you want to generate json files like those in the previous section, run a
command like this:

    python projects/controllable_dialogue/eval_wordstat.py \
    -mf models:controllable_dialogue/control_questionb11e10 \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 \
    --set-controls question:7

This will create a json file containing the output and automatic metrics for
the provided model configuration (here, question-controlled CT model with z=7
and WD repetition control).  The script `eval_wordstat.py` always places the
json file in the same place as the model file.  The script can take a while to
complete - so you can set e.g. `--num-examples 512` to generate output on a
smaller number of examples.

_Note: Due to changes in ParlAI, there might be some small differences between
the json file created via this method, and the json files downloadable in the
previous section._

## Human Evaluation code, logs, and analysis

Human evaluation logs should be downloaded automatically after following the
download instructions above. You'll find them in the `evaluation_logs/` folder.

A Jupyter notebook which generates the graphs and tables for the human experiments
is available in the
[project folder](https://github.com/facebookresearch/ParlAI/tree/master/projects/controllable_dialogue).
The notebook should be launched from the ParlAI root directory.

### Running evaluations

ParlAI is currently in the process of upgrading MTurk to
[Mephisto](https://github.com/facebookresearch/Mephisto). As part of this
process we have archived a number of older tasks. If you require need to run
this evaluation, you may rewind back to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag:

```bash
git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
cd ~/ParlAI
git checkout mturk_archive
```

The code for running your own mechanical turk evaluations is also available in
the corresponding
[mturk folder](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/projects/controllable_dialogue/mturk).
You will probably want to make changes to the `model_config.py` and `run.py` to change
which models are being evaluated, and then you can launch the experiment with:

```
python parlai/mturk/tasks/controllable_dialogue/run.py -r 0.9 --count-complete --hobby --max-resp-time 1200 --max-connections 20 -nc 1200 --sandbox
```
Change it to `--live` if you're prepared to spend actual currency. The output must be
lightly postprocessed to use it with the analysis tools released. If you intend to do
this, please file an issue on the
[ParlAI GitHub](https://github.com/facebookresearch/ParlAI/).

