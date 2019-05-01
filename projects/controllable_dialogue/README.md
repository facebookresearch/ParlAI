# Controllable Dialogue

## Paper information

Abigail See, Stephen Roller, Douwe Kiela, Jason Weston. [What makes a good
conversation? How controllable attributes affect human
judgments](https://arxiv.org/abs/1902.08654). To appear in NAACL 2019.


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

## Download the data
You can download all the training data by running
`projects/controllable_dialogue/tasks/build.py`.

This will download the following files to `data/controllable_dialogue`:

- `train.txt`: This is Convai2 training data, with extra annotations for three
  CT controllable attributes (`question`, `lastuttsim`, `avg_nidf`). It is in
  parlai format.
- `valid.txt`: Similarly to train.txt
- `arora.pkl`: This is a pickle file containing information necessary to
  compute Arora-style sentence embeddings, needed for the response-relatedness
  control methods
- `word2count.pkl`: This is a pickle file containing information necessary to
  compute NIDF measures, needed for the specificity control methods

In `ConvAI2_parlaiformat`, which is used if you want to generate the training
data yourself:
- `train.txt`: This is the ConvAI2 training set
  (`data/ConvAI2/train_self_original_no_cands.txt`) converted to parlai
  format
- `valid.txt`: This is the ConvAI2 validation set
  (`data/ConvAI2/valid_self_original_no_cands.txt`) converted to parlai
  format

### (Alternatively) Make the data yourself

Here are the commands to create ConvAI2\_parlaiformat and
controllable\_dialogue yourself:

First, convert the convai2 data to ParlAI format:

    cd ~/ParlAI
    mkdir -p data/controllable_dialogue/ConvAI2_parlaiformat

    python parlai/scripts/convert_data_to_parlai_format.py \
    -t convai2:SelfOriginal:no_cands -dt train:ordered \
    -of data/controllable_dialogue/ConvAI2_parlaiformat/train.txt

    python parlai/scripts/convert_data_to_parlai_format.py \
    -t convai2:SelfOriginal:no_cands -dt valid \
    -of data/controllable_dialogue/ConvAI2_parlaiformat/valid.txt

Next, create `word2count.pkl`:

    python projects/controllable_dialogue/controllable_seq2seq/nidf.py

This will create a file called `word2count.pkl` in your `controllable_dialogue`
directory. It might take a while, especially the part when it goes through the
Twitter dataset counting words. Once it's done, go to `nidf.py` and enter the
filepath to your `word2count.pkl` file.

Next, create `arora.pkl`:

    python projects/controllable_dialogue/controllable_seq2seq/arora.py

This will create a file called `arora.pkl` in your `controllable_dialogue`
directory. It might take a while -- in particular, it will download GloVe
vectors and store them in `ParlAI/data/controllable_dialogue/glove_vectors`,
which usually takes some time. Once it's done, go to `arora.py` and enter the
filepath to your `arora.pkl` file and your `glove_vectors` directory.

Next, create `controllable_dialogue/train.txt` and `valid.txt`:

    python projects/controllable_dialogue/make_control_dataset.py \
      --fromfile_datapath data/ConvAI2_parlaiformat/train.txt \
      --outfile data/controllable_dialogue/train.txt \
      --controls question,lastuttsim,avg_nidf

    python projects/controllable_dialogue/make_control_dataset.py \
      --fromfile_datapath data/ConvAI2_parlaiformat/valid.txt \
      --outfile data/controllable_dialogue/valid.txt \
      --controls question,lastuttsim,avg_nidf

This will create files called `train.txt` and `valid.txt` in your
`controllable_dialogue` directory.

Note: Due to changes in ParlAI, there might be some small differences between
the generated files obtained in this section, and the downloaded files in the
previous section.

## Download the pretrained models

Download the models and place them in `ParlAI/data/models/controllable_dialogue`.

TODO: update these instructions for downloading via ParlAI.

The directory `controllable_dialogue` contains the following models:

- `twitter_pretrained_baseline`: a seq2seq model trained on the Twitter
  dataset. TODO: is there a reference for where else this model can be
  downloaded?
- `convai2_finetuned_baseline`: the `twitter_pretrained_baseline` model, after
  fine-tuning on the ConvAI2 dataset.
- `control_avgnidf10b10e`: the `convai2_finetuned_baseline` model, after adding
  parameters for CT specificity control (10 buckets, embedding size 10), and
  fine-tuned on the ConvAI2 dataset with loss\_CT as described in Section 5.1 of
  the paper.
- `control_questionb11e10`: similarly to `control_avgnidf10b10e`, except this
  is CT question-asking control (11 buckets, embedding size 10).

The directory also contains a dictionary file:

- `dict_twit30k_train_split`: this is the dictionary used for all models. TODO:
  is there a reference for where else this dict can be downloaded?


## Chat with the pretrained models

In this section are the commands to talk to the models (as described in Table 5
of the paper).

Make sure you have downloaded the pretrained models as instructed in the
previous section.

Talk to the greedy search baseline model:

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline \
    --beam-size 1

Talk to the beam search baseline model:

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline

Talk to the repetition-controlled (WD) baseline:

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20

Talk to the question-controlled CT model (with WD repetition control):

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/control_questionb11e10 \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 \
    --set-controls question:7 # 70% questions. You can set this between 0 and 10.

Here's the "z=10 (boost)" version mentioned in the paper:

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/control_questionb11e10 \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 \
    --set-controls question:10 --beam-reorder best_extrep2gram_qn

Talk to the specificity-controlled CT model (with WD repetition control):

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/control_avgnidf10b10e \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 \
    --set-controls avg_nidf:7 # You can set this between 0 and 9.

Talk to the specificity-controlled WD model (with WD repetition control):

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:4 # You can set the nidf weight to be any real number, positive or negative

Talk to the response-relatedness WD model (with WD repetition control):

    python projects/controllable_dialogue/interactive.py \
    -mf models:controllable_dialogue/convai2_finetuned_baseline \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_2gram:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,lastuttsim:5 # You can set the lastuttsim weight to be any real number, positive or negative

Note: If you want the bot to have a persona when you talk to it, select one of
the lines in `data/controllable_dialogue/personas_validation.txt` and prepend
it to your first utterance.

Note: If you want to see the top 10 candidates produced by beam search (rather
than just the top 1), add the flag `--verbose True`.

## Train a CT model

To train a CT model (conditioned on mean NIDF) from scratch:

    # TODO update task
    cd ParlAI
    python projects/controllable_dialogue/train_controllable_seq2seq.py \
    --fromfile-datapath data/controllable_dialogue/train.txt \
    --fromfile-datapath2 data/controllable_dialogue/valid.txt \
    --dict-file models:controllable_dialogue/dict_twit30k_train_split \
    -mf /path/to/your/modelfile \
    --control-vars avg_nidf

Note: if you add your paths for `train.txt`, `valid.txt` and
`dict_twit30k_train_split` to `train_controllable_seq2seq.py` then you won't
need to enter them via command line.

This will default to embedding size 10, but you could include e.g.
`--control-embeddingsize 15` if you wanted to change it.

This code will also default to 10 NIDF buckets. If you want to use a different
number of buckets, first you need to figure out what the NIDF lower bound
should be for each bucket. Suppose you want 8 buckets. First run:

    #TODO: update task
    python projects/controllable_dialogue/get_bucket_lowerbounds.py \
      --fromfile_datapath data/controllable_dialogue/train.txt \
      --num_buckets 8 \
      --control avg_nidf

and then copy and paste the provided lower bounds into
`controllable_dialogue/controllable_seq2seq/controls.py`, similarly to the
existing `AVG_NIDF_10BUCKET_LBS`. Then you can train a model with
`--control-num-buckets 8`.

You can train a CT model conditioned on multiple controls:

    python projects/controllable_dialogue/train_controllable_seq2seq.py \
      --fromfile-datapath data/controllable_dialogue/train.txt \
      --fromfile-datapath2 data/controllable_dialogue/valid.txt \
      --dict-file models:controllable_dialogue/dict_twit30k_train_split \
      -mf /path/to/your/modelfile \
      --control-vars avg_nidf,question

To take an existing non-CT model (e.g. the baseline) and the finetune it as a
CT model do this:

    python projects/controllable_dialogue/train_controllable_seq2seq.py \
    --fromfile-datapath data/controllable_dialogue/train.txt \
    --fromfile-datapath2 data/controllable_dialogue/valid.txt \
    --dict-file models:controllable_dialogue/dict_twit30k_train_split \
    -mf /path/to/your/modelfile \
    --init-model models:controllable_dialogue/convai2_finetuned_baseline \
    --add-control True \
    -cv avg_nidf

This command will take the parameters saved in `--init-model`, load them in the
new model (which has randomly initialized weights for the new CT parameters),
and then save that model to the given modelfile (`-mf`). It should be quick.
Once that's done, run this command:

    python projects/controllable_dialogue/train_controllable_seq2seq.py \
    --fromfile-datapath data/controllable_dialogue/train.txt \
    --fromfile-datapath2 data/controllable_dialogue/valid.txt \
    --dict-file models:controllable_dialogue/dict_twit30k_train_split \
    -mf /path/to/your/modelfile \
    --add-control False \
    -cv avg_nidf

You should see your new CT model training. Note: this is how the models in the
paper were trained.

## Look at output and automatic metrics

First, download and extract [this
file](https://drive.google.com/open?id=16kXJXJ8WiSrUpAsShYcuzJhToVBrnf30) to
obtain a directory called `wordstat_files`. The directory contains json files
with the generated output and the automatic metrics computed for the pretrained
models.

Run the following:

    cd projects/controllable_dialogue
    jupyter notebook

and then open up `inspect_wordstats.ipynb`. Where it says `models_dir`, enter
the path to your `wordstat_files` directory. You will be able to recreate the
table of automatic metrics from the paper (Table 6), and you can explore the
output of the models.

TODO: <screenshot>

## Measure automatic metrics

If you want to generate json files like those in the previous section, do this:

    python projects/controllable_dialogue/eval_wordstat.py \
    --fromfile_datapath data/controllable_dialogue/valid.txt \
    -mf models:controllable_dialogue/control_questionb11e10 \
    -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 \
    --set-controls question:7

This will create a json file containing the output and other automatic metrics
for the question-controlled CT model with z=7. The script `eval_wordstat.py`
always places the json file in the same place as the model file. The script can
take a while to complete.

Note: Due to changes in ParlAI, there might be some small differences between
the json file created via this method, and the json files downloadable in the
previous section.

## Human Evaluation Logs
TODO

## Human eval logs
TODO

## Bayesian calibration code
TODO
