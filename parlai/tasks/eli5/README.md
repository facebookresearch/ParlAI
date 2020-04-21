# ELI5
Adapted from the Facebook Research found at https://github.com/facebookresearch/ELI5/

![GitHub license](https://img.shields.io/badge/license-BSD-blue.svg)


Read the Paper: https://arxiv.org/abs/1907.09190

Read the Blog Post: https://ai.facebook.com/blog/longform-qa/

Explore the Dataset: https://facebookresearch.github.io/ELI5/explore.html

--------------------------------------------------------------------------------

# Data creation

We provide a suite of scripts to download paired questions and answers from the ELI5 subreddit along with supporting documents from the CommonCrawl

### FAQ: can you provide the processed data?

No, we are not allowed to host processed Reddit or CommonCrawl data. While we are aware that it would make life a little easier, making your own version of the dataset following the instructions here is pretty straightforward if you have access to a SLURM cluster. **We're also happy to work with you if your cluster works on another operating system, open an issue and we'll get on it!** Since the creation process involves downloading, filtering and tokenizing a full CommonCrawl dump however, using a single machine is impractical.

## Downloading pre-computed files for support documents
Running `python examples/display_data.py -t eli5` to run ELI5's `build.py` file
will create the directory structure and download pre-computed files such as the list of CommonCrawl IDs for supporting documents needed for the dataset. Unlike a normal `build.py` file, this will not download the ELI5 datset, and instead tell you to follow these instructions.


## Overview of the data creation process

The process consists of three steps. *Steps 1 and 2 should be run in parallel.*

1. **Downloading and filtering the Reddit data.** This can be run on a single machine and may take up to 72 hours.
2. **Downloading and tokenizing the CommonCrawl pages.** This part requires access to a cluster. We provide a sample SLURM script using 100 threads, which on our cluster finishes in under 48 hours.
3. **Selecting passages from the downloaded pages to create the final support document.** After running steps 1 and 2, this part uses our TFIDF heuristic to create the final ~1000 words support document, and create a train/valid/test split of Question-Document-Answer triplets.

If you are having trouble with any of these, please open an issue stating which step is failing in the title, and attach the Python error text, if available.


## Downloading the Reddit Data

The first step consists in downloading the Reddit Data from the files provided at pushshift.io for all months from 07/2011 to 07/2018. This is done by running the following
from the top-level ParlAI directory:

```
python parlai/tasks/eli5/download_reddit_qalist.py -Q -o [path/to/data/folder]/eli5
python parlai/tasks/eli5/download_reddit_qalist.py -A -o [path/to/data/folder]/eli5
```
For example, if your `data` folder is found in the top ParlAI directory:
```
python parlai/tasks/eli5/download_reddit_qalist.py -Q -o data/eli5
python parlai/tasks/eli5/download_reddit_qalist.py -A -o data/eli5
```

The first line takes about 6 hours on one machine to download the questions, and the second less than 48 hours for the answers. Pushshift files are automatically removed after they've been processed, so space shouldn't be an issue there. The final product should be 689MB.


## Downloading support documents from the CommonCrawl

We provide a list of CommonCrawl IDs for supporting documents for each of the questions.
These are obtained from `build.py` which we went over above.


The next step than consists in reading through the CommonCrawl WET files to gather the text of pages which are used as support documents. In order to gather the documents for each QA pair (up to 100 per pair, sometimes less after deduplicating) using a SLURM cluster and 100 threads, run:
```
cd slurm_scripts
./eli_download_docs_launcher.sh
```
This should run in less than 48 hours. Be advised that the result is upwards of 100GB.

When you have downloaded the selected pages from all of the CommonCrawl slices, simply merge the slices from all the threads with:
```
./eli_merge_docs_launcher.sh
cd ..
python merge_support_docs.py explainlikeimfive finalize
```

### FAQ: my SLURM cluster is not very stable and some of the threads are interrupted. Do I need to re-run everything?

To check whether all slices are finished, simply look at the collected\_docs/tmp/counts\_\*.json files. All the files corresponding to completed slices should only have the string *finished*, so for example:
```
ELI5/data_creation/processed_data$ grep finished collected_docs/tmp/counts_* | wc
```
If the number of finished thread is less than 100, simply relaunch the missing ones, the code will restart them from their last saved space. To launch a single thread (e.g. 34), go to the data\_creation folder and run:
```
python download_support_docs.py -ns 34
```

## Finalizing the dataset

All that remains to do now is to map the collected passages to the question-answer pairs and, apply our provided heuristic to make a single support document to select relevant passages
```
cd slurm_scripts
./eli_select_docs_launcher.sh
```

And finally, make the train, valid and test split with:
```
cd ..
python finalize_qda.py
rm processed_data/selected_15_1/explainlikeimfive/selected_slice_*.json
```

Congrats, you can now start working on your very own Long-Form Question Answering systems!
```
ELI5/data_creation$ ls -lh processed_data/selected_15_1/explainlikeimfive*.json
-rw-rw-r-- 1 username username 142M Oct 15 06:18 processed_data/selected_15_1/explainlikeimfive_test.json
-rw-rw-r-- 1 username username 1.3G Oct 15 06:18 processed_data/selected_15_1/explainlikeimfive_train.json
-rw-rw-r-- 1 username username  57M Oct 15 06:18 processed_data/selected_15_1/explainlikeimfive_valid.json
```

# Modeling with Fairseq-py

## Formatting data files

We provide a script to convert the json formatted files to .txt files for source and target and build the multi-tasking source-target pairs. Modify the input and output path accordingly.
```
cd model_code
OUTPUT_PATH=formatted_files
PATH_TO_DATA=processed_data/selected_15_1
mkdir $OUTPUT_PATH
python process_data_to_source_target.py --input $PATH_TO_DATA --output $OUTPUT_PATH
```

## Applying BPE

We use [bpe](https://github.com/rsennrich/subword-nmt) and release the BPE codes we used. You will need to apply BPE on the data. We provide a sample command:

```
subword-nmt apply-bpe -c model_code/bpe_codes.txt < formatted_files/train.qd_source > formatted_files/train.qd_source_bpe
subword-nmt apply-bpe -c model_code/bpe_codes.txt < formatted_files/test.qd_source > formatted_files/test.qd_source_bpe
subword-nmt apply-bpe -c model_code/bpe_codes.txt < formatted_files/valid.qd_source > formatted_files/valid.qd_source_bpe
subword-nmt apply-bpe -c model_code/bpe_codes.txt < formatted_files/train.qd_target > formatted_files/train.qd_target_bpe
subword-nmt apply-bpe -c model_code/bpe_codes.txt < formatted_files/test.qd_target > formatted_files/test.qd_target_bpe
subword-nmt apply-bpe -c model_code/bpe_codes.txt < formatted_files/valid.qd_target > formatted_files/valid.qd_target_bpe
```

## Training and Generating with Fairseq-py

We use the [Fairseq-py](https://github.com/pytorch/fairseq) sequence-to-sequence library. For details about this library, please see the main repository or read the [full documentation](https://fairseq.readthedocs.io/en/latest/). We provide example commands below. Please modify the paths accordingly to where you have stored the data, where you would like the binarized data to be located, and where the trained model checkpoints have been stored.

To binarize the data:
```
cd fairseq
TEXT=formatted_files
python preprocess.py --source-lang qd_source_bpe --target-lang qd_target_bpe \
   --validpref $TEXT/valid --testpref $TEXT/test --trainpref $TEXT/train --destdir data-bin/eli5
```

If you would like to binarize the dataset with our dictionary instead of creating your own (for example, if you would like to load our pretrained model), you can add the following parameters to the above command:
```
--srcdict model_code/dict.multitask_source_bpe.txt --tgtdict model_code/dict.multitask_target_bpe.txt
```

To train the model:
```
cd fairseq
python train.py data-bin/eli5 --task translation --source-lang qd_source_bpe --target-lang qd_target_bpe --arch transformer_wmt_en_de_big_t2t --share-decoder-input-output-embed --dropout 1e-1 --attention-dropout 1e-1 --relu-dropout 1e-1 --criterion label_smoothed_cross_entropy --label-smoothing 1e-1 --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 --min-lr 1e-9 --clip-norm 0 --no-progress-bar --log-interval 100
```

Depending on the model you might need to increase the values for `--max-source-positions` and `--max-target-positions` (e.g., to 4096), and set `--max-tokens` and `--update-freq` to a suitable value.

To generate from the model:
```
cd fairseq
PATH_TO_CHECKPOINT=model_checkpoint.pt
python generate.py data-bin/eli5 --path $PATH_TO_CHECKPOINT --gen-subset valid --task translation --nbest 1 --source-lang qd_source_bpe --target-lang qd_target_bpe --beam 5 --batch-size 32 --remove-bpe --no-repeat-ngram-size 3 --max-len-b 500 --min-len 200 --max-source-positions 4096 --max-target-positions 4096 --skip-invalid-size-inputs-valid-test --model-overrides "{'max_source_positions':4096, 'max_target_positions':4096}"
```
to evaluate on the test set, set:
```
--gen-subset test
```

## Evaluating ROUGE
We provide a script used to compute ROUGE. Given a file of model generated hypotheses and a file of true references, it can be run in the following manner:
```
pip install rouge
HYPOTHESES=model_hypotheses.txt
REFERENCES=true_references.txt
python compute_rouge.py --hypotheses $HYPOTHESES --references $REFERENCES
```
The min and max length of generation were tuned. For partial fill ROUGE, we evaluated fixed length generation (as model generated answers are usually tuned to be a lot longer than human written answers) based on the validation set.

## How to use the Multi-task Pretrained Model
We provide a pretrained model, which you can download here:
```
wget https://dl.fbaipublicfiles.com/eli5qa/multitask_checkpoint.pt
```

To use the pretrained model, you will need to follow these steps:

First, the multi-task model requires labeling the source with the task label (e.g. question answering v. language modeling). We provide a script here to do this:
```
python model_code/prepare_multitask_input.py --input $DATA_FILE --output $OUTPUT_DATA_FILE
```

Then, apply the BPE:
```
subword-nmt apply-bpe -c model_code/bpe_codes.txt < $OUTPUT_DATA_FILE > $OUTPUT_DATA_FILE_BPE
```

Now, you are ready to forward the model on your BPE'd data. You can generate from the model using Fairseq-py ``generate.py`` or ``interactive.py`` commands.

## Issues running the modeling scripts?

Check out the file ``test_model_code_scripts.sh`` which runs all of the model scripts we include. The sample input/output of these scripts is included in the folder ``testing_files`` for your reference. If you are having trouble, please take a look at these sample files we used for testing to make sure you have the correct input format.


# Citation

Please cite as:
```bibtex
@inproceedings{fan2019eli5,
  title = {ELI5: Long Form Question Answering},
  author = {Angela Fan and Yacine Jernite and Ethan Perez and David Grangier and Jason Weston and Michael Auli},
  booktitle = {Proceedings of ACL 2019},
  year = {2019},
}
```
