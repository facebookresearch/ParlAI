## Pretrained Model

Running any of the following scripts will automatically download the pretrained model.

Run the `eval_f1.py` script to evaluate the f1 score of the pretrained model on the twitter dev set. (Run `python eval_f1.py -dt test` to run on the test set.)

The `eval_ppl.py` to run a standardized perplexity evaluation, which asks for one token at a time from the model.

Run the `interactive.py` script to talk to the pretrained model directly.

## Train your own

Run `python train.py` to train your own using the same hyperparameters as the pretrained model.

You can override any of these from the command line, e.g. `python train.py -bs 16 -lr 0.0001 -opt adam`.

## Sweep results

In case you're interested, the results of an example sweep over parameters is contained in `sweep_ppl.csv`.

The first column in that file contains the best validation perplexity found for each of those parameters.
