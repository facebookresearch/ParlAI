# Hugging Face

We offer wrappers for generative transformers from [Hugging Face's transformers repository](https://github.com/huggingface/transformers) for fine-tuning and evaluating in ParlAI.

## GPT2
To use GPT2, run your command with the flag: `-m hugging_face/gpt2`.

### Examples
**Talk to GPT2 large in interactive mode, with beam size 10, 3-gram beam blocking, and minimum beam length 25:**
```bash
python examples/interactive.py -m hugging_face/gpt2 --add-special-tokens False --gpt2-size large --inference beam --beam-size 10 --beam-context-block-ngram 3 --beam-block-ngram 3 --beam-min-length 25
```
_Note:_ In the above command, we must have the flag `--add-special-tokens False` if we want to use the model _without_ finetuning it.

Here is example output from the above command:
```
Enter Your Message: Parrots are
[Gpt2]:  one of the most popular pets in the world. They can be trained to do a variety of tasks, such as fetching objects, opening doors, climbing ladders, and more. They are also very intelligent and can learn new skills very quickly.
```


**Fine-tune GPT2 medium on the ConvAI2 task:**
```bash
python examples/train_model.py -m hugging_face/gpt2 --add-special-tokens True --add-start-token True --gpt2-size medium -t convai2 -bs 2 -mf <modelfile>
```

## Other models
_Other models are coming soon -- stay tuned!_