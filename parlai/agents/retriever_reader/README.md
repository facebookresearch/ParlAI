# Retriever Reader
This Retriever Reader is an agent which first retrieves from a database and then reads the dialogue + knowledge
from the database to answer.

NOTE: this model only works for evaluation; it assumes all training is already done.

## Basic Examples
Evaluate a pre-trained model on Open SQuAD. This model uses a TF-IDF retriever and the DrQA model as the document reader.
```bash
python projects/drqa/eval_opensquad.py
```
