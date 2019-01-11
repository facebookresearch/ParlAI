# TFIDF Retriever
 The *TFIDF Retriever* is an agent that constructs a [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
 matrix for all entries in a given task. It generates responses via
 returning the highest-scoring documents for a query. It uses a SQLite database
 for storing the sparse tfidf matrix, adapted from [here](http://github.com/facebookresearch/DrQA/).

 ## Basic Examples
 Construct a TFIDF matrix for use in retrieval for the personachat task
```bash
python examples/train_model.py -m tfidf_retriever -t personachat -mf /tmp/personachat_tfidf -dt train:ordered -eps 1
```
 After construction, load and evaluate that model on the Persona-Chat test set.
```bash
python examples/eval_model.py -t personachat -mf /tmp/personachat_tfidf -dt test
```

 Alternatively, interact with a Wikipedia-based TFIDF model from the model zoo
 ```bash
 python examples/interactive.py -mf models:wikipedia_full/tfidf_retriever/model
 ```
