# Retrieval-Augmented Generation (RAG)

The code in this directory implements the [RAG Model](https://arxiv.org/abs/2005.11401) as used in the [reducing hallucination](https://parl.ai/projects/hallucination/) project. The README is broken up into the following sections:

1. Installation instructions
1. Quick-start tutorial to using RAG.
2. In-depth discussion of RAG Options
2. Tutorial for generating your own embeddings / build your own index.
3. Directory structure/overview.

If you have any questions, please reach out to @klshuster or @spencerp.

## Installation / Memory Requirements.

Before using RAG, you'll need to make sure that you have installed FAISS; preferably, you should install the `faiss-gpu` library (installation instructions [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)), but RAG will work with `faiss-cpu` as well (`faiss-gpu` will simply speed up index construction).

To train a RAG model with the default options -- RAG-Token with BART-Large generator, and DPR Retrieval over all of Wikipedia -- you'll need the following system requirements:

### RAM

Loading the Wikipedia passages into memory requires ~22GB of RAM.

If you use `--indexer-type compressed --path-to-index zoo:hallucination/wiki_passages_compressed/compressed_pq`, you'll only require an additional ~3GB of RAM; if you use `--indexer-type exact --path-to-index zoo:hallucination/wiki_passages_exact/exact`, you'll need an additional ~80GB of RAM.

### GPU

To train BART-Large RAG / FiD models, with a batchsize of 16 (or DPR-Poly models with a batchsize of 8), you'll want to have at least 4x32gb GPUs. You can adjust the batchsize to fit your GPU memory constraints.

To evaluate / interact with any pre-trained models (e.g., those [mentioned here](https://parl.ai/projects/hallucination/)), you'll only need 1 16gb GPU.


## RAG Quick Start
You can use RAG like any other model in ParlAI; simply specify `-m rag`, and you're good to go! Here's an example command to train RAG on the [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/) Dataset:

```python
parlai train_model -m rag -t wizard_of_wikipedia -mf /path/to/model_file \
# standard optimization/truncation parameters
--batchsize 16 --fp16 True --gradient-clip 0.1 --label-truncate 128 \
--log-every-n-secs 30 --lr-scheduler reduceonplateau --lr-scheduler-patience 1 \
--model-parallel True --optimizer adam --text-truncate 512 --truncate 512 \
-lr 1e-05 -vmm min -veps 0.25 -vme 1000 -vmt ppl -vp 5 \
# BART-Large parameters
-o arch/bart_large
```


## RAG Options

RAG in ParlAI is quite flexible, and can support a variety of different base seq2seq models, retrievers, and "model types"; we outline the different options below. **Bolded options** are the default options.

### RAG Seq2Seq Generators: `--generation-model`

We support three backbones:

1. **`--generation-model bart`**: The default option uses BART as the backbone generator, which was used in the vast majority of experiments in [this paper](https://arxiv.org/abs/2104.07567).
2. `--generation-model transformer/generator`: If you want to use/initialize RAG with a standard Transformer model trained in ParlAI, set to `transformer/generator`
3. `--generation-model t5`: Finally, we provide T5 as a generator backbone as well; see [here](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents/hugging_face#t5) for additional T5-specific parameters.

### RAG Model Types: `--rag-model-type`

RAG comes in three flavors: RAG-Token, RAG-Sequence, and RAG-Turn. The first two are outlined in the original [RAG paper](https://arxiv.org/abs/2005.11401); the third is outlined in our [retrieval-augmented dialogue work](https://arxiv.org/abs/2104.07567).

1. **`--rag-model-type token`**: The RAG-Token model jointly attends to all documents, allowing each token to draw from a latent document.
2. `--rag-model-type sequence`: The RAG-Sequence model attends to each retrieved document separately, re-ranking generations according to document probabilities.
3. `--rag-model-type turn`: The RAG-Turn model retrieves documents for each _turn_ of dialogue context, and either attends jointly over all turns and documents (`--rag-turn-marginalize doc_then_turn`) or over each turn separately (`--rag-turn-marginalize doc_only`).

### RAG Retriever Types: `--rag-retriever-type`

We provide a few of the several retrievers considered in [our work](https://parl.ai/projects/hallucination/); we outline them below:

1. **`--rag-retriever-type dpr`**: The canonical retrieval system for RAG uses a [Dense Passage Retriever](https://github.com/facebookresearch/DPR) for retrieval over a FAISS Index. The default options retrieve over all of Wikipedia.
2. `--rag-retriever-type tfidf`: One can additionally use a [TFIDF retriever](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents/tfidf_retriever); the default retrieves over all of Wikipedia.
3. `--rag-retriever-type dpr_then_poly`: The RAG DPR-Poly model adds a re-ranking step with a Poly-encoder that re-ranks the retrieved documents from a DPR model.
4. `--rag-retriever-type poly_faiss`: If you have trained a [Dropout Poly-encoder](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents/transformer/dropout_poly) and have built an index with that model, you can use the PolyFAISS method, which uses a Poly-encoder model directly to both query FAISS and re-rank retrieved documents.

### Other RAG Options

All of the options for using RAG can be found in the `args.py` file; below, we highlight a few that are important.

#### Number of Retrieved Documents

Set the `--n-docs` flag to tell RAG how many documents to retrieve.

#### Thorough Decoding

For RAG-Sequence, and RAG-Turn Doc-Only, you can specify `--thorough True` to use **thorough** decoding; this method will rescore hypotheses by running an additional forward pass of the model.

#### FAISS Indexes

We provide two indexes in our model zoo, which can be specified via the `--path-to-index` flag:

1. `--path-to-index zoo:hallucination/wiki_passages/exact --indexer-type exact`: The "exact" representation of the document embeddings in a FAISS Index. This index is over 80gb of RAM but provides the fastest/most accurate results.
2. **`--path-to-index zoo:hallucination/wiki_passages/compressed --indexer-type compressed`**: The "compressed" representation of the document embeddings a FAISS Index. This index is only ~3gb of RAM but comes at the price of performance degradation. This is the default option as it works quite well despite the compression.

## Generating your own FAISS Index.

The default RAG parameters use the `zoo:hallucination/wiki_passages/psgs_w100.tsv` corpus, which is 21m passages spanning all of Wikipedia. You can also generate your own FAISS index by following the steps below:

### 1a. [**Recommended**] Obtain/Choose a (Pre-trained) DPR Model

The RAG model works **really well** with DPR models as the backbone retrievers; check out the [DPR repository](https://github.com/facebookresearch/DPR) for some pre-trained DPR models (or, train your own!). Alternatively, you can specify a RAG or FiD model with DPR weights (perhaps, e.g., one from the ParlAI model zoo, such as `zoo:hallucination/bart_rag_token/model`).

### 1b. Train your own Dropout Poly-encoder

To utilize the PolyFAISS method, you can train your own [`DropoutPolyencoder`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/agents/transformer/dropout_poly.py)) as usual in ParlAI.

### 2. Generate Dense Embeddings (~1-2 hours minutes if sharded appropriately - 50 x 1 GPU).

**WARNING**: If you generated passage embeddings prior to 11/19/2021, you *may* have corrupted embeddings, especially if you were using a relatively small set of passages (anything under ~50k), and found that indexing took excessively long (anything over a couple minutes); see [#4199](https://github.com/facebookresearch/ParlAI/pull/4199) for more details.

After obtaining a DPR model, you'll need to generate dense embeddings on a dataset. The data should be in a tab-separated (tsv) file with the following format:

      integer document id starting at zero<tab>document text<tab>document title

Check `/path/to/ParlAI/data/models/hallucination/wiki_passages/psgs_w100.tsv` for inspiration.

Then, you can use the [`generate_dense_embeddings.py`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/agents/rag/scripts/generate_dense_embeddings.py) script to run the following command:

```bash
python generate_dense_embeddings.py --model-file /path/to/dpr/model --dpr-model True \
--passages-file /path/to/passages --outfile /path/to/saved/embeddings \
--shard-id <shard_id> --num-shards <num_shards> -bs <batchsize>
```

If the provided `--model-file` is either a path to a DPR model or a path to a ParlAI RAG/FiD model, specify `--dpr-model True` so that the script can appropriately extract the DPR weights; if you use a Dropout Poly-encoder, set `--dpr-model` to `False`. The script will generate embeddings with the DPR model for shard `<shard_id>` of the data, and save two files:

- `/path/to/saved/embeddings_<shard_id>`: The concatenated tensor of embeddings
- `/path/to/saved/ids_<shard_id>`: The list of document ids that corresponds to these embeddings.

An example command would look like this:

```bash
python generate_dense_embeddings.py -mf zoo:hallucination/multiset_dpr/hf_bert_base.cp --dpr-model True \
--passages-file zoo:hallucination/wiki_passages/psgs_w100.tsv  \
--outfile /tmp/wiki_passage_embeddings/wiki_passages --num-shards 50 --shard-id 0 -bs 32
```

**`--num-shards`**: If your dataset is relatively small, you can feel free to only generate with only one shard.

### 3. Index the Dense Embeddings

The final step is to build the full FAISS index from these dense embeddings. You can use the [`index_dense_embeddings.py`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/agents/rag/scripts/index_dense_embeddings.py) script to achieve this. You can choose one of the following options when indexing your embeddings for varying results, depending on the size of your dataset:

1. **Recommended for large passage sets** `--indexer-type compressed`: This will build a compressed index using FAISS compression techniques; this usually only takes a couple hours, and results in small index files, but comes at the cost of accuracy. Only use this if your machine would struggle to fit all of your dense embedding vectors in memory.
2. **Recommended for small passage sets** `--indexer-type exact`: This will build a large HNSW-style index with the flat embeddings. The index that is built is generally as large, if not more so, than the sum of the sizes of the embeddings. Use with caution with large passage sets; however, if you can reasonably fit all of your dense embedding vectors in memory, this is a suitable option.
3. `--indexer-type compressed --compressed-indexer-factory <index_factory>`: If you know what you're doing (and understand how to use the [index factory in FAISS](https://github.com/facebookresearch/faiss/wiki/The-index-factory)), feel free to specify your own Index Factory settings. This method is only recommended if you're an advanced FAISS user.

If we saved our embedding shards at `/path/to/saved/embeddings_0`, the script is used as follows:

```bash
python index_dense_embeddings.py --retriever-embedding-size <retriever_emb_size>  \
--embeddings-dir /path/to/saved/ --embeddings-name <embeddings> --indexer-type <indexer_type>
```

Example:

```bash
python index_dense_embeddings.py --retriever-embedding-size 768  \
--embeddings-dir /tmp/wiki_passage_embeddings/ --embeddings-name wiki_passages
```

Note the default index factory setting is `IVF4096_HNSW128,PQ128`, if you are processing small files, you may encounter errors such as `Error: 'nx >= k' failed`, then you need to set `--compressed-indexer-factory` to other indexes in the [index factory in FAISS](https://github.com/facebookresearch/faiss/wiki/The-index-factory) such as `HNSW32`.

## Directory Structure / Custom Components

I will outline here the structure of the RAG directory, and where you might want to add custom components if you so desire.

- `args.py`: Contains the parameters used to train RAG Models. Explore at your leisure
- `conversion_utils.py`: Utility functions for converting DPR models to ParlAI-style models
- `dpr_agent.py`: A wrapper around DPR Models for use in ParlAI.
- `indexers.py`: Contains implementations of "Indexers", which are essentially wrappers for interacting with FAISS Indexes.
- `model_types.py`: Contains the interfaces for RAG-Token, RAG-Sequence, and RAG-Turn. The interfaces define the model-type-specific functionality for each RAG type.
- `modules.py`: Contains the actual `RagModel` implementation. The components of a `RagModel` are model-type-agnostic, and thus they are separate from the implementations in `model_types.py`
- `rag.py`: Contains the `RagAgent` implementation.
- `retrievers.py`: Contains retrievers used in the `RagModel`

### Custom Components

#### Sequence to Sequence Models

The `RagModel` tries to be as generic as possible with the underlying seq2seq architecture; to fit future generator models, one can look to the `T5RagModel` in `modules.py` as inspiration for what a custom model may look like.

#### Retriever Models

The RAG Retriever models are generic as well, and simply require that a `retrieve` function is defined. The base `RagRetriever` defines the interface for the retriever, so as long as a subclass implements the necessary functions, adding new retrievers is a straightforward exercise.
