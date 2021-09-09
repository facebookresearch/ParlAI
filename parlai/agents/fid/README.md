# Fusion in Decoder (FiD)

The FiD model is first described in [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282) (G. Izacard, E. Grave 2020); the original implementation can be found [here](https://github.com/facebookresearch/FiD). The implementation we provide uses the RAG models as a backbone; thus, instructions for options to use when running a FiD model can be found in the [RAG README](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents/rag#readme), as well as the corresponding [project page](https://parl.ai/projects/hallucination/).

Simply swap `--model rag` with `--model fid`, and you're good to go!
