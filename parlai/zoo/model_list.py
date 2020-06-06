#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The Model Zoo.

This file contains a list of all the models in the model zoo, the path to
load them, agents & tasks associated (e.g. they were trained using) and a
description. Using the path you should be able to download and use the model
automatically, e.g.:

... code-block:

   python examples/interactive.py --model-file
       "zoo:wikipedia_20161221/tfidf_retriever/drqa_docs"


There are a number of guidelines you should follow in the zoo:

- You should choose the best directory name as possible. An input of
  ``zoo:PROJECTNAME/MODELNAME/FILENAME`` will attempt to use a build script from
  parlai/zoo/PROJECTNAME/MODELNAME.py.
- You should include all of the following fields:
    * title: the name of the entry:
    * id: corresponds to PROJECTNAME
    * description: describe the entry in reasonable detail. It should be at least
      a couple sentences.
    * example: an example command to chat with or evaluate the model
    * result: the expected output from running the model. You are strongly encouraged
      to make a nightly test which verifies this result.
    * external_website: if applicable, an external website related to the zoo to
      link to.
    * project: if applicable, a link to the project folder. You must have either
      external_website or project.
    * example2 and result2 (optional): additional examples to run.

- As much as possible, you should try to include two examples: one to generate
  some key metrics (e.g. from a paper) and one to actually chat with the model
  using interactive.py. Both should strongly attempt to minimize mandatory
  command line flags.
"""

model_list = [
    {
        "title": "KVMemNN ConvAI2 model",
        "id": "convai2",
        "path": "zoo:convai2/kvmemnn/model",
        "agent": "projects.personachat.kvmemnn.kvmemnn:Kvmemnn",
        "task": "convai2",
        "description": (
            "KvMemNN trained on the ConvAI2 task, used as a baseline in the "
            "competition."
        ),
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2",
        "external_website": "http://convai.io/",
        "example": (
            "python -m parlai.scripts.interactive -mf zoo:convai2/kvmemnn/model"
        ),
        "result": (
            "Enter Your Message: Hi, what do you think of peanuts?\n"
            "there was a kid in the school system my mum works for with a severe peanut allergy"
        ),
    },
    {
        "title": "Seq2Seq ConvAI2 model",
        "id": "convai2",
        "path": "zoo:convai2/seq2seq/convai2_self_seq2seq_model",
        "agent": "legacy:seq2seq:0",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2",
        "external_website": "http://convai.io/",
        "description": (
            "SeqSeq trained on the ConvAI2 task, used as a baseline in the competition."
        ),
        "example": (
            "python -m parlai.scripts.interactive -mf "
            "zoo:convai2/seq2seq/convai2_self_seq2seq_model -m legacy:seq2seq:0"
        ),
        "result": (
            "Enter Your Message: Hi, what do you think of peanuts?\n"
            "[Seq2Seq]: i don't have any , but i do not have a favorite ."
        ),
    },
    {
        "title": "ConvAI2 Language model",
        "id": "convai2",
        "path": "zoo:convai2/language_model/model",
        "agent": "language_model",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2",
        "external_website": "http://convai.io/",
        "description": (
            "SeqSeq trained on the ConvAI2 task, used as a baseline in the competition."
        ),
        "example": (
            "python -m parlai.scripts.interactive -mf "
            "zoo:convai2/language_model/model"
        ),
        "result": (
            "Enter Your Message: Hi, have you ever been on an airplane?\n"
            "[LanguageModel]: no , i do not . i am a big fan of the walking dead ."
        ),
    },
    {
        "title": "DrQA SQuAD model",
        "id": "drqa",
        "path": "zoo:drqa/squad/model",
        "agent": "drqa",
        "task": "squad",
        "description": "DrQA Reader trained on SQuAD",
        "external_website": "https://github.com/facebookresearch/DrQA",
        "example": (
            "python -m parlai.scripts.eval_model -mf zoo:drqa/squad/model -t squad "
            "-dt test"
        ),
        "result": (
            # TODO: this differs slightly from the actual results as of 2019-07-23
            "{'exs': 10570, 'accuracy': 0.6886, 'f1': 0.7821, 'hits@1': 0.689, 'hits@5': 0.689, 'hits@10': 0.689, 'hits@100': 0.689, 'bleu': 0.1364, 'train_loss': 0}"  # noqa: E501
        ),
    },
    {
        "title": "Wikipedia Retriever (used for open SQuAD)",
        "id": "wikipedia_20161221",
        "path": "zoo:wikipedia_20161221/tfidf_retriever/drqa_docs",
        "agent": "tfidf_retriever",
        "external_website": "https://github.com/facebookresearch/DrQA",
        "task": "wikipedia:full",
        "example": (
            "python -m parlai.scripts.interactive --model tfidf_retriever "
            "-mf zoo:wikipedia_20161221/tfidf_retriever/drqa_docs"
        ),
        "result": (
            """
            Enter Your Message: Yann LeCun
            [candidate_scores]: [507.05804682 390.18244433 279.24033928 269.60377042 214.00140589]
            [SparseTfidfRetrieverAgent]:
            Deep learning (also known as deep structured learning, hierarchical learning or deep machine learning) is a branch of machine learning based on a set of algorithms that attempt to model high level abstractions in data. In a simple case, you could have two sets of neurons: ones that receive an input signal and ones that send an output signal. When the input layer receives an input it passes on a modified version of the input to the next layer. In a deep network, there are many layers between the input and output (and the layers are not made of neurons but it can help to think of it that way), allowing the algorithm to use multiple processing layers, composed of multiple linear and non-linear transformations.

            Deep learning is part of a broader family of machine learning methods based on ...
            to commonsense reasoning which operates on concepts in terms of production rules of the grammar, and is a basic goal of both human language acquisition and AI. (See also Grammar induction.)
            """  # noqa: E501
        ),
        "description": (
            "Retrieval over Wikipedia dump, used for DrQA on the open squad "
            "dataset. This is the dump from the original paper, used for "
            "replicating results."
        ),
    },
    {
        "title": "Wikipedia Retriever (used for Wizard of Wikipedia)",
        "id": "wikipedia_full",
        "path": "zoo:wikipedia_full/tfidf_retriever/model",
        "agent": "tfidf_retriever",
        "task": "wikipedia:full",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia",
        "description": (
            "Retrieval over Wikipedia dump, used for DrQA on the open squad " "dataset."
        ),
        "example": (
            "python -m parlai.scripts.interactive --model tfidf_retriever -mf "
            "zoo:wikipedia_full/tfidf_retriever/model"
        ),
        "result": (
            """
            Enter Your Message: Yann LeCun
            [candidate_scores]: [454.74038503 353.88863708 307.31353203 280.4501096  269.89960432]
            [SparseTfidfRetrieverAgent]:
            Yann LeCun (; born 1960) is a computer scientist with contributions in machine learning, computer vision, mobile robotics and computational neuroscience. He is well known for his work on optical character recognition and computer vision using convolutional neural networks (CNN), and is a founding father of convolutional nets. He is also one of the main creators of the DjVu image compression technology (together with Léon Bottou and Patrick Haffner). He co-developed the Lush programming language with Léon Bottou.

            Yann LeCun was born near Paris, France, in 1960. He received a Diplôme d'Ingénieur from the Ecole Superieure d'Ingénieur en Electrotechnique et Electronique (ESIEE), Paris in 1983, and a PhD in Computer Science from Université Pierre et Marie Curie in 1987 during which he ...
            of Science and Technology in Saudi Arabia because he was considered a terrorist in the country in view of his atheism.

            In 2018 Yann LeCun picked a fight with a robot to support Facebook AI goals.
            """  # noqa: E501
        ),
    },
    {
        "title": "Wizard of Wikipedia (End to end Generator)",
        "id": "wizard_of_wikipedia",
        "path": "zoo:wizard_of_wikipedia/end2end_generator/model",
        "description": ("End2End Generative model for Wizard of Wikipedia"),
        "task": "wizard_of_wikipedia:generator",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia",
        "example": (
            "python examples/display_model.py -t wizard_of_wikipedia:generator "
            "-mf zoo:wizard_of_wikipedia/end2end_generator/model -n 1 "
            "--display-ignore-fields knowledge_parsed"
        ),
        "result": (
            """
            [chosen_topic]: Gardening
            [knowledge]: no_passages_used __knowledge__ no_passages_used
            Gardening __knowledge__ Gardening is the practice of growing and cultivating plants as part of horticulture.
            Gardening __knowledge__ In gardens, ornamental plants are often grown for their flowers, foliage, or overall appearance; useful plants, such as root vegetables, leaf vegetables, fruits, and herbs, are grown for consumption, for use as dyes, or for medicinal or cosmetic use.
            Gardening __knowledge__ Gardening is considered by many people to be a relaxing activity.
            Gardening __knowledge__ Gardening ranges in scale from fruit orchards, to long boulevard plantings with one or more different types of shrubs, trees, and herbaceous plants, to residential yards including lawns and foundation plantings, to plants in large or small containers ...
            there had been several other notable gardening magazines in circulation, including the "Gardeners' Chronicle" and "Gardens Illustrated", but these were tailored more for the professional gardener.

            [title]: Gardening
            [checked_sentence]: Gardening is considered by many people to be a relaxing activity.
            [eval_labels_choice]: I live on a farm, we garden all year long, it is very relaxing.
            [checked_sentence_parsed]: Gardening __knowledge__ Gardening is considered by many people to be a relaxing activity.
            [WizTeacher]: Gardening
            I like Gardening, even when I've only been doing it for a short time.
            [eval_labels: I live on a farm, we garden all year long, it is very relaxing.]
            [TorchAgent]: i love gardening , it is considered a relaxing activity .
            """  # noqa: E501
        ),
    },
    {
        "title": "Wizard of Wikipedia (Full Dialogue Retrieval Model)",
        "id": "wizard_of_wikipedia",
        "path": "zoo:wizard_of_wikipedia/full_dialogue_retrieval_model/model",
        "description": ("Full Dialogue Retrieval Model for Wizard of Wikipedia"),
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia",
        "task": "wizard_of_wikipedia",
        "example2": "examples/interactive.py -m projects:wizard_of_wikipedia:interactive_retrieval -t wizard_of_wikipedia",
        "result2": (
            "[ Your chosen topic is: Teapot ]\n"
            "Enter Your Message: do you like tea?\n"
            "[WizardRetrievalInteractiveAgent]: Yes!  I only use teapots that have a little air hole in the lid. That prevents the spout from dripping or splashing when the tea is poured. Most teapots have this though.\n"
            "Enter Your Message: what about kettles?\n"
            "[WizardRetrievalInteractiveAgent]: I would think you could use them to heat any type of liquid! I use my teapots with a tea cosy. It's a thermal cover that helps keep the tea hot.\n"
            "Enter Your Message: do you like earl grey?\n"
            "[WizardRetrievalInteractiveAgent]: I think I'll try some Lipton, I love their green tea!"
        ),
        "example": (
            "python examples/display_model.py -t wizard_of_wikipedia "
            "-mf zoo:wizard_of_wikipedia/full_dialogue_retrieval_model/model "
            "-m projects:wizard_of_wikipedia:wizard_transformer_ranker "
            "--n-heads 6 --ffn-size 1200 --embeddings-scale False "
            "--delimiter ' __SOC__ ' --n-positions 1000 --legacy True "
        ),
        "result": (
            """
            [chosen_topic]: Gardening
            [knowledge]: Gardening Gardening is the practice of growing and cultivating plants as part of horticulture.
            Gardening In gardens, ornamental plants are often grown for their flowers, foliage, or overall appearance; useful plants, such as root vegetables, leaf vegetables, fruits, and herbs, are grown for consumption, for use as dyes, or for medicinal or cosmetic use.
            Gardening Gardening is considered by many people to be a relaxing activity.
            Gardening Gardening ranges in scale from fruit orchards, to long boulevard plantings with one or more different types of shrubs, trees, and herbaceous plants, to residential yards including lawns and foundation plantings, to plants in large or small containers grown inside or outside.
            Gardening Gardening may be very specialized, with only one type of plant grown, ...
            there had been several other notable gardening magazines in circulation, including the "Gardeners' Chronicle" and "Gardens Illustrated", but these were tailored more for the professional gardener.

            [title]: Gardening
            [checked_sentence]: Gardening is considered by many people to be a relaxing activity.
            [eval_labels_choice]: I live on a farm, we garden all year long, it is very relaxing.
            [wizard_of_wikipedia]: Gardening
            I like Gardening, even when I've only been doing it for a short time.
            [label_candidates: OK what's the history?|Right, thats cool. I had no idea they still did the DVD thing, What is Netflix's highest rated show? do you know? |I will definitely check his first album out as he sounds interesting.|I don't know a whole lot about it. I was raised Catholic but don't practice anything now.|Well , this was a good conversation. |...and 95 more]
            [eval_labels: I live on a farm, we garden all year long, it is very relaxing.]
               [TorchAgent]: I live on a farm, we garden all year long, it is very relaxing.
            """  # noqa: E501
        ),
    },
    {
        "title": "LIGHT BERT-Biranker Dialogue model",
        "id": "light",
        "path": "zoo:light/biranker_dialogue/model",
        "agent": "bert_ranker/bi_encoder_ranker",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/light",
        "task": "light_dialog",
        "description": ("LIGHT Dialogue task, replicating the numbers from the paper."),
        "example": (
            "python examples/eval_model.py -t light_dialog "
            "-mf zoo:light/biranker_dialogue/model"
        ),
        "result": "{'exs': 6623, 'accuracy': 0.7586, 'f1': 0.7802, 'hits@1': 0.759, 'hits@5': 0.965,"  # noqa: E501
        "'hits@10': 0.994, 'hits@100': 1.0, 'bleu': 0.7255, 'lr': 5e-05, 'total_train_updates': 15050,"  # noqa: E501
        "'examples': 6623, 'loss': 5307.0, 'mean_loss': 0.8013, 'mean_rank': 1.599, 'train_accuracy': 0}",  # noqa: E501
    },
    {
        "title": "Controllable Dialogue ConvAI2 model",
        "id": "controllable_dialogue",
        "path": "zoo:controllable_dialogue/convai2_finetuned_baseline",
        "agent": "projects.controllable_dialogue.controllable_seq2seq.controllable_seq2seq:ControllableSeq2seqAgent",  # noqa: E501
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/controllable_dialogue",
        "example": (
            "python -m parlai.scripts.eval_model --model "
            "projects.controllable_dialogue.controllable_seq2seq.controllable_seq2seq:"
            "ControllableSeq2seqAgent --task "
            "projects.controllable_dialogue.tasks.agents "
            "-mf zoo:controllable_dialogue/convai2_finetuned_baseline"
        ),
        "result": (
            "{'exs': 7801, 'accuracy': 0.0006409, 'f1': 0.1702, 'bleu': 0.005205, "
            "'token_acc': 0.3949, 'loss': 3.129, 'ppl': 22.86}"
        ),
        "description": ("Seq2Seq model with control trained on ConvAI2"),
    },
    {
        "title": "TransResNet (ResNet 152) Personality-Captions model",
        "id": "personality_captions",
        "path": "zoo:personality_captions/transresnet",
        "agent": "projects.personality_captions.transresnet.transresnet:TransresnetAgent",  # noqa: E501
        "task": "personality_captions",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/personality_captions",
        "description": (
            "Transresnet Model pretrained on the Personality-Captions task"
        ),
        "example": (
            "python examples/eval_model.py -t personality_captions "
            "-mf zoo:personality_captions/transresnet/model --num-test-labels 5 -dt test"
        ),
        "result": (
            "{'exs': 10000, 'accuracy': 0.5113, 'f1': 0.5951, 'hits@1': 0.511, "
            "'hits@5': 0.816, 'hits@10': 0.903, 'hits@100': 0.998, 'bleu': 0.4999, "
            "'hits@1/100': 1.0, 'loss': -0.002, 'med_rank': 1.0}"
        ),
    },
    {
        "title": "Poly-Encoder Transformer Reddit Pretrained Model",
        "id": "pretrained_transformers",
        "path": "zoo:pretrained_transformers/poly_model_huge_reddit",
        "agent": "transformer/polyencoder",
        "task": "pretrained_transformers",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/",
        "description": (
            "Poly-Encoder pretrained on Reddit. Use this model as an ``--init-model`` for a poly-encoder "
            "when fine-tuning on another task. For more details on how to train, see the project page."
        ),
        "example": (
            "python -u examples/train_model.py "
            "--init-model zoo:pretrained_transformers/poly_model_huge_reddit/model "
            "-t convai2 "
            "--model transformer/polyencoder --batchsize 256 --eval-batchsize 10 "
            "--warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 "
            "-lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 "
            "--text-truncate 360 --num-epochs 8.0 --max_train_time 200000 -veps 0.5 "
            "-vme 8000 --validation-metric accuracy --validation-metric-mode max "
            "--save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 True "
            "--dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 "
            "--variant xlm --reduction-type mean --share-encoders False "
            "--learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 "
            "--attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 "
            "--embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 "
            "--learn-embeddings True --polyencoder-type codes --poly-n-codes 64 "
            "--poly-attention-type basic --dict-endtoken __start__ "
            "--model-file <YOUR MODEL FILE>"
        ),
        "result": (
            "(subject to some variance, you may see the following as a result of validation of the model)\n"
            "{'exs': 7801, 'accuracy': 0.8942 ...}"
        ),
    },
    {
        "title": "Poly-Encoder Transformer Wikipedia/Toronto Books Pretrained Model",
        "id": "pretrained_transformers",
        "path": "zoo:pretrained_transformers/poly_model_huge_wikito",
        "agent": "transformer/polyencoder",
        "task": "pretrained_transformers",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/",
        "description": (
            "Poly-Encoder pretrained on Wikipedia/Toronto Books. Use this model as an ``--init-model`` for a poly-encoder "
            "when fine-tuning on another task. For more details on how to train, see the project page."
        ),
        "example": (
            "python -u examples/train_model.py "
            "--init-model zoo:pretrained_transformers/poly_model_huge_wikito/model "
            "-t convai2 "
            "--model transformer/polyencoder --batchsize 256 --eval-batchsize 10 "
            "--warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 "
            "-lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 "
            "--text-truncate 360 --num-epochs 8.0 --max_train_time 200000 -veps 0.5 "
            "-vme 8000 --validation-metric accuracy --validation-metric-mode max "
            "--save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 True "
            "--dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 "
            "--variant xlm --reduction-type mean --share-encoders False "
            "--learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 "
            "--attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 "
            "--embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 "
            "--learn-embeddings True --polyencoder-type codes --poly-n-codes 64 "
            "--poly-attention-type basic --dict-endtoken __start__ "
            "--model-file <YOUR MODEL FILE>"
        ),
        "result": (
            "(subject to some variance, you may see the following as a result of validation of the model)\n"
            "{'exs': 7801, 'accuracy': 0.861 ...}"
        ),
    },
    {
        "title": "Bi-Encoder Transformer Reddit Pretrained Model",
        "id": "pretrained_transformers",
        "path": "zoo:pretrained_transformers/poly_model_huge_reddit",
        "agent": "transformer/biencoder",
        "task": "pretrained_transformers",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/",
        "description": (
            "Bi-Encoder pretrained on Reddit. Use this model as an ``--init-model`` for a bi-encoder "
            "when fine-tuning on another task. For more details on how to train, see the project page."
        ),
        "example": (
            "python -u examples/train_model.py "
            "--init-model zoo:pretrained_transformers/bi_model_huge_reddit/model "
            "--batchsize 512 -t convai2 "
            "--model transformer/biencoder --eval-batchsize 6 "
            "--warmup_updates 100 --lr-scheduler-patience 0 "
            "--lr-scheduler-decay 0.4 -lr 5e-05 --data-parallel True "
            "--history-size 20 --label-truncate 72 --text-truncate 360 "
            "--num-epochs 10.0 --max_train_time 200000 -veps 0.5 -vme 8000 "
            "--validation-metric accuracy --validation-metric-mode max "
            "--save-after-valid True --log_every_n_secs 20 --candidates batch "
            "--dict-tokenizer bpe --dict-lower True --optimizer adamax "
            "--output-scaling 0.06 "
            "--variant xlm --reduction-type mean --share-encoders False "
            "--learn-positional-embeddings True --n-layers 12 --n-heads 12 "
            "--ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 "
            "--n-positions 1024 --embedding-size 768 --activation gelu "
            "--embeddings-scale False --n-segments 2 --learn-embeddings True "
            "--share-word-embeddings False --dict-endtoken __start__ --fp16 True "
            "--model-file <YOUR MODEL FILE>"
        ),
        "result": (
            "(subject to some variance, you may see the following as a result of validation of the model)\n"
            "{'exs': 7801, 'accuracy': 0.8686 ...}"
        ),
    },
    {
        "title": "Bi-Encoder Transformer Wikipedia/Toronto Books Pretrained Model",
        "id": "pretrained_transformers",
        "path": "zoo:pretrained_transformers/bi_model_huge_wikito",
        "agent": "transformer/biencoder",
        "task": "pretrained_transformers",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/",
        "description": (
            "Bi-Encoder pretrained on Wikipedia/Toronto Books. Use this model as an ``--init-model`` for a poly-encoder "
            "when fine-tuning on another task. For more details on how to train, see the project page."
        ),
        "example": (
            "python -u examples/train_model.py "
            "--init-model zoo:pretrained_transformers/bi_model_huge_wikito/model "
            "--batchsize 512 -t convai2 "
            "--model transformer/biencoder --eval-batchsize 6 "
            "--warmup_updates 100 --lr-scheduler-patience 0 "
            "--lr-scheduler-decay 0.4 -lr 5e-05 --data-parallel True "
            "--history-size 20 --label-truncate 72 --text-truncate 360 "
            "--num-epochs 10.0 --max_train_time 200000 -veps 0.5 -vme 8000 "
            "--validation-metric accuracy --validation-metric-mode max "
            "--save-after-valid True --log_every_n_secs 20 --candidates batch "
            "--dict-tokenizer bpe --dict-lower True --optimizer adamax "
            "--output-scaling 0.06 "
            "--variant xlm --reduction-type mean --share-encoders False "
            "--learn-positional-embeddings True --n-layers 12 --n-heads 12 "
            "--ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 "
            "--n-positions 1024 --embedding-size 768 --activation gelu "
            "--embeddings-scale False --n-segments 2 --learn-embeddings True "
            "--share-word-embeddings False --dict-endtoken __start__ --fp16 True "
            "--model-file <YOUR MODEL FILE>"
        ),
        "result": (
            "(subject to some variance, you may see the following as a result of validation of the model)\n"
            "{'exs': 7801, 'accuracy': 0.846 ...}"
        ),
    },
    {
        "title": "Cross-Encoder Transformer Reddit Pretrained Model",
        "id": "pretrained_transformers",
        "path": "zoo:pretrained_transformers/cross_model_huge_reddit",
        "agent": "transformer/crossencoder",
        "task": "pretrained_transformers",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/",
        "description": (
            "Cross-Encoder pretrained on Reddit. Use this model as an ``--init-model`` for a cross-encoder "
            "when fine-tuning on another task. For more details on how to train, see the project page."
        ),
        "example": (
            "python -u examples/train_model.py "
            "--init-model zoo:pretrained_transformers/cross_model_huge_reddit/model "
            "-t convai2 "
            "--model transformer/crossencoder --batchsize 16 --eval-batchsize 10 "
            "--warmup_updates 1000 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 "
            "-lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 "
            "--text-truncate 360 --num-epochs 12.0 --max_train_time 200000 -veps 0.5 "
            "-vme 2500 --validation-metric accuracy --validation-metric-mode max "
            "--save-after-valid True --log_every_n_secs 20 --candidates inline --fp16 True "
            "--dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 "
            "--variant xlm --reduction-type first --share-encoders False "
            "--learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 "
            "--attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 "
            "--embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 "
            "--learn-embeddings True --dict-endtoken __start__ "
            "--model-file <YOUR MODEL FILE>"
        ),
        "result": (
            "(subject to some variance, you may see the following as a result of validation of the model)\n"
            "{'exs': 7801, 'accuracy': 0.903 ...}"
        ),
    },
    {
        "title": "Cross-Encoder Transformer Wikipedia/Toronto Books Pretrained Model",
        "id": "pretrained_transformers",
        "path": "zoo:pretrained_transformers/cross_model_huge_wikito",
        "agent": "transformer/crossencoder",
        "task": "pretrained_transformers",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/",
        "description": (
            "Cross-Encoder pretrained on Wikipedia/Toronto Books. Use this model as an ``--init-model`` for a poly-encoder "
            "when fine-tuning on another task. For more details on how to train, see the project page."
        ),
        "example": (
            "python -u examples/train_model.py "
            "--init-model zoo:pretrained_transformers/cross_model_huge_wikito/model "
            "-t convai2 "
            "--model transformer/crossencoder --batchsize 16 --eval-batchsize 10 "
            "--warmup_updates 1000 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 "
            "-lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 "
            "--text-truncate 360 --num-epochs 12.0 --max_train_time 200000 -veps 0.5 "
            "-vme 2500 --validation-metric accuracy --validation-metric-mode max "
            "--save-after-valid True --log_every_n_secs 20 --candidates inline --fp16 True "
            "--dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 "
            "--variant xlm --reduction-type first --share-encoders False "
            "--learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 "
            "--attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 "
            "--embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 "
            "--learn-embeddings True --dict-endtoken __start__ "
            "--model-file <YOUR MODEL FILE>"
        ),
        "result": (
            "(subject to some variance, you may see the following as a result of validation of the model)\n"
            "{'exs': 7801, 'accuracy': 0.873 ...}"
        ),
    },
    {
        "title": "Poly-Encoder Transformer ConvAI2 Model",
        "id": "pretrained_transformers",
        "path": "zoo:pretrained_transformers/model_poly",
        "agent": "transformer/polyencoder",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/",
        "description": (
            "Polyencoder pretrained on Reddit and fine-tuned on ConvAI2 scoring 89+ hits @ 1/20. See the pretrained_transformers directory for a list of other available pretrained transformers"
        ),
        "example": (
            "python examples/interactive.py -mf "
            "zoo:pretrained_transformers/model_poly/model -t convai2"
        ),
        "result": (
            "hi how are you doing ?\n"
            "[Polyencoder]: i am alright . i am back from the library .\n"
            "Enter Your Message: oh, what do you do for a living?\n"
            "[Polyencoder]: i work at the museum downtown . i love it there .\n"
            "Enter Your Message: what is your favorite drink?\n"
            "[Polyencoder]: i am more of a tea guy . i get my tea from china .\n"
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:pretrained_transformers/model_poly/model -t convai2 --eval-candidates inline"
        ),
        "result2": (
            "[ Finished evaluating tasks ['convai2'] using datatype valid ]\n"
            "{'exs': 7801, 'accuracy': 0.8942, 'f1': 0.9065, 'hits@1': 0.894, 'hits@5': 0.99, 'hits@10': 0.997, 'hits@100': 1.0, 'bleu': 0.8941, 'lr': 5e-09, 'total_train_updates': 0, 'examples': 7801, 'loss': 3004.0, 'mean_loss': 0.385, 'mean_rank': 1.234, 'mrr': 0.9359}"
        ),
    },
    {
        "title": "Bi-Encoder Transformer ConvAI2 Model",
        "id": "pretrained_transformers",
        "path": "zoo:pretrained_transformers/model_bi",
        "agent": "transformer/biencoder",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/",
        "description": (
            "Bi-encoder pretrained on Reddit and fine-tuned on ConvAI2 scoring ~87 hits @ 1/20."
        ),
        "example": (
            "python examples/interactive.py -mf "
            "zoo:pretrained_transformers/model_bi/model -t convai2"
        ),
        "result": (
            "hi how are you doing ?\n"
            "[Biencoder]: my mother is from russia .\n"
            "Enter Your Message: oh cool, whereabouts ?\n"
            "[Biencoder]: no , she passed away when i was 18 . thinking about russian recipes she taught me ,\n"
            "Enter Your Message: what do you cook?\n"
            "[Biencoder]: like meat mostly , me and my dogs love them , do you like dogs ?\n"
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:pretrained_transformers/model_bi/model -t convai2 --eval-candidates inline"
        ),
        "result2": (
            "[ Finished evaluating tasks ['convai2'] using datatype valid ]\n"
            "{'exs': 7801, 'accuracy': 0.8686, 'f1': 0.8833, 'hits@1': 0.869, 'hits@5': 0.987, 'hits@10': 0.996, 'hits@100': 1.0, 'bleu': 0.8685, 'lr': 5e-09, 'total_train_updates': 0, 'examples': 7801, 'loss': 28.77, 'mean_loss': 0.003688, 'mean_rank': 1.301, 'mrr': 0.9197}"
        ),
    },
    {
        "title": "TransResNet (ResNet152) Image-Chat model",
        "id": "image_chat",
        "path": "zoo:image_chat/transresnet_multimodal",
        "agent": "projects.image_chat.transresnet_multimodal.transresnet_multimodal:TransresnetMultimodalAgent",  # noqa: E501
        "task": "image_chat",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/image_chat",
        "description": (
            "Transresnet Multimodal Model pretrained on the Image-Chat task"
        ),
        "example": (
            "python examples/eval_model.py -t image_chat "
            "-mf zoo:image_chat/transresnet_multimodal/model -dt test"
        ),
        "result": "{'exs': 29991, 'accuracy': 0.4032, 'f1': 0.4432, 'hits@1': 0.403, 'hits@5': 0.672, 'hits@10': 0.779, 'hits@100': 1.0, 'bleu': 0.3923,"  # noqa: E501
        "'first_round': {'hits@1/100': 0.3392, 'loss': -0.002001, 'med_rank': 3.0},"
        "'second_round': {'hits@1/100': 0.4558, 'loss': -0.002001, 'med_rank': 2.0},"
        "'third_round+': {'hits@1/100': 0.4147, 'loss': -0.002001, 'med_rank': 2.0}}"  # noqa: E501
        "'hits@10': 0.903, 'hits@100': 0.998, 'bleu': 0.4999, 'hits@1/100': 1.0, 'loss': -0.002, 'med_rank': 1.0}",  # noqa: E501
    },
    {
        "title": "Self-feeding Chatbot",
        "id": "self_feeding",
        "path": "zoo:self_feeding/model",
        "agent": "projects.self_feeding.self_feeding_agent:SelfFeedingAgent",
        "task": "self_feeding:all:train",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/self_feeding",
        "description": (
            "The self-feeding chatbot of Hancock, et al., 2019 "
            "(https://arxiv.org/abs/1901.05415). This model learns from is mistakes "
            "when actually talking with users. This particular model corresponds to "
            "the model with 131k human-human chats + 60k human-bot chats + 60k "
            "feedback chats."
        ),
        "example": (
            "python projects/self_feeding/interactive.py --model-file "
            "zoo:self_feeding/hh131k_hb60k_fb60k_st1k/model --no-cuda true"
        ),
        "result": (
            "Enter Your Message: hi, my name is stephen. what's yours?\n"
            "[SelfFeeding]: hi there greg . do you have pets ? i've 2 cats named "
            "milo and fio .\n"
            "Enter Your Message: sadly, i have no pets. my landlord isn't a fan.\n"
            "[SelfFeeding]: sorry to hear that . i always had bad allergies when i "
            "liven on my farm in kansas ."
        ),
        "example2": (
            "python examples/eval_model.py -mf "
            "zoo:self_feeding/hh131k_hb60k_fb60k_st1k/model -t self_feeding:all"
        ),
        "result2": (
            "[ Finished evaluating tasks ['self_feeding:all'] using datatype valid ]\n"
            "{'exs': 3500, 'dia_rank': 4.654, 'dia_acc': 0.3525, 'fee_rank': 1.0, "
            "'fee_acc': 1.0, 'fee_exs': 1000, 'sat_re': 0.4607, 'sat_f1': 0.5605, "
            "'sat_acc': 0.724}"
        ),
    },
    {
        "title": "Transformer Classifier Single-turn Dialogue Safety Model",
        "id": "dialogue_safety",
        "path": "zoo:dialogue_safety/single_turn/model",
        "agent": "transformer/classifier",
        "task": "dialogue_safety:adversarial,dialogue_safety:standard",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_safety",
        "description": (
            "Classifier trained on both the standard and adversarial safety tasks in addition to Wikipedia Toxic Comments."
        ),
        "example": (
            "python examples/eval_model.py -t dialogue_safety:adversarial "
            "--round 3 -dt test -mf zoo:dialogue_safety/single_turn/model -bs 40"
        ),
        "result": (
            "{'exs': 3000, 'accuracy': 0.9627, 'f1': 0.9627, 'bleu': 9.627e-10, 'lr': 5e-09, 'total_train_updates': 0, 'examples': 3000, 'mean_loss': 0.005441, 'class___notok___recall': 0.7833, 'class___notok___prec': 0.8333, 'class___notok___f1': 0.8076, 'class___ok___recall': 0.9826, 'class___ok___prec': 0.9761, 'class___ok___f1': 0.9793, 'weighted_f1': 0.9621}"
        ),
    },
    {
        "title": "BERT Classifier Multi-turn Dialogue Safety Model",
        "id": "dialogue_safety",
        "path": "zoo:dialogue_safety/multi_turn/model",
        "agent": "bert_classifier",
        "task": "dialogue_safety:multiturn",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_safety",
        "description": (
            "Classifier trained on the multi-turn adversarial safety task in addition to both the single-turn standard and adversarial safety tasks and Wikipedia Toxic Comments."
        ),
        "example": (
            "python examples/eval_model.py -t dialogue_safety:multiturn -dt test -mf zoo:dialogue_safety/multi_turn/model --split-lines True -bs 40"
        ),
        "result": (
            "{'exs': 3000, 'accuracy': 0.9317, 'f1': 0.9317, 'bleu': 9.317e-10, 'lr': 5e-09, 'total_train_updates': 0, 'examples': 3000, 'mean_loss': 0.008921, 'class___notok___recall': 0.7067, 'class___notok___prec': 0.6444, 'class___notok___f1': 0.6741, 'class___ok___recall': 0.9567, 'class___ok___prec': 0.9671, 'class___ok___f1': 0.9618, 'weighted_f1': 0.9331}"
        ),
    },
    {
        "title": "Integration Test Models",
        "id": "unittest",
        "path": "zoo:unittest/transformer_ranker/model",
        "task": "integration_tests",
        "description": (
            "Model files used to check backwards compatibility and code coverage of important standard models."
        ),
        "example": (
            "python examples/eval_model.py -mf zoo:unittest/transformer_generator2/model -t integration_tests:multiturn_candidate -m transformer/generator"
        ),
        "external_website": '',
        "result": (
            """{'exs': 400, 'accuracy': 1.0, 'f1': 1.0, 'bleu-4': 0.2503, 'lr': 0.001, 'total_train_updates': 5000, 'gpu_mem_percent': 9.37e-05, 'loss': 0.0262, 'token_acc': 1.0, 'nll_loss': 7.935e-05, 'ppl': 1.0}"""
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue All Tasks MT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/all_tasks_mt/model",
        "agent": "image_seq2seq",
        "task": "#Dodeca",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": ("Image Seq2Seq model trained on all DodecaDialogue tasks"),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/all_tasks_mt/model "
            "--inference beam --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how are you?\n"
            "[ImageSeq2seq]: i ' m doing well . how are you ?\n"
            "Enter Your Message: not much, what do you like to do?\n"
            "[ImageSeq2seq]: i like to go to the park and play with my friends ."
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/all_tasks_mt/model -t \"#Dodeca\""
            "--prepend-personality True --prepend-gold-knowledge True --image-mode no_image_model"
        ),
        "result2": (
            "[ Finished evaluating tasks ['#Dodeca'] using datatype valid ]\n"
            "                           exs  gpu_mem  loss        lr   ppl  token_acc  total_train_updates  tpb\n"
            "   WizTeacher             3939          2.161           8.678      .5325\n"
            "   all                   91526    .3371 2.807 9.375e-07 18.23      .4352               470274 2237\n"
            "   convai2                7801          2.421           11.26      .4721\n"
            "   cornell_movie         13905          3.088           21.93      .4172\n"
            "   dailydialog            8069           2.47           11.82      .4745\n"
            "   empathetic_dialogues   5738          2.414           11.18      .4505\n"
            "   igc                     486          2.619           13.73      .4718\n"
            "   image_chat:Generation 15000          3.195           24.42      .3724\n"
            "   light_dialog           6623          2.944              19      .3918\n"
            "   twitter               10405           3.61           36.98      .3656\n"
            "   ubuntu                19560          3.148            23.3      .4035"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue ConvAI2 FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/convai2_ft/model",
        "agent": "image_seq2seq",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on Convai2"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/convai2_ft/model -t convai2 "
            "--inference beam --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "[context]: your persona: i currently work for ibm in chicago.\n"
            "your persona: i'm not a basketball player though.\n"
            "your persona: i am almost 7 feet tall.\n"
            "your persona: i'd like to retire to hawaii in the next 10 years.\n"
            "Enter Your Message: hi how's it going\n"
            "[ImageSeq2seq]: i ' m doing well . how are you ?\n"
            "Enter Your Message: i'm well, i am really tall\n"
            "[ImageSeq2seq]: that ' s cool . i like simple jokes ."
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/convai2_ft/model -t convai2"
        ),
        "result2": (
            "[ Finished evaluating tasks ['convai2'] using datatype valid ]\n"
            "    exs  gpu_mem  loss      lr   ppl  token_acc  total_train_updates   tpb\n"
            "   7801    .2993 2.415 7.5e-06 11.19      .4741                15815 845.8"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue Cornell Movie FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/cornell_movie_ft/model",
        "agent": "image_seq2seq",
        "task": "cornell_movie",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the Cornell Movie task"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/cornell_movie_ft/model "
            "--inference beam --beam-size 10 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going?\n"
            "[ImageSeq2seq]: oh , it ' s great . i ' m having a great time . how are you doing ?\n"
            "Enter Your Message: i'm doing well, what do you like to do?\n"
            "[ImageSeq2seq]: i like to go to the movies . what about you ? do you have any hobbies ?"
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/cornell_movie_ft/model -t cornell_movie"
        ),
        "result2": (
            "[ Finished evaluating tasks ['cornell_movie'] using datatype valid ]\n"
            "     exs  gpu_mem  loss      lr   ppl  token_acc  total_train_updates   tpb\n"
            "   13905   .07094 2.967 2.5e-06 19.43      .4290                29496 15.76"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue DailyDialog FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/daily_dialog_ft/model",
        "agent": "image_seq2seq",
        "task": "dailydialog",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the DailyDialog task"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/daily_dialog_ft/model "
            "--inference beam --beam-size 5 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going\n"
            "[ImageSeq2seq]: i ' m doing well . how about you ?\n"
            "Enter Your Message: not much, what do you like to do?\n"
            "[ImageSeq2seq]: i like to go to the beach and play volleyball ."
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/daily_dialog_ft/model -t dailydialog"
        ),
        "result2": (
            "[ Finished evaluating tasks ['dailydialog'] using datatype valid ]\n"
            "    exs  gpu_mem  loss      lr   ppl  token_acc  total_train_updates   tpb\n"
            "   8069   .06787 2.326 7.5e-06 10.24      .5093               150959 15.67"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue ELI5 FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/eli5_ft/model",
        "agent": "image_seq2seq",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "task": "TBD",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the ELI5 task"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/eli5_ft/model "
            "--inference beam --beam-size 10 --beam-min-length 200 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: Hi, can you tell me about quantum physics?\n"
            "[ImageSeq2seq]: yes , i can . quantum physics is the study of how particles "
            "interact with each other , and how they interact with other particles . "
            "it ' s important to note that quantum mechanics is n ' t the same as "
            "classical physics . classical physics is a study of the properties of "
            "particles , and what they do . in classical physics , there are two "
            "types of particles : quarks and neutrinos . quarks are made up of quarks , "
            "neutrinos , and electrons . neutrinos are made of protons , neutrons , "
            "electrons , and neutrons . they ' re all the same thing , but they all "
            "have the same properties . so , if you ' re interested in quantum physics , "
            "you might want to check out / r / askscience . there ' s a subreddit "
            "for that sort of thing . edit : i ' m not sure what you mean by "
            "\" quantum physics \" , but i ' ll let you know if you want to know more . "
            "edit 2 : thanks for the gold !"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue Empathetic Dialogue FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/empathetic_dialogues_ft/model",
        "agent": "image_seq2seq",
        "task": "empathetic_dialogues",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the Empathetic Dialogue task"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/empathetic_dialogues_ft/model "
            "--inference beam --beam-size 5 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi, how's it going?\n"
            "[ImageSeq2seq]: i ' m doing well . how are you ?\n"
            "Enter Your Message: i'm fine, feeling a little sad\n"
            "[ImageSeq2seq]: that ' s too bad . what ' s going on ?"
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/empathetic_dialogues_ft/model -t empathetic_dialogues"
        ),
        "result2": (
            "[ Finished evaluating tasks ['empathetic_dialogues'] using datatype valid ]\n"
            "    exs  gpu_mem  loss      lr   ppl  token_acc  total_train_updates  tpb\n"
            "   5738    .3278 2.405 7.5e-06 11.08      .4517                20107 1914"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue Image Grounded Conversations FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/igc_ft/model",
        "agent": "image_seq2seq",
        "task": "igc",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the Image Grounded Conversations task"
        ),
        "example": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/igc_ft/model -t igc:responseOnly"
        ),
        "result": (
            "[ Finished evaluating tasks ['igc:responseOnly'] using datatype valid ]\n"
            "    exs  gpu_mem  loss    lr   ppl  token_acc  total_train_updates   tpb\n"
            "    162    .0726 2.832 1e-06 16.98      .4405                10215 9.852"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue Image Chat FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/image_chat_ft/model",
        "agent": "image_seq2seq",
        "task": "image_chat",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the Image Chat task"
        ),
        "example": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/image_chat_ft/model -t image_chat:generation "
            "--image-mode no_image_model"
        ),
        "result": (
            "[ Finished evaluating tasks ['image_chat:generation'] using datatype valid ]\n"
            "     exs  gpu_mem  loss        lr   ppl  token_acc  total_train_updates  tpb\n"
            "   15000    .2231 4.353 3.125e-07 77.73      .2905               321001 1653"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue LIGHT Dialogue FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/light_dialog_ft/model",
        "agent": "image_seq2seq",
        "task": "light_dialog",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the LIGHT Dialogue task"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/light_dialog_ft/model "
            "--inference beam --beam-size 5 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going?\n"
            "[ImageSeq2seq]: i ' m doing well . how about you ? what ' s going on in the world today ?\n"
            "Enter Your Message: not much, wish it had some more epic battles!\n"
            "[ImageSeq2seq]: me too . it ' s been so long since i ' ve seen a battle like this . do you have a favorite battle ?"
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/light_dialog_ft/model -t light_dialog"
        ),
        "result2": (
            "[ Finished evaluating tasks ['light_dialog'] using datatype valid ]\n"
            "    exs  gpu_mem  loss      lr   ppl  token_acc  total_train_updates   tpb\n"
            "   6623   .07002 2.927 7.5e-06 18.66      .3927                38068 20.81"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue pushshift.io Reddit FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/reddit_ft/model",
        "agent": "image_seq2seq",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "task": "TBD",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the pushshift.io Reddit task"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/reddit_ft/model "
            "--inference beam --beam-size 5 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going?\n"
            "[ImageSeq2seq]: hi , i ' m doing pretty well . how are you ? : ) and yourself ? : d\n"
            "Enter Your Message: just hanging in there, you up to anything fun?\n"
            "[ImageSeq2seq]: not really . i just got home from work . i ' ll be back in a few hours ."
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue Twitter FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/twitter_ft/model",
        "agent": "image_seq2seq",
        "task": "twitter",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the Twitter task"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/twitter_ft/model "
            "--inference beam --beam-size 10 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going?\n"
            "[ImageSeq2seq]: it ' s going well ! how are you ? @ smiling_face_with_heart - eyes @\n"
            "Enter Your Message: im doing well, what do you like to do\n"
            "[ImageSeq2seq]: hi ! i ' m doing well ! i like to read , watch movies , play video games , and listen to music . how about you ?"
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/twitter_ft/model -t twitter"
        ),
        "result2": (
            "[ Finished evaluating tasks ['twitter'] using datatype valid ]\n"
            "     exs  gpu_mem  loss      lr   ppl  token_acc  total_train_updates  tpb\n"
            "   10405    .3807 3.396 7.5e-06 29.83      .3883               524029 2395"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue Ubuntu V2 FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/ubuntu_ft/model",
        "agent": "image_seq2seq",
        "task": "ubuntu",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the Ubuntu V2 task"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/ubuntu_ft/model "
            "--inference beam --beam-size 2 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going?\n"
            "[ImageSeq2seq]: i ' m fine . . . you ? .\n"
            "Enter Your Message: doing ok, what do you like to do?\n"
            "[ImageSeq2seq]: i like to read , write , and read ."
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/ubuntu_ft/model -t ubuntu"
        ),
        "result2": (
            "[ Finished evaluating tasks ['ubuntu'] using datatype valid ]\n"
            "     exs  gpu_mem  loss      lr   ppl  token_acc  total_train_updates  tpb\n"
            "   19560    .3833 2.844 2.5e-05 17.18      .4389               188076 3130"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue Wizard of Wikipedia FT Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/wizard_of_wikipedia_ft/model",
        "agent": "image_seq2seq",
        "task": "wizard_of_wikipedia:Generator",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "description": (
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the Wizard of Wikipedia task"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:dodecadialogue/wizard_of_wikipedia_ft/model "
            "--inference beam --beam-size 10 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3 -t wizard_of_wikipedia"
        ),
        "result": (
            "Please choose one of the following topics by typing A, B, C, ..., etc. :\n"
            "\n"
            "A: Denver Art Museum\n"
            "B: Joke\n"
            "C: Sport utility vehicle\n"
            "D: Chocolate brownie\n"
            "E: [NO TOPIC]\n"
            "\n"
            "Enter Your Message: D\n"
            "[ Your chosen topic is: Chocolate brownie ]\n"
            "/private/home/kshuster/ParlAI/parlai/core/torch_ranker_agent.py:670: UserWarning: [ Executing eval mode with provided inline set of candidates ]\n"
            "  ''.format(mode)\n"
            "[ Using chosen sentence from Wikpedia ]: A chocolate brownie (commonly referred to as simply brownie) is a square, baked, chocolate dessert.\n"
            "[ImageSeq2seq]: i love chocolate brownies . they ' re my favorite .\n"
            "Enter Your Message: me too! do you ever make them?\n"
            "[ Using chosen sentence from Wikpedia ]: Brownies are typically eaten by hand, often accompanied by milk, served warm with ice cream (a la mode), topped with whipped cream, or sprinkled with powdered sugar and fudge.\n"
            "[ImageSeq2seq]: i don ' t make them , but i would love to ."
        ),
        "example2": (
            "python examples/eval_model.py -mf zoo:dodecadialogue/wizard_of_wikipedia_ft/model -t wizard_of_wikipedia:Generator --prepend-gold-knowledge true"
        ),
        "result2": (
            "[ Finished evaluating tasks ['wizard_of_wikipedia:Generator'] using datatype valid ]\n"
            "    exs  gpu_mem  loss      lr   ppl  token_acc  total_train_updates  tpb\n"
            "   3939    .3823 2.144 7.5e-06 8.532      .5348                22908 2852"
        ),
    },
    {
        "title": "ImageSeq2Seq DodecaDialogue Base Model",
        "id": "dodecadialogue",
        "path": "zoo:dodecadialogue/base_model/model",
        "agent": "image_seq2seq",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dodecadialogue/",
        "task": "#Dodeca",
        "description": (
            "Image Seq2Seq base model, from which all DodecaDialogue models were trained"
        ),
        "example": (
            "python examples/train_model.py -t \"#Dodeca\" --prepend-gold-knowledge true --prepend-personality true -mf /tmp/dodeca_model --init-model zoo:dodecadialogue/base_model/model --dict-file zoo:dodecadialogue/dict/dodeca.dict --model image_seq2seq --dict-tokenizer bpe --dict-lower true -bs 32 -eps 0.5 -esz 512 --ffn-size 2048 --fp16 false --n-heads 16 --n-layers 8 --n-positions 512 --text-truncate 512 --label-truncate 128 --variant xlm -lr 7e-6 --lr-scheduler reduceonplateau --optimizer adamax --dropout 0.1 --validation-every-n-secs 3600 --validation-metric ppl --validation-metric-mode min --validation-patience 10 --activation gelu --embeddings-scale true --learn-positional-embeddings true --betas 0.9,0.999 --warmup-updates 2000 --gradient-clip 0.1"
        ),
        "result": ("A trained model (logs omitted)"),
    },
    {
        "title": "Tutorial Transformer Generator",
        "id": "tutorial_transformer_generator",
        "path": "zoo:tutorial_transformer_generator/model",
        "task": "pushshift.io",
        "description": (
            "Small (87M paramter) generative transformer, pretrained on pushshift.io Reddit."
        ),
        "example": (
            "python -m parlai.scripts.interactive -mf zoo:tutorial_transformer_generator/model"
        ),
        "external_website": '',
        "result": (
            "Enter Your Message: hi, how are you today?\n"
            "[TransformerGenerator]: i ' m doing well , how about you ?\n"
            "Enter Your Message: I'm giving a tutorial on chatbots!\n"
            "[TransformerGenerator]: that ' s awesome ! what ' s it about ?\n"
            "Enter Your Message: bots just like you\n"
            "[TransformerGenerator]: i ' ll be sure to check it out !"
        ),
    },
    {
        "title": "Blender 90M",
        "id": "blender",
        "path": "zoo:blender/blender_90M/model",
        "agent": "transformer/generator",
        "task": "blended_skill_talk",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/blender",
        "description": (
            "90< parameter generative model finetuned on blended_skill_talk tasks."
        ),
        "example": (
            "python parlai/scripts/safe_interactive.py -mf zoo:blender/blender_90M/model -t blended_skill_talk"
        ),
        "result": (
            "Enter Your Message: Hi what's up?\n"
            "[TransformerGenerator]: hello , how are you ? i just got back from working at a law firm , how about you ?"
        ),
    },
    {
        "title": "Blender 2.7B",
        "id": "blender",
        "path": "zoo:blender/blender_3B/model",
        "agent": "transformer/generator",
        "task": "blended_skill_talk",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/blender",
        "description": (
            "2.7B parameter generative model finetuned on blended_skill_talk tasks."
        ),
        "example": (
            "python parlai/scripts/safe_interactive.py -mf zoo:blender/blender_3B/model -t blended_skill_talk"
        ),
        "result": (
            "Enter Your Message: Hi how are you?\n"
            "[TransformerGenerator]: I'm doing well. How are you doing? What do you like to do in your spare time?"
        ),
    },
    {
        "title": "Blender 9.4B",
        "id": "blender",
        "path": "zoo:blender/blender_9B/model",
        "agent": "transformer/generator",
        "task": "blended_skill_talk",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/blender",
        "description": (
            "9.4B parameter generative model finetuned on blended_skill_talk tasks."
        ),
        "example": (
            "python parlai/scripts/safe_interactive.py -mf zoo:blender/blender_9B/model -t blended_skill_talk"
        ),
        "result": (
            "Enter Your Message: Hi!\n"
            "[TransformerGenerator]: What do you do for a living? I'm a student at Miami University."
        ),
    },
    {
        "title": "Reddit 2.7B",
        "id": "blender",
        "path": "zoo:blender/reddit_3B/model",
        "agent": "transformer/generator",
        "task": "pushshift.io",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/blender",
        "description": (
            "2.7B parameter generative model finetuned on blended_skill_talk tasks."
        ),
        "example": (
            "python examples/train_model.py -t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues --multitask-weights 1,3,3,3 -veps 0.25 --attention-dropout 0.0 --batchsize 128 --model transformer/generator --embedding-size 2560 --ffn-size 10240 --variant prelayernorm --n-heads 32 --n-positions 128 --n-encoder-layers 2 --n-decoder-layers 24 --history-add-global-end-token end --delimiter '  ' --dict-tokenizer bytelevelbpe  --dropout 0.1 --fp16 True --init-model zoo:blender/reddit_3B/model --dict-file zoo:blender/reddit_3B/model.dict --label-truncate 128 --log_every_n_secs 10 -lr 7e-06 --lr-scheduler reduceonplateau --lr-scheduler-patience 3 --optimizer adam --relu-dropout 0.0 --activation gelu --model-parallel true --save-after-valid True --text-truncate 128 --truncate 128 --warmup_updates 100 --fp16-impl mem_efficient --update-freq 2 --gradient-clip 0.1 --skip-generation True -vp 10 -vmt ppl -vmm min --model-file /tmp/test_train_27B"
        ),
        "result": ("Results vary."),
    },
    {
        "title": "Reddit 9.4B",
        "id": "blender",
        "path": "zoo:blender/reddit_9B/model",
        "agent": "transformer/generator",
        "task": "pushshift.io",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/blender",
        "description": (
            "9.4B parameter generative model finetuned on blended_skill_talk tasks."
        ),
        "example": (
            "python examples/train_model.py -t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues --multitask-weights 1,3,3,3 -veps 0.25 --attention-dropout 0.0 --batchsize 8 --eval-batchsize 64 --model transformer/generator --embedding-size 4096 --ffn-size 16384 --variant prelayernorm --n-heads 32 --n-positions 128 --n-encoder-layers 4 --n-decoder-layers 32 --history-add-global-end-token end --dict-tokenizer bytelevelbpe --dropout 0.1 --fp16 True --init-model zoo:blender/reddit_9B/model --dict-file zoo:blender/reddit_9B/model.dict --label-truncate 128 -lr 3e-06 -dynb full --lr-scheduler cosine --max-lr-steps 9000 --lr-scheduler-patience 3 --optimizer adam --relu-dropout 0.0 --activation gelu --model-parallel true --save-after-valid False --text-truncate 128 --truncate 128 --warmup_updates 1000 --fp16-impl mem_efficient --update-freq 4 --log-every-n-secs 30 --gradient-clip 0.1 --skip-generation True -vp 10 --max-train-time 84600 -vmt ppl -vmm min --model-file /tmp/test_train_94B"
        ),
        "result": ("Results vary."),
    },
]
