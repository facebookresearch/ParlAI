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
            "-t convai2 --shuffle true "
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
            "-t convai2 --shuffle true "
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
            "--shuffle true --model transformer/biencoder --eval-batchsize 6 "
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
            "--shuffle true --model transformer/biencoder --eval-batchsize 6 "
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
            "-t convai2 --shuffle true "
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
            "-t convai2 --shuffle true "
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
]
