#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains a list of all the models in the model zoo, the path to
load them, agents & tasks associated (e.g. they were trained using) and a
description. Using the path you should be able to download and use the model
automatically, e.g.:
 python examples/interactive.py --model-file\
    "zoo:wikipedia_2016-12-21/tfidf_retriever/drqa_docs"
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
        "example": (
            "python -m parlai.scripts.interactive -mf zoo:convai2/kvmemnn/model"
        ),
    },
    {
        "title": "Seq2Seq ConvAI2 model",
        "id": "convai2",
        "path": "zoo:convai2/seq2seq/convai2_self_seq2seq_model",
        "agent": "legacy:seq2seq:0",
        "task": "convai2",
        "description": (
            "SeqSeq trained on the ConvAI2 task, used as a baseline in the competition."
        ),
        "example": (
            "python -m parlai.scripts.interactive -mf "
            "zoo:convai2/seq2seq/convai2_self_seq2seq_model -m legacy:seq2seq:0"
        ),
    },
    {
        "title": "Seq2Seq Twitter model",
        "id": "twitter",
        "path": "zoo:twitter/seq2seq/twitter_seq2seq_model",
        "agent": "legacy:seq2seq:0",
        "task": "twitter",
        "description": ("Seq2Seq conversational model trained on the Twitter task"),
        "result": "{'exs': 10405, 'accuracy': 0.001538, 'f1': 0.07537, 'bleu': 0.002304, 'loss': 3.93, 'ppl': 50.9}",  # noqa: E501
    },
    {
        "title": "DrQA SQuAD model",
        "id": "drqa",
        "path": "zoo:drqa/squad/model",
        "agent": "drqa",
        "task": "squad",
        "description": "drqa reader trained on SQuAD",
        "result": (
            "{'exs': 10570, 'accuracy': 0.6886, 'f1': 0.7821, 'hits@1': 0.689, 'hits@5': 0.689, 'hits@10': 0.689, 'hits@100': 0.689, 'bleu': 0.1364, 'train_loss': 0}"  # noqa: E501
        ),
    },
    {
        "title": "Wikipedia Retriever (used for open SQuAD)",
        "id": "wikipedia_2016-12-21",
        "path": "zoo:wikipedia_2016-12-21/tfidf_retriever/drqa_docs",
        "agent": "tfidf_retriever",
        "task": "wikipedia:full",
        "example": (
            "python -m parlai.scripts.interactive --model tfidf_retriever "
            "-mf zoo:wikipedia_2016-12-21/tfidf_retriever/drqa_docs"
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
        "description": (
            "Retrieval over Wikipedia dump, used for DrQA on the open squad " "dataset."
        ),
        "example": "python -m parlai.scripts.interactive --model tfidf_retriever -mf zoo:wikipedia_full/tfidf_retriever/model",  # noqa: E501
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
        "task": "wizard_of_wikipedia",
        "example2": "examples/interactive.py -m projects:wizard_of_wikipedia:interactive_retrieval -t wizard_of_wikipedia",
        "result2": 
        ("[ Your chosen topic is: Teapot ]\n"
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
        "task": "light_dialog",
        "description": ("LIGHT Dialogue task, replicating the numbers from the paper."),
        "example": (
           "python examples/eval_model.py -t light_dialog "
            "-mf zoo:light/biranker_dialogue/model"
        ),
        "result": "{'exs': 6623, 'accuracy': 0.7586, 'f1': 0.7802, 'hits@1': 0.759, 'hits@5': 0.965,"  # noqa: E501
        "'hits@10': 0.994, 'hits@100': 1.0, 'bleu': 0.7255, 'lr': 5e-05, 'num_updates': 15050,"  # noqa: E501
        "'examples': 6623, 'loss': 5307.0, 'mean_loss': 0.8013, 'mean_rank': 1.599, 'train_accuracy': 0}",  # noqa: E501
    },
    {
        "title": "Legacy Seq2Seq Twitter model",
        "id": "twitter",
        "path": "zoo:twitter/seq2seq/twitter_seq2seq_model",
        "agent": "legacy:seq2seq:0",
        "task": "twitter",
        "description": ("Generic conversational model trained on the twitter task"),
        "result": "{'exs': 10405, 'accuracy': 0.001538, 'f1': 0.07537, 'bleu': 0.002304, 'loss': 3.93, 'ppl': 50.9}",  # noqa: E501
    },
    {
        "title": "Controllable Dialogue ConvAI2 model",
        "id": "controllable_dialogue",
        "path": "zoo:controllable_dialogue/convai2_finetuned_baseline",
        "agent": "projects.controllable_dialogue.controllable_seq2seq.controllable_seq2seq:ControllableSeq2seqAgent",  # noqa: E501
        "task": "convai2",
        "example": "python -m parlai.scripts.eval_model --model projects.controllable_dialogue.controllable_seq2seq.controllable_seq2seq:ControllableSeq2seqAgent --task projects.controllable_dialogue.tasks.agents -mf zoo:controllable_dialogue/convai2_finetuned_baseline",
        "description": ("Seq2Seq model with control trained on ConvAI2"),
    },
    {
        "title": "TransResNet (ResNet 152) Personality-Captions model",
        "id": "personality_captions",
        "path": "zoo:personality_captions/transresnet",
        "agent": "projects.personality_captions.transresnet.transresnet:TransresnetAgent",  # noqa: E501
        "task": "personality_captions",
        "description": (
            "Transresnet Model pretrained on the Personality-Captions task"
        ),
        "example": (
            "python examples/eval_model.py -t personality_captions "
            "-mf zoo:personality_captions/transresnet/model --num-test-labels 5 -dt test"
        ),
        "result": "{'exs': 10000, 'accuracy': 0.5113, 'f1': 0.5951, 'hits@1': 0.511, 'hits@5': 0.816, "  # noqa: E501
        "'hits@10': 0.903, 'hits@100': 0.998, 'bleu': 0.4999, 'hits@1/100': 1.0, 'loss': -0.002, 'med_rank': 1.0}",  # noqa: E501
    },
    {
        "title": "Poly-Encoder Transformer ConvAI2 Model",
        "id": "pretrained_transformers",
        "path": "zoo:pretrained_transformers/model_poly",
        "agent": "transformer/polyencoder",  # noqa: E501
        "task": "convai2",
        "description": (
            "Polyencoder pretrained on Reddit and fine-tuned on ConvAI2 scoring 89+ hits@1/20. See the pretrained_transformers directory for a list of other available pretrained transformers"
        ),
        "example": (
            "python examples/interactive.py -mf zoo:pretrained_transformers/model_poly/model -t convai2"
        ),
            #"python examples/interactive.py -m transformer/polyencoder "
            #"-mf zoo:pretrained_transformers/model_poly/model --encode-candidate-vecs true "
            #"--eval-candidates fixed  "
            #"--fixed-candidates-path data/models/pretrained_transformers/convai_trainset_cands.txt"
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
            "{'exs': 7801, 'accuracy': 0.8942, 'f1': 0.9065, 'hits@1': 0.894, 'hits@5': 0.99, 'hits@10': 0.997, 'hits@100': 1.0, 'bleu': 0.8941, 'lr': 5e-09, 'num_updates': 0, 'examples': 7801, 'loss': 3004.0, 'mean_loss': 0.385, 'mean_rank': 1.234, 'mrr': 0.9359}"
            )
    },
    {
        "title": "TransResNet (ResNet152) Image-Chat model",
        "id": "image_chat",
        "path": "zoo:image_chat/transresnet_multimodal",
        "agent": "projects.image_chat.transresnet_multimodal.transresnet_multimodal:TransresnetMultimodalAgent",  # noqa: E501
        "task": "image_chat",
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
]
