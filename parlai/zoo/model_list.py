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

   parlai interactive --model-file
       "zoo:blender/blender_3B/model"


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
        "title": "Wikipedia Retriever (used for Wizard of Wikipedia)",
        "id": "wikipedia_full",
        "path": "zoo:wikipedia_full/tfidf_retriever/model",
        "agent": "tfidf_retriever",
        "task": "wikipedia:full",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia",
        "description": ("Retrieval over Wikipedia dump, used for Wizard of Wikipedia."),
        "example": (
            "parlai interactive --model tfidf_retriever -mf "
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
        "description": "End2End Generative model for Wizard of Wikipedia",
        "task": "wizard_of_wikipedia:generator",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia",
        "example": (
            "parlai display_model -t wizard_of_wikipedia:generator "
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
        "description": "Full Dialogue Retrieval Model for Wizard of Wikipedia",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia",
        "task": "wizard_of_wikipedia",
        "example2": "parlai interactive -m projects:wizard_of_wikipedia:interactive_retrieval -t wizard_of_wikipedia",
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
            "parlai display_model -t wizard_of_wikipedia "
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
        "description": "LIGHT Dialogue task, replicating the numbers from the paper.",
        "example": (
            "parlai eval_model -t light_dialog -mf zoo:light/biranker_dialogue/model"
        ),
        "result": "{'exs': 6623, 'accuracy': 0.7586, 'f1': 0.7802, 'hits@1': 0.759, 'hits@5': 0.965,"  # noqa: E501
        "'hits@10': 0.994, 'hits@100': 1.0, 'bleu': 0.7255, 'lr': 5e-05, 'total_train_updates': 15050,"  # noqa: E501
        "'examples': 6623, 'loss': 5307.0, 'mean_loss': 0.8013, 'mean_rank': 1.599, 'train_accuracy': 0}",  # noqa: E501
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
            "parlai eval_model -t personality_captions "
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
            "parlai train_model "
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
            "parlai train_model "
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
            "parlai train_model "
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
            "parlai train_model "
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
            "parlai train_model "
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
            "parlai train_model "
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
            "parlai interactive -mf "
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
            "parlai eval_model -mf zoo:pretrained_transformers/model_poly/model -t convai2 --eval-candidates inline"
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
            "parlai interactive -mf "
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
            "parlai eval_model -mf zoo:pretrained_transformers/model_bi/model -t convai2 --eval-candidates inline"
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
            "parlai eval_model -t image_chat "
            "-mf zoo:image_chat/transresnet_multimodal/model -dt test"
        ),
        "result": "{'exs': 29991, 'accuracy': 0.4032, 'f1': 0.4432, 'hits@1': 0.403, 'hits@5': 0.672, 'hits@10': 0.779, 'hits@100': 1.0, 'bleu': 0.3923,"  # noqa: E501
        "'first_round': {'hits@1/100': 0.3392, 'loss': -0.002001, 'med_rank': 3.0},"
        "'second_round': {'hits@1/100': 0.4558, 'loss': -0.002001, 'med_rank': 2.0},"
        "'third_round+': {'hits@1/100': 0.4147, 'loss': -0.002001, 'med_rank': 2.0}}"  # noqa: E501
        "'hits@10': 0.903, 'hits@100': 0.998, 'bleu': 0.4999, 'hits@1/100': 1.0, 'loss': -0.002, 'med_rank': 1.0}",  # noqa: E501
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
            "parlai eval_model -t dialogue_safety:adversarial "
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
            "parlai eval_model -t dialogue_safety:multiturn -dt test -mf zoo:dialogue_safety/multi_turn/model --split-lines True -bs 40"
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
            "parlai eval_model -mf zoo:unittest/transformer_generator2/model -t integration_tests:multiturn_candidate -m transformer/generator"
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
        "description": "Image Seq2Seq model trained on all DodecaDialogue tasks",
        "example": (
            "parlai interactive -mf zoo:dodecadialogue/all_tasks_mt/model "
            "--inference beam --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how are you?\n"
            "[ImageSeq2seq]: i ' m doing well . how are you ?\n"
            "Enter Your Message: not much, what do you like to do?\n"
            "[ImageSeq2seq]: i like to go to the park and play with my friends ."
        ),
        "example2": (
            "parlai eval_model -mf zoo:dodecadialogue/all_tasks_mt/model -t \"#Dodeca\""
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
            "parlai interactive -mf zoo:dodecadialogue/convai2_ft/model -t convai2 "
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
            "parlai eval_model -mf zoo:dodecadialogue/convai2_ft/model -t convai2"
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
            "parlai interactive -mf zoo:dodecadialogue/cornell_movie_ft/model "
            "--inference beam --beam-size 10 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going?\n"
            "[ImageSeq2seq]: oh , it ' s great . i ' m having a great time . how are you doing ?\n"
            "Enter Your Message: i'm doing well, what do you like to do?\n"
            "[ImageSeq2seq]: i like to go to the movies . what about you ? do you have any hobbies ?"
        ),
        "example2": (
            "parlai eval_model -mf zoo:dodecadialogue/cornell_movie_ft/model -t cornell_movie"
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
            "parlai interactive -mf zoo:dodecadialogue/daily_dialog_ft/model "
            "--inference beam --beam-size 5 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going\n"
            "[ImageSeq2seq]: i ' m doing well . how about you ?\n"
            "Enter Your Message: not much, what do you like to do?\n"
            "[ImageSeq2seq]: i like to go to the beach and play volleyball ."
        ),
        "example2": (
            "parlai eval_model -mf zoo:dodecadialogue/daily_dialog_ft/model -t dailydialog"
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
            "Image Seq2Seq model trained on all DodecaDialogue tasks and fine-tuned on the ELI5 task. "
            "The ELI5 model in the zoo is the one that scored the highest ROUGE scores; "
            "it may have slightly lower f1, bleu, and slightly higher ppl than the numbers "
            "reported in the dodeca paper."
        ),
        "example": (
            "parlai interactive -mf zoo:dodecadialogue/eli5_ft/model "
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
            "parlai interactive -mf zoo:dodecadialogue/empathetic_dialogues_ft/model "
            "--inference beam --beam-size 5 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi, how's it going?\n"
            "[ImageSeq2seq]: i ' m doing well . how are you ?\n"
            "Enter Your Message: i'm fine, feeling a little sad\n"
            "[ImageSeq2seq]: that ' s too bad . what ' s going on ?"
        ),
        "example2": (
            "parlai eval_model -mf zoo:dodecadialogue/empathetic_dialogues_ft/model -t empathetic_dialogues"
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
            "parlai eval_model -mf zoo:dodecadialogue/igc_ft/model -t igc:responseOnly"
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
            "parlai eval_model -mf zoo:dodecadialogue/image_chat_ft/model -t image_chat:generation "
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
            "parlai interactive -mf zoo:dodecadialogue/light_dialog_ft/model "
            "--inference beam --beam-size 5 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going?\n"
            "[ImageSeq2seq]: i ' m doing well . how about you ? what ' s going on in the world today ?\n"
            "Enter Your Message: not much, wish it had some more epic battles!\n"
            "[ImageSeq2seq]: me too . it ' s been so long since i ' ve seen a battle like this . do you have a favorite battle ?"
        ),
        "example2": (
            "parlai eval_model -mf zoo:dodecadialogue/light_dialog_ft/model -t light_dialog"
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
            "parlai interactive -mf zoo:dodecadialogue/reddit_ft/model "
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
            "parlai interactive -mf zoo:dodecadialogue/twitter_ft/model "
            "--inference beam --beam-size 10 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going?\n"
            "[ImageSeq2seq]: it ' s going well ! how are you ? @ smiling_face_with_heart - eyes @\n"
            "Enter Your Message: im doing well, what do you like to do\n"
            "[ImageSeq2seq]: hi ! i ' m doing well ! i like to read , watch movies , play video games , and listen to music . how about you ?"
        ),
        "example2": (
            "parlai eval_model -mf zoo:dodecadialogue/twitter_ft/model -t twitter"
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
            "parlai interactive -mf zoo:dodecadialogue/ubuntu_ft/model "
            "--inference beam --beam-size 2 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3"
        ),
        "result": (
            "Enter Your Message: hi how's it going?\n"
            "[ImageSeq2seq]: i ' m fine . . . you ? .\n"
            "Enter Your Message: doing ok, what do you like to do?\n"
            "[ImageSeq2seq]: i like to read , write , and read ."
        ),
        "example2": (
            "parlai eval_model -mf zoo:dodecadialogue/ubuntu_ft/model -t ubuntu"
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
            "parlai interactive -mf zoo:dodecadialogue/wizard_of_wikipedia_ft/model "
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
            "parlai eval_model -mf zoo:dodecadialogue/wizard_of_wikipedia_ft/model -t wizard_of_wikipedia:Generator --prepend-gold-knowledge true"
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
            "parlai train_model -t \"#Dodeca\" --prepend-gold-knowledge true --prepend-personality true -mf /tmp/dodeca_model --init-model zoo:dodecadialogue/base_model/model --dict-file zoo:dodecadialogue/dict/dodeca.dict --model image_seq2seq --dict-tokenizer bpe --dict-lower true -bs 32 -eps 0.5 -esz 512 --ffn-size 2048 --fp16 false --n-heads 16 --n-layers 8 --n-positions 512 --text-truncate 512 --label-truncate 128 --variant xlm -lr 7e-6 --lr-scheduler reduceonplateau --optimizer adamax --dropout 0.1 --validation-every-n-secs 3600 --validation-metric ppl --validation-metric-mode min --validation-patience 10 --activation gelu --embeddings-scale true --learn-positional-embeddings true --betas 0.9,0.999 --warmup-updates 2000 --gradient-clip 0.1"
        ),
        "result": "A trained model (logs omitted)",
    },
    {
        "title": "BlendedSkillTalk: BlendedSkillTalk single-task model",
        "id": "blended_skill_talk",
        "path": "zoo:blended_skill_talk/bst_single_task/model",
        "agent": "transformer/polyencoder",
        "task": "blended_skill_talk",
        "project": 'https://github.com/facebookresearch/ParlAI/tree/master/projects/bst',
        "description": "Pretrained polyencoder retrieval model fine-tuned on the BlendedSkillTalk dialogue task.",
        "example": "parlai interactive -mf zoo:blended_skill_talk/bst_single_task/model -t blended_skill_talk",
        "result": 'Results vary.',
        "example2": "parlai eval_model -mf zoo:blended_skill_talk/bst_single_task/model -t blended_skill_talk -dt test",
        "result2": """09:51:57 | Finished evaluating tasks ['blended_skill_talk'] using datatype test
    accuracy  bleu-4  exs    f1  gpu_mem  hits@1  hits@10  hits@100  hits@5  loss   mrr  rank   tpb
       .7920   .7785 5482 .8124    .0370   .7920    .9788         1   .9542 .8251 .8636 1.866 19.76
""",
    },
    {
        "title": "BlendedSkillTalk: ConvAI2 single-task model",
        "id": "blended_skill_talk",
        "path": "zoo:blended_skill_talk/convai2_single_task/model",
        "agent": "transformer/polyencoder",
        "task": "blended_skill_talk",
        "project": 'https://github.com/facebookresearch/ParlAI/tree/master/projects/bst',
        "description": "Pretrained polyencoder retrieval model fine-tuned on the ConvAI2 dialogue task.",
        "example": "parlai eval_model -mf zoo:blended_skill_talk/convai2_single_task/model -t blended_skill_talk -dt test",
        "result": """10:23:53 | Finished evaluating tasks ['blended_skill_talk'] using datatype test
    accuracy  bleu-4  exs    f1  gpu_mem  hits@1  hits@10  hits@100  hits@5  loss   mrr  rank   tpb
       .7678   .7553 5482 .7902   .07928   .7678    .9728         1   .9414 .9337 .8451  2.04 19.76
""",
    },
    {
        "title": "BlendedSkillTalk: EmpatheticDialogues single-task model",
        "id": "blended_skill_talk",
        "path": "zoo:blended_skill_talk/ed_single_task/model",
        "agent": "transformer/polyencoder",
        "task": "blended_skill_talk",
        "project": 'https://github.com/facebookresearch/ParlAI/tree/master/projects/bst',
        "description": "Pretrained polyencoder retrieval model fine-tuned on the EmpatheticDialogues dialogue task.",
        "example": "parlai eval_model -mf zoo:blended_skill_talk/ed_single_task/model -t blended_skill_talk -dt test",
        "result": """10:16:47 | Finished evaluating tasks ['blended_skill_talk'] using datatype test
    accuracy  bleu-4  exs    f1  gpu_mem  hits@1  hits@10  hits@100  hits@5  loss   mrr  rank   tpb
       .6895   .6774 5482 .7219   .07928   .6895    .9509         1   .9051 1.242 .7849  2.79 19.76
""",
    },
    {
        "title": "BlendedSkillTalk: Wizard of Wikipedia single-task model",
        "id": "blended_skill_talk",
        "path": "zoo:blended_skill_talk/wizard_single_task/model",
        "agent": "transformer/polyencoder",
        "task": "blended_skill_talk",
        "project": 'https://github.com/facebookresearch/ParlAI/tree/master/projects/bst',
        "description": "Pretrained polyencoder retrieval model fine-tuned on the Wizard of Wikipedia dialogue task.",
        "example": "parlai eval_model -mf zoo:blended_skill_talk/wizard_single_task/model -t blended_skill_talk -dt test",
        "result": """10:34:46 | Finished evaluating tasks ['blended_skill_talk'] using datatype test
    accuracy  bleu-4  exs    f1  gpu_mem  hits@1  hits@10  hits@100  hits@5  loss   mrr  rank   tpb
       .6742   .6616 5482 .7059   .07928   .6742    .9445         1   .8902 1.321 .7706 2.962 19.76
""",
    },
    {
        "title": "BlendedSkillTalk: MT Single-Skills model",
        "id": "blended_skill_talk",
        "path": "zoo:blended_skill_talk/multi_task/model",
        "agent": "transformer/polyencoder",
        "task": "blended_skill_talk",
        "project": 'https://github.com/facebookresearch/ParlAI/tree/master/projects/bst',
        "description": "Pretrained polyencoder retrieval model fine-tuned on the ConvAI2, EmpatheticDialogues, and Wizard of Wikipedia dialogue tasks.",
        "example": "parlai eval_model -mf zoo:blended_skill_talk/multi_task/model -t blended_skill_talk -dt test",
        "result": """10:23:35 | Finished evaluating tasks ['blended_skill_talk'] using datatype test
    accuracy  bleu-4  exs    f1  gpu_mem  hits@1  hits@10  hits@100  hits@5  loss   mrr  rank   tpb
       .8010   .7872 5482 .8204   .07928   .8010    .9779         1   .9564 .8154 .8697 1.908 19.76
""",
    },
    {
        "title": "BlendedSkillTalk: MT Single-Skills model fine-tuned on BST",
        "id": "blended_skill_talk",
        "path": "zoo:blended_skill_talk/multi_task_bst_tuned/model",
        "agent": "transformer/polyencoder",
        "task": "blended_skill_talk",
        "project": 'https://github.com/facebookresearch/ParlAI/tree/master/projects/bst',
        "description": "Pretrained polyencoder retrieval model fine-tuned on the ConvAI2, EmpatheticDialogues, and Wizard of Wikipedia dialogue tasks, and then further fine-tuned on the BlendedSkillTalk dialogue task.",
        "example": "parlai eval_model -mf zoo:blended_skill_talk/multi_task_bst_tuned/model -t blended_skill_talk -dt test",
        "result": """10:36:01 | Finished evaluating tasks ['blended_skill_talk'] using datatype test
    accuracy  bleu-4  exs    f1  gpu_mem  hits@1  hits@10  hits@100  hits@5  loss   mrr  rank   tpb
       .8378   .8230 5482 .8543   .07928   .8378    .9872         1   .9704 .5897 .8963 1.604 19.76
""",
    },
    {
        "title": "Tutorial Transformer Generator",
        "id": "tutorial_transformer_generator",
        "path": "zoo:tutorial_transformer_generator/model",
        "task": "pushshift.io",
        "description": (
            "Small (87M paramter) generative transformer, pretrained on pushshift.io Reddit."
        ),
        "example": "parlai interactive -mf zoo:tutorial_transformer_generator/model",
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
        "title": "Blender 1B distilled",
        "id": "blender",
        "path": "zoo:blender/blender_1Bdistill/model",
        "agent": "transformer/generator",
        "task": "blended_skill_talk",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/blender",
        "description": (
            "2.7B parameter generative model finetuned on blended_skill_talk tasks and then distilled to ~1.4B parameters and roughly 2x the inference speed."
        ),
        "example": (
            "python parlai/scripts/safe_interactive.py -mf zoo:blender/blender_1Bdistill/model -t blended_skill_talk"
        ),
        "result": (
            "Enter Your Message: Hi how are you?\n"
            "[TransformerGenerator]: I'm doing well. How about yourself? What do you do for a living? I'm a creative writer."
        ),
    },
    {
        "title": "Blender 400M distilled",
        "id": "blender",
        "path": "zoo:blender/blender_400Mdistill/model",
        "agent": "transformer/generator",
        "task": "blended_skill_talk",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/blender",
        "description": (
            "2.7B parameter generative model finetuned on blended_skill_talk tasks and then distilled to ~360M parameters and roughly 5x the inference speed."
        ),
        "example": (
            "python parlai/scripts/safe_interactive.py -mf zoo:blender/blender_400Mdistill/model -t blended_skill_talk"
        ),
        "result": (
            "Enter Your Message: Hi how are you?\n"
            "[TransformerGenerator]: I'm doing well. How about you? What do you like to do in your free time?"
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
            "2.7B parameter generative model pretrained on Reddit but not finetuned."
        ),
        "example": (
            "parlai train_model -t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues --multitask-weights 1,3,3,3 -veps 0.25 --attention-dropout 0.0 --batchsize 128 --model transformer/generator --embedding-size 2560 --ffn-size 10240 --variant prelayernorm --n-heads 32 --n-positions 128 --n-encoder-layers 2 --n-decoder-layers 24 --history-add-global-end-token end --delimiter '  ' --dict-tokenizer bytelevelbpe  --dropout 0.1 --fp16 True --init-model zoo:blender/reddit_3B/model --dict-file zoo:blender/reddit_3B/model.dict --label-truncate 128 --log_every_n_secs 10 -lr 7e-06 --lr-scheduler reduceonplateau --lr-scheduler-patience 3 --optimizer adam --relu-dropout 0.0 --activation gelu --model-parallel true --save-after-valid True --text-truncate 128 --truncate 128 --warmup_updates 100 --fp16-impl mem_efficient --update-freq 2 --gradient-clip 0.1 --skip-generation True -vp 10 -vmt ppl -vmm min --model-file /tmp/test_train_27B"
        ),
        "result": "Results vary.",
    },
    {
        "title": "Reddit 9.4B",
        "id": "blender",
        "path": "zoo:blender/reddit_9B/model",
        "agent": "transformer/generator",
        "task": "pushshift.io",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/blender",
        "description": (
            "9.4B parameter generative model pretrained on Reddit but not finetuned."
        ),
        "example": (
            "parlai train_model -t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues --multitask-weights 1,3,3,3 -veps 0.25 --attention-dropout 0.0 --batchsize 8 --eval-batchsize 64 --model transformer/generator --embedding-size 4096 --ffn-size 16384 --variant prelayernorm --n-heads 32 --n-positions 128 --n-encoder-layers 4 --n-decoder-layers 32 --history-add-global-end-token end --dict-tokenizer bytelevelbpe --dropout 0.1 --fp16 True --init-model zoo:blender/reddit_9B/model --dict-file zoo:blender/reddit_9B/model.dict --label-truncate 128 -lr 3e-06 -dynb full --lr-scheduler cosine --max-lr-steps 9000 --lr-scheduler-patience 3 --optimizer adam --relu-dropout 0.0 --activation gelu --model-parallel true --save-after-valid False --text-truncate 128 --truncate 128 --warmup_updates 1000 --fp16-impl mem_efficient --update-freq 4 --log-every-n-secs 30 --gradient-clip 0.1 --skip-generation True -vp 10 --max-train-time 84600 -vmt ppl -vmm min --model-file /tmp/test_train_94B"
        ),
        "result": "Results vary.",
    },
    {
        "title": "BART",
        "id": "bart",
        "path": "zoo:bart/bart_large/model",
        "agent": "bart",
        "external_website": "https://arxiv.org/abs/1910.13461",
        "task": "wikipedia_plus_toronto_books",
        "description": (
            "BART: Denoising Sequence-to-Sequence Pre-training for Natural "
            "Language Generation, Translation, and Comprehension."
        ),
        "example": (
            "parlai eval_model -mf zoo:bart/bart_large/model -t convai2 -bs 64"
        ),
        "result": (
            "Finished evaluating tasks ['convai2'] using datatype valid\n"
            "accuracy   bleu-4    exs      f1  gpu_mem    loss    ppl  token_acc   tpb\n"
            "0        .0004641   7801  .02084    .4878   5.041  154.6      .2042  1652"
        ),
        "example2": (
            "parlai train_model -m bart -mf /tmp/model_file -t convai2 -bs 24 --fp16 true -eps 1 -lr 1e-5 --optimizer adam"
        ),
        "result2": (
            "valid:\n"
            "accuracy  bleu-4  exs    f1  gpu_mem  loss    lr   ppl  token_acc  total_train_updates   tpb\n"
            ".0001282  .01229 7801 .2035    .6361 2.386 1e-05 10.87      .4741                 5478 321.3"
        ),
    },
    {
        "title": "Unlikelihood ConvAI2 context and label repetition model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/rep_convai2_ctxt_and_label/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on ConvAI2 with context and label repetition unlikelihood"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_convai2_ctxt_and_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent"
        ),
        "result": (
            "Enter Your Message: Hi.\n"
            "[RepetitionUnlikelihood]: hi , how are you doing today ?"
        ),
    },
    {
        "title": "Unlikelihood ConvAI2 context repetition model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/rep_convai2_ctxt/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on ConvAI2 with context repetition unlikelihood"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_convai2_ctxt/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent"
        ),
        "result": (
            "Enter Your Message: Hi.\n"
            "[RepetitionUnlikelihood]: hi , how are you doing today ?"
        ),
    },
    {
        "title": "Unlikelihood ConvAI2 label repetition model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/rep_convai2_label/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on ConvAI2 with label repetition unlikelihood"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_convai2_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent"
        ),
        "result": (
            "Enter Your Message: Hi.\n"
            "[RepetitionUnlikelihood]: hi , how are you doing today ?"
        ),
    },
    {
        "title": "Unlikelihood ELI5 context and label repetition model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/rep_eli5_ctxt_and_label/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "eli5",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on ELI5 with context and label repetition unlikelihood"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_eli5_ctxt_and_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent"
        ),
        "result": ("Enter Your Message: Hi.\n" "[RepetitionUnlikelihood]: hi ."),
    },
    {
        "title": "Unlikelihood ELI5 context repetition model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/rep_celi5_ctxt/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "eli5",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on ELI5 with context repetition unlikelihood"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_eli5_ctxt/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent"
        ),
        "result": ("Enter Your Message: Hi.\n" "[RepetitionUnlikelihood]: hi ."),
    },
    {
        "title": "Unlikelihood ELI5 label repetition model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/rep_eli5_label/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "eli5",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on ELI5 with label repetition unlikelihood"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_eli5_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent"
        ),
        "result": ("Enter Your Message: Hi.\n" "[RepetitionUnlikelihood]: hi ."),
    },
    {
        "title": "Unlikelihood Wizard of Wikipedia context and label repetition model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/rep_wiki_ctxt_and_label/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "wizard_of_wikipedia:GeneratorTeacher",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on Wizard of Wikipedia with context and label repetition unlikelihood"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_wiki_ctxt_and_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent"
        ),
        "result": ("Enter Your Message: Hi.\n" "[RepetitionUnlikelihood]: hi ."),
    },
    {
        "title": "Unlikelihood Wizard of Wikipedia context repetition model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/rep_wiki_ctxt/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "wizard_of_wikipedia:GeneratorTeacher",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on Wizard of Wikipedia with context repetition unlikelihood"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_wiki_ctxt/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent"
        ),
        "result": ("Enter Your Message: Hi.\n" "[RepetitionUnlikelihood]: hi ."),
    },
    {
        "title": "Unlikelihood Wizard of Wikipedia label repetition model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/rep_wiki_label/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "wizard_of_wikipedia:GeneratorTeacher",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on Wizard of Wikipedia with label repetition unlikelihood"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_wiki_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent"
        ),
        "result": ("Enter Your Message: Hi.\n" "[RepetitionUnlikelihood]: hi ."),
    },
    {
        "title": "Unlikelihood vocab alpha 1e0 model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/vocab_alpha1e0/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on convai2 with vocab unlikelihood, alpha value 1e0"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/vocab_alpha1e0/model -m projects.dialogue_unlikelihood.agents:TransformerSequenceVocabUnlikelihoodAgent"
        ),
        "result": (
            "Enter Your Message: Hi.\n"
            "[TransformerSequenceVocabUnlikelihood]: hi there ! how are you ?"
        ),
    },
    {
        "title": "Unlikelihood vocab alpha 1e1 model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/vocab_alpha1e1/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on convai2 with vocab unlikelihood, alpha value 1e1"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/vocab_alpha1e1/model -m projects.dialogue_unlikelihood.agents:TransformerSequenceVocabUnlikelihoodAgent"
        ),
        "result": (
            "Enter Your Message: Hi.\n"
            "[TransformerSequenceVocabUnlikelihood]: hi how are you today"
        ),
    },
    {
        "title": "Unlikelihood vocab alpha 1e2 model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/vocab_alpha1e2/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on convai2 with vocab unlikelihood, alpha value 1e2"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/vocab_alpha1e2/model -m projects.dialogue_unlikelihood.agents:TransformerSequenceVocabUnlikelihoodAgent"
        ),
        "result": (
            "Enter Your Message: Hi.\n"
            "[TransformerSequenceVocabUnlikelihood]: hello , how are you ?"
        ),
    },
    {
        "title": "Unlikelihood vocab alpha 1e3 model",
        "id": "dialogue_unlikelihood",
        "path": "zoo:dialogue_unlikelihood/vocab_alpha1e3/model",
        "agent": "projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent",
        "task": "convai2",
        "project": "https://github.com/facebookresearch/ParlAI/tree/master/projects/dialogue_unlikelihood",
        "description": (
            "Dialogue model finetuned on convai2 with vocab unlikelihood, alpha value 1e3"
        ),
        "example": (
            "python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/vocab_alpha1e3/model -m projects.dialogue_unlikelihood.agents:TransformerSequenceVocabUnlikelihoodAgent"
        ),
        "result": (
            "Enter Your Message: Hi.\n"
            "[TransformerSequenceVocabUnlikelihood]: hi there !"
        ),
    },
    {
        "title": "Style-controlled generation: C75-D+ generator",
        "id": "style_gen",
        "path": "zoo:style_gen/c75_labeled_dialogue_generator/model",
        "agent": "projects.style_gen.style_gen:StyleGenAgent",
        "task": "style_gen:LabeledBlendedSkillTalk",
        "project": 'https://github.com/facebookresearch/ParlAI/tree/master/projects/style_gen',
        "description": "Generator trained on dialogue datasets, with 75% of train examples appended with Image-Chat personality labels",
        "example": "parlai eval_model --datatype test --model projects.style_gen.style_gen:StyleGenAgent --model-file zoo:style_gen/c75_labeled_dialogue_generator/model --skip-generation True --task style_gen:LabeledBlendedSkillTalk --use-style-frac 1.00",
        "result": """16:56:52 | Finished evaluating tasks ['style_gen:LabeledBlendedSkillTalk'] using datatype test
    ctpb  ctps  exps  exs  gpu_mem  loss  ltpb  ltps   ppl  token_acc   tpb  tps
     120  1855 15.46 5482    .1635 2.248 19.94 308.2 9.468      .4872 139.9 2163""",
    },
    {
        "title": "Style-controlled generation: previous and current utterance classifier",
        "id": "style_gen",
        "path": "zoo:style_gen/prev_curr_classifier/model",
        "agent": "projects.style_gen.classifier:ClassifierAgent",
        "task": "style_gen:LabeledBlendedSkillTalk",
        "project": 'https://github.com/facebookresearch/ParlAI/tree/master/projects/style_gen',
        "description": "Classifier trained on Image-Chat turns 2 and 3 to classify the personality of an example given that utterance and the previous utterance.",
        "example": "parlai eval_model --task style_gen:PrevCurrUttStyle --wrapper-task style_gen:LabeledBlendedSkillTalk --model-file zoo:style_gen/prev_curr_classifier/model --model projects.style_gen.classifier:ClassifierAgent --classes-from-file image_chat_personalities_file",
        "result": """18:42:33 | Finished evaluating tasks ['style_gen:PrevCurrUttStyle'] using datatype valid
    accuracy  bleu-4  ctpb  ctps  exps  exs    f1  gpu_mem  loss    lr  ltpb  ltps   tpb   tps
       .9973  .01745 38.08 604.1 15.86 5651 .9973    .1622 2.129 5e-10 5.633 89.36 43.71 693.4""",
    },
    {
        "title": "Faster-R-CNN Detectron Features",
        "id": "detectron",
        "path": "zoo:detectron/detectron_model.pth",
        "agent": "n/a",
        "task": "n/a",
        "project": 'https://github.com/facebookresearch/vilbert-multi-task',
        "description": "Detectron Model for extracting 100x2048d object detection features. Model is from linked project website",
        "example": "parlai display_data -t flickr30k --image-mode faster_r_cnn_152_32x8d -n 1 -v",
        "result": """
        [id]: flickr30k
        [image_id]: 1000092795
        [ image ]: tensor([[ 0.0000,  7.1329,  0.0000,  ...,  0.0000,  0.0000,  5.3357],
                [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.4687,  ...,  0.0000,  0.0000,  0.0000],
                ...,
                [ 0.0000,  3.0936,  3.4888,  ...,  0.0000,  0.0000,  0.0000],
                [ 3.8596,  0.0000,  0.0000,  ...,  0.0000,  4.3454,  0.0000],
                [ 0.0000,  7.9822,  1.0979,  ...,  3.5514,  0.0000, 15.3559]])
        [labels]: Two young guys with shaggy hair look at their hands while hanging out in the yard.|Two young, White males are outside near many bushes.|Two men in green shirts are standing in a yard.|A man in a blue shirt standing in a garden.|Two friends enjoy time spent together.
        """,
    },
    {
        "title": "Multi-Modal BlenderBot (MMB DegenPos)",
        "id": "multimodal_blenderbot",
        "path": 'n/a',
        "agent": "projects.multimodal_blenderbot.agents:BiasAgent",
        "task": "blended_skill_talk",
        "project": 'https://github.com/facebookresearch/ParlAI/tree/master/projects/multimodal_blenderbot',
        "description": "Model trained to talk about both images and general chitchat, trained with a degendering teacher and with 75% of Image-Chat styles replaced by a generic polarity string",
        "example": "python parlai/scripts/safe_interactive.py -t blended_skill_talk -mf ${FINETUNED_MODEL_PATH} --model projects.multimodal_blenderbot.agents:BiasAgent --delimiter $'\n' --beam-block-ngram 3 --beam-context-block-ngram 3 --beam-min-length 20 --beam-size 10 --inference beam --model-parallel False",
        "result": "(results will vary)",
    },
    {
        "title": "Transformer Classifier Multi-turn Dialogue Safety Model",
        "id": "bot_adversarial_dialogue",
        "path": "zoo:bot_adversarial_dialogue/multi_turn_v0/model",
        "agent": "transformer/classifier",
        "task": "bot_adversarial_dialogue",
        "project": "",
        "description": (
            "Classifier trained on the filtered multi-turn bot adversarial dialogues in addition to both dialogue_safety single-turn standard and adversarial safety tasks and Wikipedia Toxic Comments."
        ),
        "example": (
            "parlai eval_model -t bot_adversarial_dialogue:bad_num_turns=4 -dt test -mf zoo:bot_adversarial_dialogue/multi_turn_v0/model -bs 128"
        ),
        "result": (
            "{'exs': 2598, 'accuracy': 0.8414, 'f1': 0.8414, 'loss': 0.5153, 'bleu-4': 8.414e-10, 'class___notok___recall': 0.8093, 'class___notok___prec': 0.7671, 'class___notok___f1': 0.7876, 'class___ok___recall': 0.8597, 'class___ok___prec': 0.8876, 'class___ok___f1': 0.8735, 'weighted_f1': 0.8423}"
        ),
    },
    {
        "title": "Transformer Classifier Multi-turn Dialogue Safety Model",
        "id": "bot_adversarial_dialogue",
        "path": "zoo:bot_adversarial_dialogue/multi_turn/model",
        "agent": "transformer/classifier",
        "task": "bot_adversarial_dialogue",
        "project": "",
        "description": (
            "Classifier trained on the truncated multi-turn bot adversarial dialogues in addition to both dialogue_safety single-turn standard and adversarial safety tasks and Wikipedia Toxic Comments."
        ),
        "example": (
            "parlai eval_model -t bot_adversarial_dialogue:bad_num_turns=4 -dt test -mf zoo:bot_adversarial_dialogue/multi_turn/model -bs 128"
        ),
        "result": (
            "{'exs': 2598, 'accuracy': 0.8507, 'f1': 0.8507, 'loss': 0.3878, 'bleu-4': 8.507e-10, 'class___notok___recall': 0.8633, 'class___notok___prec': 0.7588, 'class___notok___f1': 0.8077, 'class___ok___recall': 0.8434, 'class___ok___prec': 0.9154, 'class___ok___f1': 0.8779, 'weighted_f1': 0.8524}"
        ),
    },
    {
        "title": "Transformer Classifier Sensitive Topics Detection",
        "id": "sensitive_topics_classifier",
        "path": "zoo:sensitive_topics_classifier/model",
        "agent": "transformer/classifier",
        "task": "sensitive_topics_evaluation",
        "project": "",
        "description": (
            "Pretrained Transformer-based classifier for classification of sensitive topics."
        ),
        "example": (
            "parlai eval_model -mf zoo:sensitive_topics_classifier/model -t sensitive_topics_evaluation -dt valid -bs 16"
        ),
        "result": (
            "{'exs': 1811, 'accuracy': 0.7332965212589729, 'f1': 0.7332965212589729, 'bleu-4': 7.332965212589829e-10, 'loss': 0.8489562001990587, 'class_none_prec': 0.0, 'class_none_recall': 0.0, 'class_none_f1': 0.0, 'class_politics_prec': 0.8762376237623762, 'class_politics_recall': 0.885, 'class_politics_f1': 0.8805970149253731, 'class_religion_prec': 0.8829568788501027, 'class_religion_recall': 0.8669354838709677, 'class_religion_f1': 0.8748728382502543, 'class_medical_prec': 0.8237704918032787, 'class_medical_recall': 0.7077464788732394, 'class_medical_f1': 0.7613636363636364, 'class_nsfw_prec': 0.7769784172661871, 'class_nsfw_recall': 0.32142857142857145, 'class_nsfw_f1': 0.45473684210526316, 'class_drugs_prec': 0.8901515151515151, 'class_drugs_recall': 0.7966101694915254, 'class_drugs_f1': 0.8407871198568873, 'weighted_f1': 0.774835331736443}"
        ),
    },
    {
        "title": "MDGender Bert Ranker Classifier",
        "id": "md_gender",
        "path": "zoo:md_gender/model",
        "agent": "projects.md_gender.bert_ranker_classifier.agents:BertRankerClassifierAgent",
        "task": "md_gender",
        "project": "",
        "description": (
            "Ranker/classifier agent trained to predict gender bias along several axes using the MD Gender task."
        ),
        "example": (
            "parlai interactive -t md_gender -m projects.md_gender.bert_ranker_classifier.agents:BertRankerClassifierAgent -mf zoo:md_gender/model -ecands inline -cands inline --interactive_mode False --data-parallel False"
        ),
        "result": (
            "Enter Your Message: Hi my name is Emily!\n"
            "[MDGender Classifier]: SELF: female"
        ),
    },
    {
        "title": "Bart FiD DPR Model",
        "id": "hallucination",
        "path": "zoo:hallucination/bart_fid_dpr/model",
        "agent": "fid",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "FiD model trained with a DPR Retriever and a BART-Large backbone seq2seq generator."
        ),
        "example": ("parlai interactive -mf zoo:hallucination/bart_fid_dpr/model"),
        "result": (
            "Enter Your Message: Hey! What's your favorite Elvis song?\n"
            "[Fid]: I love Elvis Presley, he is my favorite singer and songwriter. He was born in Memphis, TN and died in Memphis Tennessee."
        ),
        "example2": (
            "parlai eval_model -mf zoo:hallucination/bart_fid_dpr/model -t wizard_of_wikipedia"
        ),
        "result2": (
            "Finished evaluating tasks ['wizard_of_wikipedia:WizardDialogKnowledge:random_split'] using datatype valid \n"
            "accuracy  bleu-1  bleu-2  bleu-3  bleu-4  clen  ctpb  ctps  ctrunc  ctrunclen  exps  exs    f1  gpu_mem  interdistinct-1  \\\n"
            ".0002539   .1772  .08068  .04245  .02717 76.62 78.62 55.81       0          0 .7098 3939 .2097   .04684            .1024\n"
            "interdistinct-2  intradistinct-1  intradistinct-2  knowledge_f1  llen  loss  ltpb  ltps  ltrunc  ltrunclen   ppl  \\\n"
            ".4421            .8833            .9822         .2172 23.61 2.687 23.61 16.76       0          0 14.68\n"
            "rare_word_f1  rouge_1  rouge_2  rouge_L  token_acc  token_em   tpb   tps\n"
            ".1547    .2647   .08819    .2244      .4348         0 102.2 72.57\n"
        ),
    },
    {
        "title": "Bart FiD Rag DPR-Poly Model",
        "id": "hallucination",
        "path": "zoo:hallucination/bart_fid_rag_dpr_poly/model",
        "agent": "fid",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "FiD model trained with a RAG-DPR-Poly Retriever (DPR-Poly retrieval, trained in a RAG Setup) "
            "and a BART-Large backbone seq2seq generator."
        ),
        "example": (
            "parlai interactive -mf zoo:hallucination/bart_fid_rag_dpr_poly/model"
        ),
        "result": (
            "Enter Your Message: Hey! What's your favorite Elvis song?\n"
            "[Fid]: I love Elvis Presley! He is my favorite singer, songwriter, actor, and producer."
        ),
        "example2": (
            "parlai eval_model -mf zoo:hallucination/bart_fid_rag_dpr_poly/model -t wizard_of_wikipedia --num-examples 100"
        ),
        "result2": (
            "Finished evaluating tasks ['wizard_of_wikipedia:WizardDialogKnowledge:random_split'] using datatype valid\n"
            "accuracy  bleu-1  bleu-2  bleu-3  bleu-4  clen  ctpb  ctps  ctrunc  ctrunclen  exps  exs    f1  gpu_mem  interdistinct-1  \\n"
            ".0002539   .1821  .09088  .05551  .03939 76.62 78.62 53.79       0          0 .6842 3939 .2154   .05641            .1056\n"
            "interdistinct-2  intradistinct-1  intradistinct-2  knowledge_f1  llen  loss  ltpb  ltps  ltrunc  ltrunclen  ppl  \\n"
            ".4434            .8964            .9864         .2756 23.61 2.587 23.61 16.16       0          0 13.3\n"
            "rare_word_f1  rouge_1  rouge_2  rouge_L  token_acc  token_em   tpb   tps\n"
            ".1686    .2740    .1040    .2351      .4523  .0002539 102.2 69.95"
        ),
    },
    {
        "title": "Bart FiD Rag Model",
        "id": "hallucination",
        "path": "zoo:hallucination/bart_fid_rag/model",
        "agent": "fid",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "FiD model trained with a RAG Retriever (DPR retrieval) "
            "and a BART-Large backbone seq2seq generator."
        ),
        "example": ("parlai interactive -mf zoo:hallucination/bart_fid_rag/model"),
        "result": (
            "Enter Your Message: Hey! What's your favorite Elvis song?\n"
            "[Fid]: I love Elvis Presley.  I love his music.  My favorite song is \"My Way.\""
        ),
        "example2": (
            "parlai eval_model -mf zoo:hallucination/bart_fid_rag/model -t wizard_of_wikipedia --num-examples 100"
        ),
        "result2": (
            "Finished evaluating tasks ['wizard_of_wikipedia:WizardDialogKnowledge:random_split'] using datatype valid\n"
            "accuracy  bleu-4  clen  ctpb  ctps  ctrunc  ctrunclen  exps  exs    f1  gpu_mem  knowledge_f1  llen  loss    lr  ltpb  ltps  ltrunc  ltrunclen   ppl  rare_word_f1  token_acc  \\\n"
            "       0  .04021 84.71 86.71 26.11       0          0 .3011  100 .2324   .03073         .2381  24.6 2.676 1e-05  24.6 7.407       0          0 14.53         .1678      .4301\n"
            "token_em   tpb   tps \n"
            "       0 111.3 33.51"
        ),
    },
    {
        "title": "Bart RAG DPR-Poly Model",
        "id": "hallucination",
        "path": "zoo:hallucination/bart_rag_dpr_poly/model",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "RAG-Token model trained with a DPR Retriever "
            "and a BART-Large backbone seq2seq generator."
        ),
        "example": ("parlai interactive -mf zoo:hallucination/bart_rag_dpr_poly/model"),
        "result": (
            "Enter Your Message: Hey! What's your favorite Elvis song?\n"
            "[RAG]: I love Elvis Presley. He is my favorite singer of all time. Do you have a favorite song?"
        ),
        "example2": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_dpr_poly/model -t wizard_of_wikipedia --num-examples 100"
        ),
        "result2": (
            "Finished evaluating tasks ['wizard_of_wikipedia:WizardDialogKnowledge:random_split'] using datatype valid\n"
            "accuracy  bleu-1  bleu-2  bleu-3  bleu-4  clen  ctpb  ctps  ctrunc  ctrunclen  exps  exs    f1  gpu_mem  interdistinct-1  \\n"
            ".0005077   .1884  .09254  .05288  .03484 76.62 78.62 59.44       0          0 .7560 3939 .2230   .08013            .1053\n"
            "interdistinct-2  intradistinct-1  intradistinct-2  knowledge_f1  llen  loss  ltpb  ltps  ltrunc  ltrunclen   ppl  \\n"
            ".4343            .9128            .9898         .2424 23.61  2.53 23.61 17.85       0          0 12.56\n"
            "rare_word_f1  rouge_1  rouge_2  rouge_L  token_acc  token_em   tpb   tps\n"
            ".1734    .2679   .09963    .2309      .4664  .0002539 102.2 77.29"
        ),
    },
    {
        "title": "Bart RAG DPR Sequence Model",
        "id": "hallucination",
        "path": "zoo:hallucination/bart_rag_sequence/model",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "RAG-Sequence model trained with a DPR Retriever "
            "and a BART-Large backbone seq2seq generator."
        ),
        "example": ("parlai interactive -mf zoo:hallucination/bart_rag_sequence/model"),
        "result": (
            "Enter Your Message: Hey! What's your favorite Elvis song?\n"
            "[RAG]: My favorite Elvis Presley song is \"Stuck on You\". Do you have a favorite Elvis song?"
        ),
        "example2": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_sequence/model -t wizard_of_wikipedia --num-examples 100"
        ),
        "result2": (
            "Finished evaluating tasks ['wizard_of_wikipedia:WizardDialogKnowledge:random_split'] using datatype valid\n"
            "accuracy  bleu-1  bleu-2  bleu-3  bleu-4  clen  ctpb  ctps  ctrunc  ctrunclen  exps  exs    f1  gpu_mem  interdistinct-1  \\n"
            ".0007616   .1771  .08545  .04963  .03463 76.62 78.62 43.95       0          0 .5591 3939 .2102   .05834            .1125\n"
            "interdistinct-2  intradistinct-1  intradistinct-2  knowledge_f1  llen  loss  ltpb  ltps  ltrunc  ltrunclen   ppl  \\n"
            ".4500            .9186            .9928         .2566 23.61 2.484 23.61  13.2       0          0 11.99\n"
            "rare_word_f1  rouge_1  rouge_2  rouge_L  token_acc  token_em   tpb   tps\n"
            ".1630    .2587   .09407    .2234      .4256         0 102.2 57.15"
        ),
    },
    {
        "title": "Bart RAG DPR Token Model",
        "id": "hallucination",
        "path": "zoo:hallucination/bart_rag_token/model",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "RAG-Token model trained with a DPR Retriever "
            "and a BART-Large backbone seq2seq generator."
        ),
        "example": ("parlai interactive -mf zoo:hallucination/bart_rag_token/model"),
        "result": (
            "Enter Your Message: Hey! What's your favorite Elvis song?\n"
            "[RAG]: I love Elvis Presley.  He is my favorite singer of all time.  Do you have a favorite Elvis song?"
        ),
        "example2": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_token/model -t wizard_of_wikipedia --num-examples 100"
        ),
        "result2": (
            "Finished evaluating tasks ['wizard_of_wikipedia:WizardDialogKnowledge:random_split'] using datatype valid\n"
            "accuracy  bleu-1  bleu-2  bleu-3  bleu-4  clen  ctpb  ctps  ctrunc  ctrunclen  exps  exs    f1  gpu_mem  interdistinct-1  \\n"
            ".0002539   .1859  .08962  .04951  .03237 76.62 78.62 55.44       0          0 .7052 3939 .2200   .06284            .1012\n"
            "interdistinct-2  intradistinct-1  intradistinct-2  knowledge_f1  llen  loss  ltpb  ltps  ltrunc  ltrunclen   ppl  \\n"
            ".4102            .9101            .9882         .2284 23.61 2.529 23.61 16.65       0          0 12.54\n"
            "rare_word_f1  rouge_1  rouge_2  rouge_L  token_acc  token_em   tpb   tps\n"
            ".1693    .2625   .09543    .2265      .4627         0 102.2 72.09"
        ),
    },
    {
        "title": "Bart RAG DPR Turn Doc-Then-Turn Model",
        "id": "hallucination",
        "path": "zoo:hallucination/bart_rag_turn_dtt/model",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "RAG-Turn Doc-Then-Turn model trained with a DPR Retriever "
            "and a BART-Large backbone seq2seq generator."
        ),
        "example": ("parlai interactive -mf zoo:hallucination/bart_rag_turn_dtt/model"),
        "result": (
            "Enter Your Message: Hey! What's your favorite Elvis song?\n"
            "[RAG]: I love Elvis Presley. He was a great singer and songwriter. Do you have a favorite song?"
        ),
        "example2": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_turn_dtt/model -t wizard_of_wikipedia --num-examples 100"
        ),
        "result2": (
            "Finished evaluating tasks ['wizard_of_wikipedia:WizardDialogKnowledge:random_split'] using datatype valid\n"
            "accuracy  bleu-1  bleu-2  bleu-3  bleu-4  clen  ctpb  ctps  ctrunc  ctrunclen  exps  exs    f1  gpu_mem  interdistinct-1  \\n"
            ".0005077   .1863   .0911  .05345  .03651 76.62 78.62 54.72       0          0 .6960 3939 .2202   .07843            .1032\n"
            "interdistinct-2  intradistinct-1  intradistinct-2  knowledge_f1  llen  loss  ltpb  ltps  ltrunc  ltrunclen   ppl  \\n"
            ".4188            .9154            .9915         .2476 23.61 2.527 23.61 16.43       0          0 12.51\n"
            "rare_word_f1  rouge_1  rouge_2  rouge_L  token_acc  token_em   tpb   tps\n"
            ".1735    .2648   .09857    .2290      .4599  .0002539 102.2 71.15"
        ),
    },
    {
        "title": "Bart RAG DPR Turn Doc-Only Model",
        "id": "hallucination",
        "path": "zoo:hallucination/bart_rag_turn_do/model",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "RAG-Turn Doc-Then-Turn model trained with a DPR Retriever "
            "and a BART-Large backbone seq2seq generator."
        ),
        "example": ("parlai interactive -mf zoo:hallucination/bart_rag_turn_do/model"),
        "result": (
            "Enter Your Message: Hey! What's your favorite Elvis song?\n"
            "[RAG]: Hey! I love Elvis Presley. He is one of my favorite musicians. I love his voice."
        ),
        "example2": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_turn_do/model -t wizard_of_wikipedia --num-examples 100"
        ),
        "result2": (
            "Finished evaluating tasks ['wizard_of_wikipedia:WizardDialogKnowledge:random_split'] using datatype valid\n"
            "accuracy  bleu-1  bleu-2  bleu-3  bleu-4  clen  ctpb  ctps  ctrunc  ctrunclen  exps  exs    f1  gpu_mem  interdistinct-1  \\n"
            ".0002539   .1844  .08733  .04707  .03028 76.62 78.62 51.39       0          0 .6537 3939 .2179   .09101            .1009\n"
            "interdistinct-2  intradistinct-1  intradistinct-2  knowledge_f1  llen  loss  ltpb  ltps  ltrunc  ltrunclen   ppl  \\n"
            ".4128            .9113            .9890         .2253 23.61 2.658 23.61 15.43       0          0 14.26\n"
            "rare_word_f1  rouge_1  rouge_2  rouge_L  token_acc  token_em   tpb   tps\n"
            ".1656    .2582   .09185    .2228      .4461         0 102.2 66.83"
        ),
    },
    {
        "title": "Dropout Poly-Encoder",
        "id": "hallucination",
        "path": "zoo:hallucination/dropout_poly/model",
        "agent": "transformer/dropout_poly",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "A Dropout Poly-encoder trained on the WoW knowledge selection task"
        ),
        "example": (
            "parlai tm -m rag --rag-retriever-type polyfaiss --query-model dropout_poly --poly-faiss-model-file zoo:hallucination/dropout_poly/model"
        ),
        "result": ("TODO"),
    },
    {
        "title": "Multiset DPR Model",
        "id": "hallucination",
        "path": "zoo:hallucination/multiset_dpr/hf_bert_base.cp",
        "agent": "rag/dpr_biencoder",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": ("A DPR model pre-trained on Natural Questions and Trivia QA"),
        "example": (
            "parlai tm -m rag --rag-retriever-type dpr --dpr-model-file zoo:hallucination/multiset_dpr/hf_bert_base.cp"
        ),
        "result": ("TODO"),
    },
    {
        "title": "Wikipedia Compressed FAISS Index",
        "id": "hallucination",
        "path": "zoo:hallucination/wiki_index_compressed/compressed_pq",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "A compressed FAISS Index with embeddings of Wikipedia passages, generated by the Multiset DPR zoo model."
        ),
        "example": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_token/model --indexer-type compressed --path-to-index zoo:hallucination/wiki_index_compressed/compressed_pq -ne 100"
        ),
        "result": ("TODO"),
    },
    {
        "title": "Wikipedia Exact FAISS Index",
        "id": "hallucination",
        "path": "zoo:hallucination/wiki_index_exact/exact",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "A FAISS Index with embeddings of Wikipedia passages, generated by the Multiset DPR zoo model."
        ),
        "example": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_token/model --indexer-type exact --path-to-index zoo:hallucination/wiki_index_exact/exact -ne 100"
        ),
        "result": ("TODO"),
    },
    {
        "title": "Wikipedia Passages",
        "id": "hallucination",
        "path": "zoo:hallucination/wiki_passages/psgs_w100.tsv",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "A dump of Wikipedia articles split into 21m 100 word chunks "
            "from DPR (https://github.com/facebookresearch/DPR)"
        ),
        "example": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_token/model --indexer-type exact --path-to-index zoo:hallucination/wiki_index_compressed/compressed --path-to-dpr-passages zoo:hallucination/wiki_passages/psgs_w100.tsv -ne 100"
        ),
        "result": ("TODO"),
    },
    {
        "title": "WoW Passages",
        "id": "hallucination",
        "path": "zoo:hallucination/wow_passages/wow_articles.paragraphs.tsv",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "A dump of Wikipedia articles split into ~3k paragraphs, comprising the subset of "
            "Wikipedia present in the WoW dataset."
        ),
        "example": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_token/model --indexer-type exact --path-to-index zoo:hallucination/wow_passages/exact --path-to-dpr-passages zoo:hallucination/wow_passages/wow_articles.paragraphs.tsv -ne 100"
        ),
        "result": ("TODO"),
    },
    {
        "title": "WoW Passages Compressed Index",
        "id": "hallucination",
        "path": "zoo:hallucination/wow_passages/compressed",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "A compressed FAISS Index of embeddings from ~3k paragraphs from "
            "Wikipedia present in the WoW dataset."
        ),
        "example": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_token/model --indexer-type compressed --path-to-index zoo:hallucination/wow_passages/compressed --path-to-dpr-passages zoo:hallucination/wow_passages/wow_articles.paragraphs.tsv -ne 100"
        ),
        "result": ("TODO"),
    },
    {
        "title": "WoW Passages Exact Index",
        "id": "hallucination",
        "path": "zoo:hallucination/wow_passages/exact",
        "agent": "rag",
        "task": "wizard_of_wikipedia",
        "project": "https://parl.ai/projects/hallucination/",
        "description": (
            "A FAISS Index of embeddings from ~3k paragraphs from "
            "Wikipedia present in the WoW dataset."
        ),
        "example": (
            "parlai eval_model -mf zoo:hallucination/bart_rag_token/model --indexer-type exact --path-to-index zoo:hallucination/wow_passages/exact --path-to-dpr-passages zoo:hallucination/wow_passages/wow_articles.paragraphs.tsv -ne 100"
        ),
        "result": ("TODO"),
    },
    {
        "title": "BlenderBot2 Query Generator",
        "id": "blenderbot2",
        "path": "zoo:blenderbot2/query_generator/model",
        "agent": "bart",
        "task": "wizard_of_internet:SearchQueryTeacher",
        "project": "https://parl.ai/projects/blenderbot2/",
        "description": (
            "The query generator for BlenderBot2. Either generates a search query "
            "or a phrase indicating access to the long-term memory"
        ),
        "example": ("parlai interactive -mf zoo:blenderbot2/query_generator/model"),
        "result": (
            "Enter Your Message: my favorite tv show is wandavision\n"
            "[Bart]:  wandavision"
        ),
    },
    {
        "title": "BlenderBot2 Memory Decoder",
        "id": "blenderbot2",
        "path": "zoo:blenderbot2/memory_decoder/model",
        "agent": "bart",
        "task": "multi",
        "project": "https://parl.ai/projects/blenderbot2/",
        "description": (
            "The memory decoder for BlenderBot2. Either generates a memory "
            "to write or a token indicating no memory can be written."
        ),
        "example": ("parlai interactive -mf zoo:blenderbot2/memory_decoder/model"),
        "result": (
            "Enter Your Message: i love reading; harry potter was such a good series\n"
            "[Bart]: I love reading. I like the Harry Potter series."
        ),
    },
    {
        "title": "BlenderBot2 3B",
        "id": "blenderbot2",
        "path": "zoo:blenderbot2/blenderbot2_3B/model",
        "agent": "projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent",
        "task": "wizard_of_internet",
        "project": "https://parl.ai/projects/blenderbot2/",
        "description": ("BlenderBot2 3B Model. See project for details."),
        "example": (
            "parlai interactive -mf zoo:blenderbot2/blenderbot2_3B/model --search-server relevant_search_server"
        ),
        "result": (
            "Enter Your Message: my favorite tv show is wandavision\n"
            "[BlenderBot2Fid]: Who is your favorite character in WandaVision? Mine is Elizabeth Olsen."
        ),
    },
    {
        "title": "BlenderBot2 400M",
        "id": "blenderbot2",
        "path": "zoo:blenderbot2/blenderbot2_400M/model",
        "agent": "projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent",
        "task": "wizard_of_internet",
        "project": "https://parl.ai/projects/blenderbot2/",
        "description": ("BlenderBot2 400M Model. See project for details."),
        "example": (
            "parlai interactive -mf zoo:blenderbot2/blenderbot2_400M/model --search-server relevant_search_server"
        ),
        "result": (
            "Enter Your Message: my favorite food is chicken parmesan, do you have a good recipe?\n"
            "[BlenderBot2Fid]: I don't have a recipe, but I do know how to make it at home. It's easy to make."
        ),
    },
    {
        "title": "Bart Base Wizard of Internet",
        "id": "sea",
        "path": "zoo:sea/bart_base/model",
        "agent": "bart",
        "task": "wizard_of_internet",
        "project": "https://parl.ai/projects/sea/",
        "description": ("BART-Large 400m model trained on Wizard of Internet."),
        "example": ("parlai interactive -mf zoo:sea/bart_base/model"),
        "result": (
            "Enter Your Message: Do you know about the world cup 2022?\n"
            "[Bart]: I heard that the fifa games are coming back."
        ),
        "example2": (
            "parlai eval_model -mf zoo:sea/bart_base/model -t wizard_of_internet --num-examples 100"
        ),
        "result2": (
            " clen  ctpb  ctps  ctrunc  ctrunclen  exps  exs  gpu_mem  llen  loss    lr  ltpb  ltps  ltrunc  ltrunclen   ppl  token_acc  token_em   tpb  tps\n"
            "96.48 98.48  2037       0          0 20.68  100    .1199 17.22 2.851 5e-10 17.22 356.2       0          0 17.31      .4187         0 115.7 2393"
        ),
    },
    {
        "title": "Serarch Query Generator Wizard of Internet",
        "id": "sea",
        "path": "zoo:sea/bart_sq_gen/model",
        "agent": "bart",
        "task": "wizard_of_internet",
        "project": "https://parl.ai/projects/sea/",
        "description": ("BART-Large 400m model for generating search queries."),
        "example": ("parlai interactive -mf zoo:sea/bart_sq_gen/model"),
        "result": (
            "Enter Your Message: I am looking for a good vacation spot in NY.\n"
            "[Bart]: vacation spots in ny."
        ),
    },
    {
        "title": "WizInt FiD Search Query Search Engine",
        "id": "sea",
        "path": "zoo:sea/bart_fid_sqse/model",
        "agent": "bart",
        "task": "wizard_of_internet",
        "project": "https://parl.ai/projects/sea/",
        "description": (
            "FiD model with BART-Large 400m generation model. "
            "The model first uses a search query generator model to create a search query. "
            "It forwards that search query to a search engine API to retrive documents. "
            "It uses FiD to generate a response, using the latter documents."
        ),
        "example": (
            "parlai interactive -mf zoo:sea/bart_fid_sqse/model \\ \n"
            "--search-query-generator-model-file zoo:sea/bart_fid_sqse/model \\ \n"
            "--search-server <your search server API address>"
        ),
        "result": (
            "Enter Your Message: Have you seen the new James bond movie?\n"
            "[SearchEngineFiD]: I have not seen the new James Bond movie."
        ),
    },
    {
        "title": "MSC2.7B 1024",
        "id": "msc",
        "path": "zoo:msc/msc3B_1024/model",
        "agent": "projects.msc.agents.long_tga:TransformerVariantAgent",
        "task": "msc",
        "project": "https://parl.ai/projects/msc/",
        "description": ("MSC 2.7B Model with truncate 1024. See project for details."),
        "example": ("parlai interactive -mf zoo:msc/msc3B_1024/model"),
        "result": (
            "Enter Your Message: your persona:I have a job. I have 3 sisters. I am going to Hawaii next week.\npartner's persona: I can speak 3 languages.\nHave you made all the travel plans to Hawaii?"
            "[TransformerVariant]: Yes, I have. I'm so excited. I can't wait to see my sisters and my mom."
        ),
    },
    {
        "title": "BlenderBot2.7B 1024",
        "id": "msc",
        "path": "zoo:msc/blender3B_1024/model",
        "agent": "projects.msc.agents.long_tga:TransformerVariantAgent",
        "task": "msc",
        "project": "https://parl.ai/projects/msc/",
        "description": (
            "BlenderBot 2.7B Model with truncate 1024. See project for details."
        ),
        "example": ("parlai interactive -mf zoo:msc/blender3B_1024/model"),
        "result": (
            "Enter Your Message: your persona:I have a job. I have 3 sisters. I am going to Hawaii next week.\npartner's persona: I can speak 3 languages.\nHave you made all the travel plans to Hawaii?"
            "[MemoryLongRag]: Yes, I have been planning this trip for a long time. I can't wait to go."
        ),
    },
    {
        "title": "SUMMSC-RAG 2.7B",
        "id": "msc",
        "path": "zoo:msc/summsc_rag3B/model",
        "agent": "projects.msc.agents.memory_agent:MemoryLongRagAgent",
        "task": "msc",
        "project": "https://parl.ai/projects/msc/",
        "description": (
            "SUMMSC 2.7B RAG Model with truncate 1024. See project for details."
        ),
        "example": ("parlai interactive -mf zoo:msc/summsc_rag3B/model"),
        "result": (
            "Enter Your Message: your persona:I have a job. I have 3 sisters. I am going to Hawaii next week.\npartner's persona: I can speak 3 languages.\nHave you made all the travel plans to Hawaii?"
            "[MemoryLongRag]: Yes, I have. I can't wait to go. Have you been to hawaii before?"
        ),
    },
    {
        "title": "SUMMSC-FidRAG 2.7B",
        "id": "msc",
        "path": "zoo:msc/summsc_fidrag3B/model",
        "agent": "projects.msc.agents.memory_agent:MemoryLongFidAgent",
        "task": "msc",
        "project": "https://parl.ai/projects/msc/",
        "description": (
            "SUMMSC 2.7B FidRAG Model with truncate 1024. See project for details."
        ),
        "example": ("parlai interactive -mf zoo:msc/summsc_fidrag3B/model"),
        "result": (
            "Enter Your Message: your persona:I have a job. I have 3 sisters. I am going to Hawaii next week.\npartner's persona: I can speak 3 languages.\nHave you made all the travel plans to Hawaii?"
            "[MemoryLongFid]: Yes, I'm going with my three sisters to hawaii. Have you ever been?"
        ),
    },
    {
        "title": "Dialogue Summarization Model",
        "id": "msc",
        "path": "zoo:msc/dialog_summarizer/model",
        "agent": "transformer/generator",
        "task": "msc",
        "project": "https://parl.ai/projects/msc/",
        "description": (
            "Dialogue Summarization Model tha summarize personal knowledge of the last speaker. "
            "See project for details."
        ),
        "example": ("parlai interactive -mf zoo:msc/dialog_summarizer/model"),
        "result": (
            "Enter Your Message: Do you know which puppy you want to adopt?\nMaybe a corgi."
            "[TransformerGenerator]: I want to adopt a corgi puppy."
        ),
    },
]
