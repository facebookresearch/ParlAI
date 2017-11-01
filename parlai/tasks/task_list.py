# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""This file contains a list of all the tasks, their id and task name, description
and the tags associated with them.
"""

task_list = [
    {
        "id": "bAbI-1k",
        "display_name": "bAbI 1k",
        "task": "babi:All1k",
        "tags": [ "All",  "QA" ],
        "description": "20 synthetic tasks that each test a unique aspect of text and reasoning, and hence test different capabilities of learning models. From Weston et al. '16. Link: http://arxiv.org/abs/1502.05698",
        "notes": "You can access just one of the bAbI tasks with e.g. 'babi:Task1k:3' for task 3."
    },
    {
        "id": "bAbI-10k",
        "display_name": "bAbI 10k",
        "task": "babi:All10k",
        "tags": [ "All",  "QA" ],
        "description": "20 synthetic tasks that each test a unique aspect of text and reasoning, and hence test different capabilities of learning models. From Weston et al. '16. Link: http://arxiv.org/abs/1502.05698",
        "notes": "You can access just one of the bAbI tasks with e.g. 'babi:Task10k:3' for task 3."
    },
    {
        "id": "BookTest",
        "display_name": "BookTest",
        "task": "booktest",
        "tags": [ "All",  "Cloze" ],
        "description": "Sentence completion given a few sentences as context from a book. A larger version of CBT. From Bajgar et al., 16. Link: https://arxiv.org/abs/1610.00956"
    },
    {
        "id": "CBT",
        "display_name": "Children's Book Test (CBT)",
        "task": "cbt",
        "tags": [ "All",  "Cloze" ],
        "description": "Sentence completion given a few sentences as context from a children's book. From Hill et al., '16. Link: https://arxiv.org/abs/1511.02301"
    },
    {
        "id": "CornellMovie",
        "display_name": "Cornell Movie",
        "task": "cornell_movie",
        "tags": [ "All",  "ChitChat" ],
        "description": "Fictional conversations extracted from raw movie scripts. Link: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html"
    },
    {
        "id": "DBLL-bAbI",
        "display_name": "Dialog Based Language Learning: bAbI Task",
        "task": "dbll_babi",
        "tags": [ "All",  "Goal" ],
        "description": "Short dialogs based on the bAbI tasks, but in the form of a question from a teacher, the answer from the student, and finally a comment on the answer from the teacher. The aim is to find learning models that use the comments to improve. From Weston '16. Link: https://arxiv.org/abs/1604.06045. Tasks can be accessed with a format like: 'python examples/display_data.py -t dbll_babi:task:2_p0.5' which specifies task 2, and policy with 0.5 answers correct, see the paper for more details of the tasks."
    },
    {
        "id": "DBLL-Movie",
        "display_name": "Dialog Based Language Learning: WikiMovies Task",
        "task": "dbll_movie",
        "tags": [ "All",  "Goal" ],
        "description": "Short dialogs based on WikiMovies, but in the form of a question from a teacher, the answer from the student, and finally a comment on the answer from the teacher. The aim is to find learning models that use the comments to improve. From Weston '16. Link: https://arxiv.org/abs/1604.06045"
    },
    {
        "id": "dialog-bAbI",
        "display_name": "Dialog bAbI",
        "task": "dialog_babi",
        "tags": [ "All",  "Goal" ],
        "description": "Simulated dialogs of restaurant booking, from Bordes et al. '16. Link: https://arxiv.org/abs/1605.07683"
    },
    {
        "id": "dialog-bAbI-plus",
        "display_name": "Dialog bAbI+",
        "task": "dialog_babi_plus",
        "tags": ["All", "Goal"],
        "description": "bAbI+ is an extension of the bAbI Task 1 dialogues with everyday incremental dialogue phenomena (hesitations, restarts, and corrections) which model the disfluencies and communication problems in everyday spoken interaction in real-world environments. See https://www.researchgate.net/publication/319128941_Challenging_Neural_Dialogue_Models_with_Natural_Data_Memory_Networks_Fail_on_Incremental_Phenomena, http://aclweb.org/anthology/D17-1235"
    },
    {
        "id": "FVQA",
        "display_name": "FVQA",
        "task": "fvqa",
        "tags": [ "All", "Visual" ],
        "description": "The FVQA, a VQA dataset which requires, and supports, much deeper reasoning. We extend a conventional visual question answering dataset, which contains image-question-answer triplets, through additional image-question-answer-supporting fact tuples. The supporting fact is represented as a structural triplet, such as <Cat,CapableOf,ClimbingTrees>.  Link: https://arxiv.org/abs/1606.05433"
    },
    {
        "id": "DealNoDeal",
        "display_name": "Deal or No Deal",
        "task": "dealnodeal",
        "tags": ["All", "Negotiation"],
        "description": "End-to-end negotiation task which requires two agents to agree on how to divide a set of items, with each agent assigning different values to each item. From Lewis et al. '17. Link: https://arxiv.org/abs/1706.05125"
    },
    {
        "id": "MutualFriends",
        "display_name": "MutualFriends",
        "task": "mutualfriends",
        "tags": [ "All", "Negotiation"],
        "description": "Task where two agents must discover which friend of theirs is mutual based on the friends's attributes. From He He et al. '17. Link: https://stanfordnlp.github.io/cocoa/'"
    },
    {
        "id": "MCTest",
        "display_name": "MCTest",
        "task": "mctest",
        "tags": [ "All",  "QA" ],
        "description": "Questions about short children's stories, from Richardson et al. '13. Link: https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/"
    },
    {
        "id": "MovieDD-QA",
        "display_name": "Movie Dialog QA",
        "task": "moviedialog:Task:1",
        "tags": [ "All",  "QA", "MovieDD" ],
        "description": "Closed-domain QA dataset asking templated questions about movies, answerable from Wikipedia, similar to WikiMovies. From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931"
    },
    {
        "id": "MovieDD-QARecs",
        "display_name": "Movie Dialog QA Recommendations",
        "task": "moviedialog:Task:3",
        "tags": [ "All",  "Goal", "MovieDD" ],
        "description": "Dialogs discussing questions about movies as well as recommendations. From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931"
    },
    {
        "id": "MovieDD-Recs",
        "display_name": "Movie Dialog Recommendations",
        "task": "moviedialog:Task:2",
        "tags": [ "All",  "QA", "MovieDD" ],
        "description": "Questions asking for movie recommendations. From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931"
    },
    {
        "id": "MovieDD-Reddit",
        "display_name": "Movie Dialog Reddit",
        "task": "moviedialog:Task:4",
        "tags": [ "All",  "ChitChat", "MovieDD" ],
        "description": "Dialogs discussing Movies from Reddit (the Movies SubReddit). From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931"
    },
    {
        "id": "MTurkWikiMovies",
        "display_name": "MTurk WikiMovies",
        "task": "mturkwikimovies",
        "tags": [ "All",  "QA" ],
        "description": "Closed-domain QA dataset asking MTurk-derived questions about movies, answerable from Wikipedia. From Li et al. '16. Link: https://arxiv.org/abs/1611.09823"
    },
    {
        "id": "OpenSubtitles",
        "display_name": "Open Subtitles",
        "task": "opensubtitles",
        "tags": [ "All",  "ChitChat" ],
        "description": "Dataset of dialogs from movie scripts: http://opus.lingfil.uu.se/OpenSubtitles.php. A variant of the dataset used in Vinyals & Le '15, https://arxiv.org/abs/1506.05869."
    },
    {
        "id": "personalized-dialog-full",
        "display_name": "Personalized Dialog Full Set",
        "task": "personalized_dialog:AllFull",
        "tags": [ "All",  "Goal", "Personalization" ],
        "description": "Simulated dataset of restaurant booking focused on personalization based on user profiles. From Joshi et al. '17. Link: https://arxiv.org/abs/1706.07503"
    },
    {
        "id": "personalized-dialog-small",
        "display_name": "Personalized Dialog Small Set",
        "task": "personalized_dialog:AllSmall",
        "tags": [ "All",  "Goal", "Personalization" ],
        "description": "Simulated dataset of restaurant booking focused on personalization based on user profiles. From Joshi et al. '17. Link: https://arxiv.org/abs/1706.07503"
    },
    {
        "id": "QACNN",
        "display_name": "QA CNN",
        "task": "qacnn",
        "tags": [ "All",  "Cloze" ],
        "description": "Cloze dataset based on a missing (anonymized) entity phrase from a CNN article, Hermann et al. '15. Link: https://arxiv.org/abs/1506.03340"
    },
    {
        "id": "QADailyMail",
        "display_name": "QA Daily Mail",
        "task": "qadailymail",
        "tags": [ "All",  "Cloze" ],
        "description": "Cloze dataset based on a missing (anonymized) entity phrase from a Daily Mail article, Hermann et al. '15. Link: https://arxiv.org/abs/1506.03340"
    },
    {
        "id": "SimpleQuestions",
        "display_name": "Simple Questions",
        "task": "simplequestions",
        "tags": [ "All",  "QA" ],
        "description": "Open-domain QA dataset based on Freebase triples from Bordes et al. '15. Link: https://arxiv.org/abs/1506.02075"
    },
    {
        "id": "SQuAD",
        "display_name": "SQuAD",
        "task": "squad",
        "tags": [ "All",  "QA" ],
        "description": "Open-domain QA dataset answerable from a given paragraph from Wikipedia, from Rajpurkar et al. '16. Link: https://arxiv.org/abs/1606.05250"
    },
    {
        "id": "TriviaQA",
        "display_name": "TriviaQA",
        "task": "triviaqa",
        "tags": [ "All",  "QA" ],
        "description": "Open-domain QA dataset with question-answer-evidence triples, from Joshi et al. '17. Link: https://arxiv.org/abs/1705.03551"
    },
    {
        "id": "Ubuntu",
        "display_name": "Ubuntu",
        "task": "ubuntu",
        "tags": [ "All",  "ChitChat" ],
        "description": "Dialogs between an Ubuntu user and an expert trying to fix issue, from Lowe et al. '15. Link: https://arxiv.org/abs/1506.08909"
    },
    {
        "id": "WebQuestions",
        "display_name": "Web Questions",
        "task": "webquestions",
        "tags": [ "All",  "QA" ],
        "description": "Open-domain QA dataset from Web queries from Berant et al. '13. Link: http://www.aclweb.org/anthology/D13-1160"
    },
    {
        "id": "WikiMovies",
        "display_name": "WikiMovies",
        "task": "wikimovies",
        "tags": [ "All",  "QA" ],
        "description": "Closed-domain QA dataset asking templated questions about movies, answerable from Wikipedia. From Miller et al. '16. Link: https://arxiv.org/abs/1606.03126"
    },
    {
        "id": "WikiQA",
        "display_name": "WikiQA",
        "task": "wikiqa",
        "tags": [ "All",  "QA" ],
        "description": "Open domain QA from Wikipedia dataset from Yang et al. '15. Link: https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/"
    },
    {
        "id": "VQAv1",
        "display_name": "VQAv1",
        "task": "vqa_v1",
        "tags": [ "All", "Visual" ],
        "description": "Open-ended question answering about visual content. From Agrawal et al. '15. Link: https://arxiv.org/abs/1505.00468"
    },
    {
        "id": "VQAv2",
        "display_name": "VQAv2",
        "task": "vqa_v2",
        "tags": [ "All", "Visual" ],
        "description": "Bigger, more balanced version of the original VQA dataset. From Goyal et al. '16. Link: https://arxiv.org/abs/1612.00837"
    },
    {
        "id": "VisDial",
        "display_name": "VisDial",
        "task": "visdial",
        "tags": [ "All", "Visual" ],
        "description": "Task which requires agents to hold a meaningful dialog about visual content. From Das et al. '16. Link: https://arxiv.org/abs/1611.08669"
    },
    {
        "id": "MNIST_QA",
        "display_name": "MNIST_QA",
        "task": "mnist_qa",
        "tags": [ "All", "Visual" ],
        "description": "Task which requires agents to identify which number they are seeing. From the MNIST dataset."
    },
    {
        "id": "InsuranceQA",
        "display_name": "InsuranceQA",
        "task": "insuranceqa",
        "tags": [ "All",  "QA" ],
        "description": "Task which requires agents to identify high quality answers composed by professionals with deep domain knowledge. From Feng et al. '15. Link: https://arxiv.org/abs/1508.01585"
    },
    {
        "id": "MS_MARCO",
        "display_name": "MS_MARCO",
        "task": "ms_marco",
        "tags": [ "All",  "QA" ],
        "description": "A large scale Machine Reading Comprehension Dataset with questions sampled from real anonymized user queries and contexts from web documents. From Nguyen et al. '16. Link: https://arxiv.org/abs/1611.09268"
    },
    {
        "id": "CLEVR",
        "display_name": "CLEVR",
        "task": "clevr",
        "tags": [ "All",  "Visual" ],
        "description": "A visual reasoning dataset that tests abilities such as attribute identification, counting, comparison, spatial relationships, and logical operations. From Johnson et al. '16. Link: https://arxiv.org/abs/1612.06890"
    },
    {
        "id": "WMT",
        "display_name": "WMT",
        "task": "wmt",
        "tags": [ "All", "MT" ],
        "description": "Workshop on Machine Translation task, currently only includes en_de."
    },
    {
        "id": "ConvAI_ChitChat",
        "display_name": "ConvAI_ChitChat",
        "task": "convai_chitchat",
        "tags": [ "All", "ChitChat" ],
        "description": "Human-bot dialogues containing free discussions of randomly chosen paragraphs from SQuAD. Link to dataset: http://convai.io/data/"
    }
]
