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
		"task": "babi:All1k",
		"tags": [ "all",  "QA" ],
                "description": "20 synthetic tasks that each test a unique aspect of text and reasoning, and hence test different capabilities of learning models. From Weston et al. '16. Link: http://arxiv.org/abs/1502.05698",
                "notes": "You can access just one of the bAbI tasks with e.g. 'babi:Task1k:3' for task 3."
	},
	{
		"id": "bAbI-10k",
		"task": "babi:All10k",
		"tags": [ "all",  "QA" ],
                "description": "20 synthetic tasks that each test a unique aspect of text and reasoning, and hence test different capabilities of learning models. From Weston et al. '16. Link: http://arxiv.org/abs/1502.05698",
                "notes": "You can access just one of the bAbI tasks with e.g. 'babi:Task10k:3' for task 3."
	},
	{
		"id": "BookTest",
		"task": "booktest",
		"tags": [ "all",  "Cloze" ],
                "description": "Sentence completion given a few sentences as context from a book. A larger version of CBT. From Bajgar et al., 16. Link: https://arxiv.org/abs/1610.00956"
	},
	{
		"id": "CBT",
		"task": "cbt",
		"tags": [ "all",  "Cloze" ],
                "description": "Sentence completion given a few sentences as context from a children's book. From Hill et al., '16. Link: https://arxiv.org/abs/1511.02301"
	},
	{
		"id": "CornellMovie",
		"task": "cornell_movie",
		"tags": [ "all",  "ChitChat" ],
                "description": "Fictional conversations extracted from raw movie scripts. Link: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html"
	},
	{
		"id": "DBLL-bAbI",
		"task": "dbll_babi",
		"tags": [ "all",  "Goal" ],
                "description": "Short dialogs based on the bAbI tasks, but in the form of a question from a teacher, the answer from the student, and finally a comment on the answer from the teacher. The aim is to find learning models that use the comments to improve. From Weston '16. Link: https://arxiv.org/abs/1604.06045"
	},
	{
		"id": "DBLL-Movie",
		"task": "dbll_movie",
		"tags": [ "all",  "Goal" ],
                "description": "Short dialogs based on WikiMovies, but in the form of a question from a teacher, the answer from the student, and finally a comment on the answer from the teacher. The aim is to find learning models that use the comments to improve. From Weston '16. Link: https://arxiv.org/abs/1604.06045"
	},
	{
		"id": "dialog-bAbI",
		"task": "dialog_babi",
		"tags": [ "all",  "Goal" ],
                "description": "Simulated dialogs of restaurant booking, from Bordes et al. '16. Link: https://arxiv.org/abs/1605.07683"
	},
	{
		"id": "MCTest",
		"task": "mctest",
		"tags": [ "all",  "QA" ],
                "description": "Questions about short children's stories, from Richardson et al. '13. Link: https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/"
	},
	{
		"id": "MovieDD-QA",
		"task": "moviedialog:Task:1",
		"tags": [ "all",  "QA", "MovieDD" ],
                "description": "Closed-domain QA dataset asking templated questions about movies, answerable from Wikipedia, similar to WikiMovies. From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931"
	},
	{
		"id": "MovieDD-QARecs",
		"task": "moviedialog:Task:3",
		"tags": [ "all",  "Goal", "MovieDD" ],
                "description": "Dialogs discussing questions about movies as well as recommendations. From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931"
	},
	{
		"id": "MovieDD-Recs",
		"task": "moviedialog:Task:2",
		"tags": [ "all",  "QA", "MovieDD" ],
                "description": "Questions asking for movie recommendations. From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931"
	},
	{
		"id": "MovieDD-Reddit",
		"task": "moviedialog:Task:4",
		"tags": [ "all",  "ChitChat", "MovieDD" ],
                "description": "Dialogs discussing Movies from Reddit (the Movies SubReddit). From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931"
	},
	{
		"id": "MTurkWikiMovies",
		"task": "mturkwikimovies",
		"tags": [ "all",  "QA" ],
                "description": "Closed-domain QA dataset asking MTurk-derived questions about movies, answerable from Wikipedia. From Li et al. '16. Link: https://arxiv.org/abs/1611.09823"
	},
	{
		"id": "OpenSubtitles",
		"task": "opensubtitles",
		"tags": [ "all",  "ChitChat" ],
                "description": "Dataset of dialogs from movie scripts: http://opus.lingfil.uu.se/OpenSubtitles.php. A variant of the dataset used in Vinyals & Le '15, https://arxiv.org/abs/1506.05869."
	},
	{
		"id": "QACNN",
		"task": "qacnn",
		"tags": [ "all",  "Cloze" ],
                "description": "Cloze dataset based on a missing (anonymized) entity phrase from a CNN article, Hermann et al. '15. Link: https://arxiv.org/abs/1506.03340"       
	},
	{
		"id": "QADailyMail",
		"task": "qadailymail",
		"tags": [ "all",  "Cloze" ],
                "description": "Cloze dataset based on a missing (anonymized) entity phrase from a Daily Mail article, Hermann et al. '15. Link: https://arxiv.org/abs/1506.03340"
	},
	{
		"id": "SimpleQuestions",
		"task": "simplequestions",
		"tags": [ "all",  "QA" ],
                "description": "Open-domain QA dataset based on Freebase triples from Bordes et al. '15. Link: https://arxiv.org/abs/1506.02075"
	},
	{
		"id": "SQuAD",
		"task": "squad",
		"tags": [ "all",  "QA" ],
                "description": "Open-domain QA dataset answerable from a given paragraph from Wikipedia, from Rajpurkar et al. '16. Link: https://arxiv.org/abs/1606.05250"
	},
	{
		"id": "Ubuntu",
		"task": "ubuntu",
		"tags": [ "all",  "ChitChat" ],
                "description": "Dialogs between an Ubunt user and an expert trying to fix issue, from Lowe et al. '15. Link: https://arxiv.org/abs/1506.08909"
	},
	{
		"id": "WebQuestions",
		"task": "webquestions",
		"tags": [ "all",  "QA" ],
                "description": "Open-domain QA dataset from Web queries from Berant et al. '13. Link: http://www.aclweb.org/anthology/D13-1160"
	},
	{
		"id": "WikiMovies",
		"task": "wikimovies",
		"tags": [ "all",  "QA" ],
                "description": "Closed-domain QA dataset asking templated questions about movies, answerable from Wikipedia. From Miller et al. '16. Link: https://arxiv.org/abs/1606.03126"
	},
	{
		"id": "WikiQA",
		"task": "wikiqa",
		"tags": [ "all",  "QA" ],
                "description": "Open domain QA from Wikipedia dataset from Yang, et al. '15. Link: https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/"
	},
]
