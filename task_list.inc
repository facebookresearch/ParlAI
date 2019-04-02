They consist of:  (1) QA tasks;  (2) Cloze tasks;  (3) Goal tasks;  (4) ChitChat tasks;  (5) Negotiation tasks;  (6) Visual tasks;   and (7) decanlp tasks.

QA Tasks
---------

**AQuA**   Dataset containing algebraic word problems with rationales for their answers. From Ling et. al. 2017, Link: https://arxiv.org/pdf/1705.04146.pdf  [ task:`aqua <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/aqua>`_  tags:``#AQuA``, ``#All``, ``#QA`` ]



**bAbI 1k**   20 synthetic tasks that each test a unique aspect of text and reasoning, and hence test different capabilities of learning models. From Weston et al. '16. Link: http://arxiv.org/abs/1502.05698   [ task:`babi:All1k <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/babi:All1k>`_  tags:``#bAbI-1k``, ``#All``, ``#QA`` ]



**bAbI 10k**   20 synthetic tasks that each test a unique aspect of text and reasoning, and hence test different capabilities of learning models. From Weston et al. '16. Link: http://arxiv.org/abs/1502.05698  [ task:`babi:All10k <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/babi:All10k>`_  tags:``#bAbI-10k``, ``#All``, ``#QA`` ]



**MCTest**   Questions about short children's stories, from Richardson et al. '13. Link: https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/  [ task:`mctest <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/mctest>`_  tags:``#MCTest``, ``#All``, ``#QA`` ]



**Movie Dialog QA**   Closed-domain QA dataset asking templated questions about movies, answerable from Wikipedia, similar to WikiMovies. From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931  [ task:`moviedialog:Task:1 <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/moviedialog:Task:1>`_  tags:``#MovieDD-QA``, ``#All``, ``#QA``, ``#MovieDD`` ]



**Movie Dialog Recommendations**   Questions asking for movie recommendations. From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931  [ task:`moviedialog:Task:2 <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/moviedialog:Task:2>`_  tags:``#MovieDD-Recs``, ``#All``, ``#QA``, ``#MovieDD`` ]



**MTurk WikiMovies**   Closed-domain QA dataset asking MTurk-derived questions about movies, answerable from Wikipedia. From Li et al. '16. Link: https://arxiv.org/abs/1611.09823  [ task:`mturkwikimovies <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/mturkwikimovies>`_  tags:``#MTurkWikiMovies``, ``#All``, ``#QA`` ]



**NarrativeQA**   A dataset and set of tasks in which the reader must answer questions about stories by reading entire books or movie scripts. From Kočiský et. al. '17. Link: https://arxiv.org/abs/1712.07040'  [ task:`narrative_qa <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/narrative_qa>`_  tags:``#NarrativeQA``, ``#All``, ``#QA`` ]



**Simple Questions**   Open-domain QA dataset based on Freebase triples from Bordes et al. '15. Link: https://arxiv.org/abs/1506.02075  [ task:`simplequestions <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/simplequestions>`_  tags:``#SimpleQuestions``, ``#All``, ``#QA`` ]



**SQuAD2**   Open-domain QA dataset answerable from a given paragraph from Wikipedia, from Rajpurkar & Jia et al. '18. Link: http://arxiv.org/abs/1806.03822  [ task:`squad2 <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/squad2>`_  tags:``#SQuAD2``, ``#All``, ``#QA`` ]



**SQuAD**   Open-domain QA dataset answerable from a given paragraph from Wikipedia, from Rajpurkar et al. '16. Link: https://arxiv.org/abs/1606.05250  [ task:`squad <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/squad>`_  tags:``#SQuAD``, ``#All``, ``#QA`` ]



**TriviaQA**   Open-domain QA dataset with question-answer-evidence triples, from Joshi et al. '17. Link: https://arxiv.org/abs/1705.03551  [ task:`triviaqa <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/triviaqa>`_  tags:``#TriviaQA``, ``#All``, ``#QA`` ]



**Web Questions**   Open-domain QA dataset from Web queries from Berant et al. '13. Link: http://www.aclweb.org/anthology/D13-1160  [ task:`webquestions <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/webquestions>`_  tags:``#WebQuestions``, ``#All``, ``#QA`` ]



**WikiMovies**   Closed-domain QA dataset asking templated questions about movies, answerable from Wikipedia. From Miller et al. '16. Link: https://arxiv.org/abs/1606.03126  [ task:`wikimovies <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/wikimovies>`_  tags:``#WikiMovies``, ``#All``, ``#QA`` ]



**WikiQA**   Open domain QA from Wikipedia dataset from Yang et al. '15. Link: https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/  [ task:`wikiqa <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/wikiqa>`_  tags:``#WikiQA``, ``#All``, ``#QA`` ]



**InsuranceQA**   Task which requires agents to identify high quality answers composed by professionals with deep domain knowledge. From Feng et al. '15. Link: https://arxiv.org/abs/1508.01585  [ task:`insuranceqa <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/insuranceqa>`_  tags:``#InsuranceQA``, ``#All``, ``#QA`` ]



**MS_MARCO**   A large scale Machine Reading Comprehension Dataset with questions sampled from real anonymized user queries and contexts from web documents. From Nguyen et al. '16. Link: https://arxiv.org/abs/1611.09268  [ task:`ms_marco <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/ms_marco>`_  tags:``#MS_MARCO``, ``#All``, ``#QA`` ]



**QAngaroo**   Reading Comprehension with Multiple Hop. Including two datasets: WIKIHOP built on on wikipedia, MEDHOP built on paper abstracts from PubMed. Link to dataset: http://qangaroo.cs.ucl.ac.uk/  [ task:`qangaroo <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/qangaroo>`_  tags:``#QAngaroo``, ``#All``, ``#QA`` ]



Cloze Tasks
------------

**BookTest**   Sentence completion given a few sentences as context from a book. A larger version of CBT. From Bajgar et al., 16. Link: https://arxiv.org/abs/1610.00956  [ task:`booktest <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/booktest>`_  tags:``#BookTest``, ``#All``, ``#Cloze`` ]



**Children's Book Test (CBT)**   Sentence completion given a few sentences as context from a children's book. From Hill et al., '16. Link: https://arxiv.org/abs/1511.02301  [ task:`cbt <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/cbt>`_  tags:``#CBT``, ``#All``, ``#Cloze`` ]



**QA CNN**   Cloze dataset based on a missing (anonymized) entity phrase from a CNN article, Hermann et al. '15. Link: https://arxiv.org/abs/1506.03340  [ task:`qacnn <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/qacnn>`_  tags:``#QACNN``, ``#All``, ``#Cloze`` ]



**QA Daily Mail**   Cloze dataset based on a missing (anonymized) entity phrase from a Daily Mail article, Hermann et al. '15. Link: https://arxiv.org/abs/1506.03340  [ task:`qadailymail <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/qadailymail>`_  tags:``#QADailyMail``, ``#All``, ``#Cloze`` ]



Goal Tasks
-----------

**Dialog Based Language Learning: bAbI Task**   Short dialogs based on the bAbI tasks, but in the form of a question from a teacher, the answer from the student, and finally a comment on the answer from the teacher. The aim is to find learning models that use the comments to improve. From Weston '16. Link: https://arxiv.org/abs/1604.06045. Tasks can be accessed with a format like: 'python examples/display_data.py -t dbll_babi:task:2_p0.5' which specifies task 2, and policy with 0.5 answers correct, see the paper for more details of the tasks.  [ task:`dbll_babi <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/dbll_babi>`_  tags:``#DBLL-bAbI``, ``#All``, ``#Goal`` ]



**Dialog Based Language Learning: WikiMovies Task**   Short dialogs based on WikiMovies, but in the form of a question from a teacher, the answer from the student, and finally a comment on the answer from the teacher. The aim is to find learning models that use the comments to improve. From Weston '16. Link: https://arxiv.org/abs/1604.06045  [ task:`dbll_movie <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/dbll_movie>`_  tags:``#DBLL-Movie``, ``#All``, ``#Goal`` ]



**Dialog bAbI**   Simulated dialogs of restaurant booking, from Bordes et al. '16. Link: https://arxiv.org/abs/1605.07683  [ task:`dialog_babi <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/dialog_babi>`_  tags:``#dialog-bAbI``, ``#All``, ``#Goal`` ]



**Dialog bAbI+**   bAbI+ is an extension of the bAbI Task 1 dialogues with everyday incremental dialogue phenomena (hesitations, restarts, and corrections) which model the disfluencies and communication problems in everyday spoken interaction in real-world environments. See https://www.researchgate.net/publication/319128941_Challenging_Neural_Dialogue_Models_with_Natural_Data_Memory_Networks_Fail_on_Incremental_Phenomena,http://aclweb.org/anthology/D17-1235  [ task:`dialog_babi_plus <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/dialog_babi_plus>`_  tags:``#dialog-bAbI-plus``, ``#All``, ``#Goal`` ]



**MutualFriends**   Task where two agents must discover which friend of theirs is mutual based on the friends's attributes. From He He et al. '17. Link: https://stanfordnlp.github.io/cocoa/  [ task:`mutualfriends <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/mutualfriends>`_  tags:``#MutualFriends``, ``#All``, ``#Goal`` ]



**Movie Dialog QA Recommendations**   Dialogs discussing questions about movies as well as recommendations. From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931  [ task:`moviedialog:Task:3 <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/moviedialog:Task:3>`_  tags:``#MovieDD-QARecs``, ``#All``, ``#Goal``, ``#MovieDD`` ]



**Personalized Dialog Full Set**   Simulated dataset of restaurant booking focused on personalization based on user profiles. From Joshi et al. '17. Link: https://arxiv.org/abs/1706.07503  [ task:`personalized_dialog:AllFull <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personalized_dialog:AllFull>`_  tags:``#personalized-dialog-full``, ``#All``, ``#Goal``, ``#Personalization`` ]



**Personalized Dialog Small Set**   Simulated dataset of restaurant booking focused on personalization based on user profiles. From Joshi et al. '17. Link: https://arxiv.org/abs/1706.07503  [ task:`personalized_dialog:AllSmall <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personalized_dialog:AllSmall>`_  tags:``#personalized-dialog-small``, ``#All``, ``#Goal``, ``#Personalization`` ]



**Task N' Talk**   Dataset of synthetic shapes described by attributes, for agents to play a cooperative QA game, from Kottur et al. '17. Link: https://arxiv.org/abs/1706.08502  [ task:`taskntalk <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/taskntalk>`_  tags:``#TaskNTalk``, ``#All``, ``#Goal`` ]



**SCAN**   SCAN is a set of simple language-driven navigation tasks for studying compositional learning and zero-shot generalization. The SCAN tasks were inspired by the CommAI environment, which is the origin of the acronym (Simplified versions of the CommAI Navigation tasks). See the paper: https://arxiv.org/abs/1711.00350 or data: https://github.com/brendenlake/SCAN  [ task:`scan <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/scan>`_  tags:``#SCAN``, ``#Goal``, ``#All`` ]



ChitChat Tasks
---------------

**Cornell Movie**   Fictional conversations extracted from raw movie scripts. Danescu-Niculescu-Mizil & Lee, '11. Link: https://arxiv.org/abs/1106.3077  [ task:`cornell_movie <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/cornell_movie>`_  tags:``#CornellMovie``, ``#All``, ``#ChitChat`` ]



**Movie Dialog Reddit**   Dialogs discussing Movies from Reddit (the Movies SubReddit). From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931  [ task:`moviedialog:Task:4 <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/moviedialog:Task:4>`_  tags:``#MovieDD-Reddit``, ``#All``, ``#ChitChat``, ``#MovieDD`` ]



**Open Subtitles**   Dataset of dialogs from movie scripts. Version 2018: http://opus.lingfil.uu.se/OpenSubtitles2018.php, version 2009: http://opus.lingfil.uu.se/OpenSubtitles.php. A variant of the dataset used in Vinyals & Le '15, https://arxiv.org/abs/1506.05869.  [ task:`opensubtitles <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/opensubtitles>`_  tags:``#OpenSubtitles``, ``#All``, ``#ChitChat`` ]



**Ubuntu**   Dialogs between an Ubuntu user and an expert trying to fix issue, from Lowe et al. '15. Link: https://arxiv.org/abs/1506.08909  [ task:`ubuntu <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/ubuntu>`_  tags:``#Ubuntu``, ``#All``, ``#ChitChat`` ]



**ConvAI2**   A chit-chat dataset based on PersonaChat (https://arxiv.org/abs/1801.07243) for a NIPS 2018 competition. Link: http://convai.io/.  [ task:`convai2 <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/convai2>`_  tags:``#ConvAI2``, ``#All``, ``#ChitChat`` ]



**ConvAI_ChitChat**   Human-bot dialogues containing free discussions of randomly chosen paragraphs from SQuAD. Link to dataset: http://convai.io/data/  [ task:`convai_chitchat <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/convai_chitchat>`_  tags:``#ConvAI_ChitChat``, ``#All``, ``#ChitChat``, ``#decanlp`` ]



**Persona-Chat**   A chit-chat dataset where paired Turkers are given assigned personas and chat to try to get to know each other. See the paper: https://arxiv.org/abs/1801.07243  [ task:`personachat <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat>`_  tags:``#Persona-Chat``, ``#ChitChat``, ``#All`` ]



**Twitter**   Twitter data from: https://github.com/Marsan-Ma/chat_corpus/. No train/valid/test split was provided so 10k for valid and 10k for test was chosen at random.  [ task:`twitter <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/twitter>`_  tags:``#Twitter``, ``#All``, ``#ChitChat`` ]



**ConvAI2_wild_evaluation**   Dataset collected during the wild evaluation of ConvaAI2 participants bots (http://convai.io). 60% train, 20% valid and 20% test is chosen at random from the whole dataset.  [ task:`convai2_wild_evaluation <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/convai2_wild_evaluation>`_  tags:``#ConvAI2_wild_evaluation``, ``#All``, ``#ChitChat`` ]



**Image_Chat**   202k dialogues and 401k utterances over 202k images from the YFCC100m dataset(https://multimediacommons.wordpress.com/yfcc100m-core-dataset/)using 215 possible personality traitssee https://klshuster.github.io/image_chat/ for more information.  [ task:`image_chat <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/image_chat>`_  tags:``#Image_Chat``, ``#All``, ``#Visual``, ``#ChitChat`` ]



**Wizard_of_Wikipedia**   A dataset with conversations directly grounded with knowledge retrieved from Wikipedia. Contains 201k utterances from 22k dialogues spanning over 1300 diverse topics, split into train, test, and valid sets. The test and valid sets are split into two sets each: one with overlapping topics with the train set, and one with unseen topics.See https://arxiv.org/abs/1811.01241 for more information.  [ task:`wizard_of_wikipedia <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/wizard_of_wikipedia>`_  tags:``#Wizard_of_Wikipedia``, ``#All``, ``#ChitChat`` ]



**Daily Dialog**   A dataset of chitchat dialogues with strong annotations for topic, emotion and utterance act. This version contains both sides of every conversation, and uses the official train/valid/test splits from the original authors. See https://arxiv.org/abs/1710.03957 for more information.  [ task:`dailydialog <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/dailydialog>`_  tags:``#DailyDialog``, ``#All``, ``#ChitChat`` ]



**Empathetic Dialogues**   A dataset of 25k conversations grounded in emotional situations to facilitate training and evaluating dialogue systems. See https://arxiv.org/abs/1811.00207 for more information.  [ task:`empathetic_dialogues <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/empathetic_dialogues>`_  tags:``#EmpatheticDialogues``, ``#All``, ``#ChitChat`` ]



Negotiation Tasks
------------------

**Deal or No Deal**   End-to-end negotiation task which requires two agents to agree on how to divide a set of items, with each agent assigning different values to each item. From Lewis et al. '17. Link: https://arxiv.org/abs/1706.05125  [ task:`dealnodeal <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/dealnodeal>`_  tags:``#DealNoDeal``, ``#All``, ``#Negotiation`` ]



Visual Tasks
-------------

**FVQA**   The FVQA, a VQA dataset which requires, and supports, much deeper reasoning. We extend a conventional visual question answering dataset, which contains image-question-answer triplets, through additional image-question-answer-supporting fact tuples. The supporting fact is represented as a structural triplet, such as <Cat,CapableOf,ClimbingTrees>.  Link: https://arxiv.org/abs/1606.05433  [ task:`fvqa <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/fvqa>`_  tags:``#FVQA``, ``#All``, ``#Visual`` ]



**VQAv1**   Open-ended question answering about visual content. From Agrawal et al. '15. Link: https://arxiv.org/abs/1505.00468  [ task:`vqa_v1 <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/vqa_v1>`_  tags:``#VQAv1``, ``#All``, ``#Visual`` ]



**VQAv2**   Bigger, more balanced version of the original VQA dataset. From Goyal et al. '16. Link: https://arxiv.org/abs/1612.00837  [ task:`vqa_v2 <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/vqa_v2>`_  tags:``#VQAv2``, ``#All``, ``#Visual`` ]



**VisDial**   Task which requires agents to hold a meaningful dialog about visual content. From Das et al. '16. Link: https://arxiv.org/abs/1611.08669  [ task:`visdial <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/visdial>`_  tags:``#VisDial``, ``#All``, ``#Visual`` ]



**MNIST_QA**   Task which requires agents to identify which number they are seeing. From the MNIST dataset.  [ task:`mnist_qa <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/mnist_qa>`_  tags:``#MNIST_QA``, ``#All``, ``#Visual`` ]



**CLEVR**   A visual reasoning dataset that tests abilities such as attribute identification, counting, comparison, spatial relationships, and logical operations. From Johnson et al. '16. Link: https://arxiv.org/abs/1612.06890  [ task:`clevr <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/clevr>`_  tags:``#CLEVR``, ``#All``, ``#Visual`` ]



**nlvr**   Cornell Natural Language Visual Reasoning (NLVR) is a language grounding dataset based on  pairs of natural language statements grounded in synthetic images. From Suhr et al. '17. Link: http://lic.nlp.cornell.edu/nlvr/  [ task:`nlvr <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/nlvr>`_  tags:``#nlvr``, ``#All``, ``#Visual`` ]



**Flickr30k**   30k captioned images pulled from Flickr compiled by UIUC: http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/. Based off of these papers: https://arxiv.org/abs/1505.04870v2, http://aclweb.org/anthology/Q14-1006  [ task:`flickr30k <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/flickr30k>`_  tags:``#Flickr30k``, ``#All``, ``#Visual`` ]



**COCO_Captions**   COCO annotations derived from the 2015 COCO Caption Competition. Link to dataset: http://cocodataset.org/#download  [ task:`coco_caption <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/coco_caption>`_  tags:``#COCO_Captions``, ``#All``, ``#Visual`` ]



**Personality_Captions**   200k images from the YFCC100m dataset (https://multimediacommons.wordpress.com/yfcc100m-core-dataset/), with captions conditioned on one of 215 personalities. See https://arxiv.org/abs/1810.10665 for more information.  [ task:`personality_captions <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personality_captions>`_  tags:``#Personality_Captions``, ``#All``, ``#Visual`` ]



**Image_Chat**   202k dialogues and 401k utterances over 202k images from the YFCC100m dataset(https://multimediacommons.wordpress.com/yfcc100m-core-dataset/)using 215 possible personality traitssee https://klshuster.github.io/image_chat/ for more information.  [ task:`image_chat <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/image_chat>`_  tags:``#Image_Chat``, ``#All``, ``#Visual``, ``#ChitChat`` ]



**Talk the Walk**   Talk the walk dataset.See https://arxiv.org/abs/1807.03367 for more information.  [ task:`talkthewalk <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/talkthewalk>`_  tags:``#TalkTheWalk``, ``#All``, ``#Visual`` ]



decanlp Tasks
--------------

**MultiNLI**   A dataset designed for use in the development and evaluation of machine learning models for sentence understanding. Each example contains a premise and hypothesis. Model has to predict whether premise and hypothesis entail, contradict or are neutral to each other. From Williams et al. '17. Link: https://arxiv.org/abs/1704.05426  [ task:`multinli <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/multinli>`_  tags:``#MultiNLI``, ``#All``, ``#Entailment``, ``#decanlp`` ]



**IWSLT14**   2014 International Workshop on Spoken Language task, currently only includes en_de and de_en. From Cettolo et al. '12. Link: wit3.fbk.eu  [ task:`iwslt14 <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/iwslt14>`_  tags:``#IWSLT14``, ``#All``, ``#MT``, ``#decanlp`` ]



**ConvAI_ChitChat**   Human-bot dialogues containing free discussions of randomly chosen paragraphs from SQuAD. Link to dataset: http://convai.io/data/  [ task:`convai_chitchat <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/convai_chitchat>`_  tags:``#ConvAI_ChitChat``, ``#All``, ``#ChitChat``, ``#decanlp`` ]



**SST Sentiment Analysis**   Dataset containing sentiment trees of movie reviews. We use the modified binary sentence analysis subtask given by the DecaNLP paper here, originally from Radford, et. al https://nlp.stanford.edu/sentiment/index.html https://github.com/openai/generating-reviews-discovering-sentiment/  [ task:`sst <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/sst>`_  tags:``#sst``, ``#All``, ``#decanlp`` ]



**CNN/DM Summarisation**   Dataset collected from CNN and the Daily Mail with summaries as labels, Implemented as part of the DecaNLP taskDownloaded from https://cs.nyu.edu/~kcho/DMQA/  [ task:`cnn_dm <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/cnn_dm>`_  tags:``#cnn_dm``, ``#All``, ``#decanlp`` ]



**QA-SRL Semantic Role Labeling**   QA dataset implemented as part of the DecaNLP task. More info on thedataset can be found here: https://dada.cs.washington.edu/qasrl/  [ task:`qasrl <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/qasrl>`_  tags:``#qasrl``, ``#All``, ``#decanlp`` ]



**QA-ZRE Relation Extraction**   Zero Shot relation extraction task implemented as part of the DecaNLP task. More info on the dataset can be found here:http://nlp.cs.washington.edu/zeroshot/  [ task:`qazre <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/qazre>`_  tags:``#qazre``, ``#All``, ``#decanlp`` ]



**WOZ restuarant reservation (Goal-Oriented Dialogue)**   Dataset containing dialogues dengotiating a resturant reservation. Implemented as part of the DecaNLP task, focused on the change in the dialogue state. Original paper: https://arxiv.org/abs/1604.04562  [ task:`woz <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/woz>`_  tags:``#woz``, ``#All``, ``#decanlp`` ]



**WikiSQL semantic parsing task**   Dataset for parsing sentences to SQL code, given a table. Implemented as part of the DecaNLP task. More info can be found here:https://github.com/salesforce/WikiSQL  [ task:`wikisql <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/wikisql>`_  tags:``#wikisql``, ``#All``, ``#decanlp`` ]



**MWSC pronoun resolution**   Resolving possible ambiguous pronouns. Implemented as part of the DecaNLPtask, and can be found on the decaNLP github  [ task:`mwsc <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/mwsc>`_  tags:``#mwsc``, ``#All``, ``#decanlp`` ]



