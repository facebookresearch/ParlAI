## News

2019-05-03: The [What makes a good Conversation project page](https://parl.ai/projects/controllable_dialogue/) is now available with pretrained models.

2019-03-15: The [Wizard of Wikipedia project page](http://parl.ai/projects/wizard_of_wikipedia/) is updated with pretrained models and specialized model code.

2019-03-09: Added [LIGHT](http://parl.ai/projects/light) text adventure game research platform for learning to speak and act. [[press]](https://venturebeat.com/2019/03/08/facebook-ai-researchers-create-a-text-based-adventure-to-study-how-ai-speak-and-act/)

2019-02-07: Added [BERT Ranker agents](https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/bert_ranker), several variations of a ranking model based on the pretrained language model BERT.

2019-01-16: ParlAI has been relicensed under the MIT open source license.

2018-12-13: Added [Daily Dialog](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/dailydialog/agents.py), an open-domain daily dialogue dataset.

2018-11-05: Added [Wizard of Wikipedia](http://parl.ai/projects/wizard_of_wikipedia/), a dataset for knowledge-powered conversation.

2018-11-02: Added [Image-Chat](https://klshuster.github.io/image_chat/), a dataset for engaging personality-conditioned dialogue grounded in images.

2018-10-25: Added [Personality-Captions](https://arxiv.org/abs/1810.10665), a dataset for engaging image captioning via personality.

2018-08-29: Added new cleaner version of seq2seq model with new TorchAgent parent class, along with folder (parlai/legacy_agents) for deprecated model code

2018-07-17: Added [Qangaroo](http://qangaroo.cs.ucl.ac.uk/) (a.k.a. WikiHop and MedHop), two reading comprehension datasets with multiple hops, and [SQuAD 2](https://rajpurkar.github.io/SQuAD-explorer/).

2018-05-22: Two new tasks added: [COCO Image Captioning](http://cocodataset.org/#captions-2015) and [Flickr30k Entities](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/)

2018-04-13: [NIPS ConvAI2 competition!](http://convai.io/) Train Dialogue Agents to chat about personal interests and get to know their dialogue partner -- using the PersonaChat dataset as a training source, with data and baseline code in ParlAI. Competition starts now! Ends September 1st.

2018-03-13: Added [ParlAI-Messenger](http://parl.ai/static/docs/messenger.html), a new method for connecting human agents to a world in ParlAI using Facebook Messenger. Host your bots on Facebook Messenger to expose them to a broad audience!

2018-03-07: Added [IBM's sequence to sequence](https://github.com/IBM/pytorch-seq2seq) model to parlai/agents. To use it, just set --model ibm_seq2seq.

2018-03-05: Added [Multimodal Low-Rank Bilinear Attention Network (MLB)](https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/mlb_vqa/mlb_vqa.py) model for VQA V1 and V2 tasks, adapted from an implementation [here](https://github.com/Cadene/vqa.pytorch) based on [this paper](https://arxiv.org/abs/1610.04325). To use it, please follow the instructions [in the agent file](https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/mlb_vqa/mlb_vqa.py).

2018-02-12: Added a [Wikipedia task](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/wikipedia/agents.py), which provides a dump of Wikipedia articles from 2/3/2018.

2018-02-07: Added a [language model](https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/language_model/language_model.py) adapted from [this](https://github.com/pytorch/examples/tree/master/word_language_model) Pytorch model to parlai/agents.

2018-01-23: Several new tasks added: [SNLI](https://nlp.stanford.edu/projects/snli/), [MultiNLI](https://arxiv.org/abs/1704.05426), [COPA](http://people.ict.usc.edu/~gordon/copa.html), [NarrativeQA](https://github.com/deepmind/narrativeqa), Twitter and [Persona-Chat](https://arxiv.org/abs/1801.07243).

2017-12-14: Fast, multiprocessed data loading supported with [Pytorch data loader](https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/pytorch_data_teacher.py)

2017-11-30: Several new tasks added: [SCAN](https://github.com/brendenlake/SCAN), [ConvAI](http://convai.io/data/), [NVLR](http://lic.nlp.cornell.edu/nlvr/) and [ISWLT14](http://wit3.fbk.eu).

2017-10-19: [ParlAI Request For Proposals: Winners Announced!](https://research.fb.com/announcing-the-winners-of-the-facebook-parlai-research-awards/)

2017-10-13: [New model added: Fairseq-py](https://github.com/facebookresearch/fairseq-py)

2017-10-12: [New task added: Stanford's MutualFriends](https://stanfordnlp.github.io/cocoa/)

2017-09-22: [New task added: babi+](https://www.researchgate.net/publication/319128941_Challenging_Neural_Dialogue_Models_with_Natural_Data_Memory_Networks_Fail_on_Incremental_Phenomena)

2017-09-21: [New task added: WMT En-De training set, with more WMT tasks on the way](https://nlp.stanford.edu/projects/nmt/)

2017-08-25: [New task added: Deal or No Deal](https://github.com/facebookresearch/end-to-end-negotiator)

2017-08-15: [New task added: CLEVR](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/task_list.py)

2017-07-20: [ParlAI Request For Proposals: Funding university teams - 7 awards are available - deadline Aug 25](https://research.fb.com/programs/research-awards/proposals/parlai/)

2017-07-20: [added building an (seq2seq) agent tutorial](http://www.parl.ai/static/docs/seq2seq_tutorial.html)

2017-07-12: [Several new tasks added: MS Marco, TriviaQA, InsuranceQA, personalized-dialog and MNIST_QA](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/task_list.py)

2017-06-27: [ExecutableWorld class for interactive worlds with dialog](https://github.com/facebookresearch/ParlAI/pull/170)

2017-06-21: [MTurk now supports multiple assignments per HIT](https://github.com/facebookresearch/ParlAI/pull/156)

2017-06-20: [updated MTurk tutorial to reflect new design](http://parl.ai/static/docs/mturk.html)

2017-06-20: [MTurk now uses general world and agent classes](https://github.com/facebookresearch/ParlAI/pull/128)

2017-06-16: [added Creating a New Task tutorial](http://parl.ai/static/docs/task_tutorial.html)

2017-05-31: [added Seq2Seq model](https://github.com/facebookresearch/ParlAI/pull/96)

2017-05-30: [added interactive mode with local human agent](https://github.com/facebookresearch/ParlAI/pull/110)

2017-05-22: [added MTurk tutorial](http://parl.ai/static/docs/mturk.html)

2017-05-14: [added basic tutorial](http://parl.ai/static/docs/basic_tutorial.html)

2017-05-15: ParlAI press: [TechCrunch](https://techcrunch.com/2017/05/15/facebooks-parlai-is-where-researchers-will-push-the-boundaries-of-conversational-ai/), [CNBC](http://www.cnbc.com/2017/05/12/facebook-releases-parlai-to-speed-realistic-chat-bot-development.html), [The Verge](https://www.theverge.com/2017/5/15/15640886/facebook-parlai-chatbot-research-ai-chatbot), [Scientific American](https://www.scientificamerican.com/article/facebook-wants-to-make-chatbots-more-conversational/), [Engadget](https://www.engadget.com/2017/05/15/facebook-parlAI-chatbot-training/), [Venture Beat](https://venturebeat.com/2017/05/15/facebook-to-launch-parlai-a-testing-ground-for-ai-and-bots/), [Wired](https://www.wired.com/2017/05/inside-facebooks-training-ground-making-chatbots-chattier/), [MIT Technology review](https://www.technologyreview.com/s/607854/facebook-wants-to-merge-ai-systems-for-a-smarter-chatbot/).

2017-05-12: [added VQA V2.0 and Visual Dialog V0.9 tasks](https://github.com/facebookresearch/ParlAI/pull/54)

2017-05-01: [ParlAI released!](https://code.facebook.com/posts/266433647155520/parlai-a-new-software-platform-for-dialog-research/)
