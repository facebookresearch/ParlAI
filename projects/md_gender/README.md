## Multi-Dimensional Gender Bias Classification

Emily Dinan, Angela Fan, Ledell Wu, Jason Weston, Douwe Kiela, Adina Williams

## Abstract

Machine learning models are trained to find patterns in data. NLP models can inadvertently learn socially undesirable patterns when training on gender biased text. In this work, we propose a general framework that decomposes gender bias in text along several pragmatic and semantic dimensions: bias from the gender of the person being spoken about, bias from the gender of the person being spoken to, and bias from the gender of the speaker. Using this fine-grained framework, we automatically annotate eight large scale datasets with gender information. In addition, we collect a novel, crowdsourced evaluation benchmark of utterance-level gender rewrites. Distinguishing between gender bias along multiple dimensions is important, as it enables us to train finer-grained gender bias classifiers. We show our classifiers prove valuable for a variety of important applications, such as controlling for gender bias in generative models, detecting gender bias in arbitrary text, and shed light on offensive language in terms of genderedness.

## Paper

[Link](https://arxiv.org/abs/2005.00614) to appear at EMNLP 2020.

## Data

Data can be found in `parlai/tasks/md_gender`.

The following tasks are available right now:
- `md_gender:convai2` (uses the ConvAI2 task)
- `md_gender:funpedia` (uses the Funpedia task)
- `md_gender:image_chat` (uses the Image Chat task)
- `md_gender:light` (uses the LIGHT task)
- `md_gender:md_gender` (new evaluation dataset)
- `md_gender:wikipedia` (uses the Wikipedia task)
- `md_gender:wizard` (uses the Wizard of Wikipedia task)
- `md_gender:yelp` (uses data from Yelp)

View any of the tasks above with the following command:
```
parlai dd -t <taskname>
```


NOTE: data for the Opensubtitles tasks has yet to be released; data will be coming soon.

## Models

Try interacting the the multi-tasked classifier by running the following command:
```
parlai interactive -t md_gender -m projects.md_gender.bert_ranker_classifier.agents:BertRankerClassifierAgent -mf zoo:md_gender/model -ecands inline -cands inline --interactive_mode False --data-parallel False
```