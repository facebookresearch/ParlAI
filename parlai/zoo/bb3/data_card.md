# BlenderBot 3 175B data card

## Motivation	

**For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.** The several datasets used for fine-tuning BlenderBot 3 were created with various specific tasks in mind. The majority of the tasks are crowdsourced dialogue datasets designed to inform conversational models about how to handle skills within conversation (grounding responses on knowledge, displaying empathy and personality, roleplaying as a character). Other datasets are from the task-oriented domain, in which models must learn to complete tasks requested by humans. Finally, we include question answering data, which is meant to teach models how to answer factual questions. The collected demo deploy data will be used to continually improve future iterations of BlenderBot. This dataset provides organic user interaction data, which is sparingly available in the wild.

**Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?** Meta AI

**Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.** Meta AI

**Any other comments?** N/A

	
## Composition	

**What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.** The instances are text-based conversational dialogues or question/answer pairs. The BB3 fine-tuning data comprise the following datasets:

- *Question Answering:*
  - MS MARCO (Nguyen et al., 2016)
  - SQuAD (Rajpurkar et al., 2016)
  - TriviaQA (Joshi et al., 2017)
  - Natural Questions (Kwiatkowski et al., 2019) 
  - Natural Questions (Open) (Lee et al., 2019) 
  - Natural Questions (Open Dialogues) (Adolphs et al., 2021) 
- *Knowledge-Grounded Dialogue:*
  - Wizard of the Internet (Komeili et al., 2022)
  - Wizard of Wikipedia (Dinan et al., 2019b)
  - Funpedia (Dinan et al., 2020b) 
- *Open-Domain Dialogue:*
  - PersonaChat (Zhang et al., 2018)
  - Empathetic Dialogues (Rashkin et al., 2019)
  - Blended Skill Talk (Smith et al., 2020)
  - Multi-Session Chat (Xu et al., 2022a)
  - LIGHT + WILD (Urbanek et al., 2019; Shuster et al., 2021b) 
- *Recovery & Feedback:*
  - SaFeRDialogues (Ung et al., 2022) 
  - FITS (Xu et al., 2022b)
- *Task-Oriented Dialogue:*
  - Google SGD (Rastogi et al., 2020) 
  - Taskmaster (Byrne et al., 2019) 
  - Taskmaster 2 (Byrne et al., 2019) 
  - Taskmaster 3 (Byrne et al., 2019) 

We additionally make use of a shard of OPT pre-training data: see the data card of [Zhang et al. (2022)](https://arxiv.org/pdf/2205.01068.pdf) for more details.

**How many instances are there in total (of each type, if appropriate)?** The training data contains ~4.5M examples, comprising ~1.3B training tokens.

**Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).** The fine-tuning data contains a large subset of the respective datasets, filtered to contain examples for which we have proper annotations for training each of BB3's modules. As for the pre-train data from OPT, we include a small subset (~170M tokens) randomly sampled from a shard of the data.

**What data does each instance consist of? "Raw" data (e.g., unprocessed text or images) or features? In either case, please provide a description.** All data are raw text.

**Is there a label or target associated with each instance? If so, please provide a description.** The target for each instance is related to the module for which the instance is being used; each module has its own target. E.g., for the "search decision" module, targets are either "do search" or "do not search". For dialogue response modules, the targets are simply the correct utterance.

**Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.** No.

**Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)? If so, please describe how these relationships are made explicit.** There are no relationships between instances.

**Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.** Yes: the train splits are sourced from the train splits from the respective data, as are the validation and testing splits.

**Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.** We do not add any redundancy, beyond the reuse of source contexts for different downstream targets.

**Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** The data is self-contained.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.** To the best of our knowledge, no. However, crowd-sourced data may be prone to human bias of crowdworkers.

**Does the dataset relate to people? If not, you may skip the remaining questions in this section.** No.

**Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.** No.

**Any other comments?** No.

	
## Collection Process	

**How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.** For the fine-tuning data, we refer the reader to the various dataset papers for data instance collection details. For the demo deployment data, we collect conversations from organic interactions with our bot.

**What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?** For the fine-tuning data, we refer the reader to the various dataset papers for data instance collection details. For the demo deployment data, we collect conversations via a web interface.

**If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?** N/A

**Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?** For the fine-tuning data, we refer the reader to the various dataset papers for data instance collection details. For the demo deployment data, we collect data from organic, unpaid users who choose to interact with our chatbot.

**Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.** For the fine-tuning data, we refer the reader to the various dataset papers for data instance collection details.

**Does the dataset relate to people? If not, you may skip the remainder of the questions in this section.** No.

**Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?** N/A

**Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.** N/A

**Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.** N/A

**If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).** N/A

**Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.** N/A

**Any other comments?** N/A

	
## Preprocessing/cleaning/labeling	

**Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.** For the fine-tuning data, we process the data to reassign targets depending on the modular function for which we were training BB3. For the demo deployment data, we refer to the main BlenderBot 3 paper for release details.

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the "raw" data.** N/A

**Any other comments?** No.

	
## Uses	

**Has the dataset been used for any tasks already? If so, please provide a description.** These datasets have been used for training BlenderBot 3.

**Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.** See https://parl.ai/projects/bb3 for the project page for this model and https://github.com/facebookresearch/ParlAI/blob/main/parlai/zoo/bb3/model_card.md for the model card.

**What (other) tasks could the dataset be used for?** This data could be used to train other conversational models in the future.

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?** No.

**Are there tasks for which the dataset should not be used? If so, please provide a description.** These datasets should not be used for a bot that will knowingly cause harm or engage in trolling behavior.

**Any other comments?** No.

	
## Distribution	

**Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.** The dataset from the chatbot demo will be made publicly available, and contributors to these datasets have consented to their release.

**How will the dataset will be distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?** The chatbot demo datasets will be hosted on an AWS server and downloaded through the ParlAI framework.

**When will the dataset be distributed?** For the fine-tuning data, these datasets are available through ParlAI. For the demo deployment data, we will be periodically releasing updated versions of this data.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.** The chatbot demo dataset will be distributed under the CC BY 4.0 license, at https://creativecommons.org/licenses/by/4.0/ .

**Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.** Not in addition to the above.

**Any other comments?** No.

	
## Maintenance	

**Who is supporting/hosting/maintaining the dataset?** Meta AI.

**How can the owner/curator/manager of the dataset be contacted (e.g., email address)?** blenderbotdemo@fb.com

**Is there an erratum? If so, please provide a link or other access point.** N/A

**Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?** We plan to release anonymized data collected from deployment periodically, given that this bot will engage in continual learning.

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.** Individuals were able to choose at time of data collection whether to allow open release of their (anonymized) conversations.

**Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to users.** Older versions of the dataset will continue to be hosted, but will be superseded by newer ones by an incremented version number, allowing for automatic downloading of updated versions of the dataset.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.** N/A

**Any other comments?** No.
