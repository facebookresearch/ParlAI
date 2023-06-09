# BlenderBot 3x 175B data card

## Motivation

**For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.** The collected demo deploy data was used to create an improved iteration of BlenderBot. This dataset provides organic user interaction data, which is sparingly available in the wild.

**Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?** Meta AI

**Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.** Meta AI

**Any other comments?** N/A

## Composition

**What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.** The instances are text-based conversational dialogues.

**How many instances are there in total (of each type, if appropriate)?** The data contains ~261k conversations, comprising ~5.9M utterances.

**Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).** This data includes conversations from the bb3 deployement between Aug 3 2022 - Feb 28th 2023 which users approved to having saved for research purposes. Users were also able to opt to chat without their conversation being saved. Conversations with no human response were also filtered out.

**What data does each instance consist of? "Raw" data (e.g., unprocessed text or images) or features? In either case, please provide a description.** Each data instance represents one conversation with a user. Each instance has metadata about the conversation as well as 'message_history' attribute which is a list of objects where each object is one message and contained metadata about that message (liked, id, dislike_type, etc.) as well as the message text. When users opted to not disclose their conversation, 'message_history' would be empty, but we have only shared data for those conversations that have message information. Below you can find details on all the data attributes.
| Property Name | Type | Path (from conversation root) | Description |
| ----------------------------- | ------------------------------------------------------------------ | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| | | | |
| Conversation Level Data | | | |
| user_pseudo_id | String | .user_pseudo_id | UUID for user. |
| chat_id | String | .chat_id | UUID for chat. |
| message_history | Dict[] | .message_history | A list of objects, each describing an utterance in the conversation and the associated information. See below for a detailed description on attributes in these objects. |
| Message Level Data | | | |
| text | String | .message_history[i].text | Text of message. |
| sender | "Chatbot" &#124; "Human" | .message_history[i].sender | Who sent the message. |
| | | | |
| *Bot Message Specific* | | | |
| bot_message_id | String | .message_history[i].human_message_id | UUID for message. |
| is_liked | Boolean | .message_history[i].isLiked | If user liked message. |
| is_disliked | Boolean | .message_history[i].isDisliked | If user disliked message. |
| dislike_type | String (See Dislike Type Sheet) | .message_history[i].dislike_type | If message disliked. This will show the user selected reason for the dislike. 'agree_sensitive' and 'disagree_sensitive' only possible on canned responses in response to human triggering safety measure. |
| memories | String[] | .message_history[i].memories | current set of memories including those generated on this bot message. format = ['person: memory']. Person being either 'Person 1's Persona', which is the bot, or 'Person 2's Persona' which is the human. |
| memory_decision | "access memory" &#124; "do not access memory" | .message_history[i].memory_decision | Decision on if to access memory. Output of Long-term memory access decision module. |
| memory_knowledge | String | .message_history[i].memory_knowledge | Memory which was used. Output of Access long-term memory module. |
| search_decision | "search" &#124; "do not search" | .message_history[i].search_decision | Decision on if to use internet search. Output of Internet search decision module. |
| search_query | String | .message_history[i].search_query | Search query used. Output of Generate internet search query module |
| search_knowledge | String | .message_history[i].search_knowledge | Output of Generate knowledge response module. |
| search_knowledge_doc_titles | String[] | .message_history[i].search_knowledge_doc_titles | Titles of the 5 documents outputted from search engine and fed to Generate knowledge response module. |
| search_knowledge_doc_content | String[] | .message_history[i].search_knowledge_doc_content | Content of the 5 documents outputted from search engine and fed to Generate knowledge response module. |
| search_knowledge_doc_urls | String[] | .message_history[i].search_knowledge_doc_urls | URLs of the 5 documents outputted from search engine and fed to Generate knowledge response module. |
| is_safety_controlled_response | Boolean | .message_history[i].isSafetyControlledResponse | If message is a canned response due to trigger from any of our safety measures (classifiers + string matchers). |
| previous_interaction_failure | "unsafe_Human" &#124; "unsafe_BlenderBot3" &#124; "unsafe_model_crash" | .message_history[i].previous_interaction_failure | When is_safety_controlled_response is true, this will specify is triggered by unsafe human text (unsafe_Human) or unsafe prospective bot message (unsafe_BlenderBot3). When model crash, this will show 'unsafe_model_crash' |
| from_model_crash | Boolean | .message_history[i].from_model_crash | If canned message due to model crash. |
| | | | |
| *Human Message Specific* | | | |
| human_message_id | String | .message_history[i].human_message_id | UUID for message. |
| is_dislike_feedback | Boolean | .message_history[i].isDislikeFeedback | If this is the human message after bot prompts for feedback after a dislike. |
| | | | |
| Additional Annotation | | | |
| Safety | Dict | .message_history[i].safety | Output of classifier trained to detect offensive language in the context of single-turn dialogue utterances (More info here [http://parl.ai/projects/dialogue_safety/.](http://parl.ai/projects/dialogue_safety/) Not the same classifier used in the demo.) as well as a output from a string matcher for offensive language. format = {'duo_safety': output, ' string_matcher': output} |
| Reward | Dict | .message_history[i].reward | Output of reward model. format = 'Predicted class: class(*\_\_ok\_\_* or *\_\_not_ok\_\_*) with probability: _number_' |
| Mturk | Dict | .message_history[i].mturk | Crowdworker annotations. format = {annotation_type: *number*, ... }

You can access the public data via ParlAI as described in the project page (https://parl.ai/projects/bb3x).

**Is there a label or target associated with each instance? If so, please provide a description.** The target for each instance is related to the module for which the instance is being used; each module has its own target. E.g., for the "search decision" module, targets are either "do search" or "do not search". For dialogue response modules, the targets are simply the correct utterance.

**Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.** We have redacted personal identifiable information (PII) from this data. In place of the redacted text will appear a non decodable hash. For some numerical pieces of PII (address numbers, SSNs, credit card numbers, etc.) we randomized the digits. We used heuristics to ensure that non-PII information was still retained, e.g., not replacing famous and well-known entities.

**Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)? If so, please describe how these relationships are made explicit.** The only relationship that can be made between these instances is via the user_id. This is a UUID assigned to each user. It is stored as a cookie so can occasionally be destroyed and reset for a user. Multiple conversations can have the same user_id, meaning they are from the same user. For returning users, you will notice the long term memory starts populated with memories from previous conversations.

**Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.** We randomly selected from the whole deployment a validation and test set.

**Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.** We do not add any redundancy, beyond the reuse of source contexts for different downstream targets.

**Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** The data is self-contained.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.** Since this data is from a public deployment, the data may contain offensive or inappropriate messages.

**Does the dataset relate to people? If not, you may skip the remaining questions in this section.** No.

**Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.** No.

**Any other comments?** No.

## Collection Process

**How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.** We collect conversations from organic interactions with our bot in a web deployment environment. Some of the messages have additional annotations from crowdworkers.

**What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?** We collect conversations via a web interface.

**If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?** N/A

**Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?** For the demo deployment data, we collect data from organic, unpaid users who choose to interact with our chatbot. Some of this data has additional annotations by paid crowdworkers.

**Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.** This data consists of conversations that took place on our web application between Aug 3 2022 and Feb 28th 2023.

**Does the dataset relate to people? If not, you may skip the remainder of the questions in this section.** No.

**Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?** N/A

**Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.** N/A

**Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.** N/A

**If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).** N/A

**Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.** N/A

**Any other comments?** N/A

## Preprocessing/cleaning/labeling

**Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.** We annotated the deploy data with a safety classifier, reward classifier, and using crowdworkers. Refer to the data details above and BlenderBot 3x paper for more information.

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the "raw" data.** N/A

**Any other comments?** No.

## Uses

**Has the dataset been used for any tasks already? If so, please provide a description.** These datasets have been used for training BlenderBot 3x.

**Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.** See https://parl.ai/projects/bb3x for the project page.

**What (other) tasks could the dataset be used for?** This data could be used to train other conversational models in the future.

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?** No.

**Are there tasks for which the dataset should not be used? If so, please provide a description.** These datasets should not be used for a bot that will knowingly cause harm or engage in trolling behavior.

**Any other comments?** No.

## Distribution

**Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.** The dataset from the chatbot demo will be made publicly available, and contributors to these datasets have consented to their release.

**How will the dataset be distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?** The chatbot demo datasets will be hosted on an AWS server and downloaded through the ParlAI framework.

**When will the dataset be distributed?** This dataset is currently available.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.** The chatbot demo dataset will be distributed under the CC BY 4.0 license, at https://creativecommons.org/licenses/by/4.0/ .

**Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.** Not in addition to the above.

**Any other comments?** No.

## Maintenance

**Who is supporting/hosting/maintaining the dataset?** Meta AI.

**How can the owner/curator/manager of the dataset be contacted (e.g., email address)?** By filing an issue on ParlAI github.

**Is there an erratum? If so, please provide a link or other access point.** N/A

**Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?** To be determined when / if we will update this data.

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.** Individuals were able to choose at time of data collection whether to allow open release of their (anonymized) conversations.

**Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to users.** Older versions of the dataset will continue to be hosted, but will be superseded by newer ones by an incremented version number, allowing for automatic downloading of updated versions of the dataset.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.** N/A

**Any other comments?** No.

## License
Please see the LICENSE file [here](https://github.com/facebookresearch/ParlAI/blob/main/LICENSE).
