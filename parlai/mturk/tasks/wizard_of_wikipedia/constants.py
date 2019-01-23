# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

MAX_DOC_LEN = 600

WIZARD = 'Wizard'
APPRENTICE = 'Apprentice'
ONBOARD_MSG = '\nWelcome! When you are ready to start your conversation, \
        click the "I am ready, continue" button below\n'
APPRENTICE_START_MSG = '\nSuccessfully matched. \
        Now let\'s talk about something through the chat! \n\
        You need to finish at least <b>{} chat turns</b>, \
        after which you can click the "Done" button to end the chat. '
WIZARD_START_MSG = '\nSuccessfully matched. \
        Now let\'s talk about something through the chat! \n\
        You need to finish at least <b>{} chat turns</b>, \
        after which you can click the "Done" button to end the chat. \n \
        \n <b>You will be given a set of passages relevant to the \
        other person\'s response after each round of dialog </b>\n\
        \n Please base your response on the passages provided, but please \
        <b>do not trivially copy sentences from the passages as your whole \
        response. </b>'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
EXCEED_MIN_TURNS_MSG = '\n {} chat turns finished! \n \
        You can click the "Done" button to end the chat if it\'s your turn \
        or keep chatting.'
UNEXPECTED_DISCONNECTION_MSG = 'The other worker unexpectedly diconnected. \n \
        Please click <span style="color:blue"><b>Done with this HIT</b>\
        </span> button below to finish this HIT.'
CHAT_ENDED_MSG = 'One of you ended the chat. Thanks for your time! \n\
        Please click <span style="color:blue"><b>Done with this HIT</b>\
        </span> button below to finish this HIT.'
PARTNER_RETRIEVED_PASSAGES_INST_MSG = 'Please take a look at the relevant \
        information to your left and \
        check the appropriate sentence before answering, but try not to \
        copy the sentence as your whole response.'
WAITING_MSG = 'Please wait while we match you with another worker...'
EVAL_WIZARD_MSG = 'Thank you for participating in this conversation! \
        Please rate the quality of your conversation with your partner, on a \
        scale from 1 to 5, where 1 is a low quality conversation and 5 is high'
PICK_TOPIC_MSG = 'To start, please select a topic on the left, then click the \
        \'Pick Topic\' button.'
AFTER_PICK_TOPIC_MSG = 'Thank you for selecting a topic! Now, begin the \
        conversation with your partner about the topic.'
AFTER_PICK_TOPIC_WIZARD_MSG = 'Thank you for selecting a topic! Now, begin the \
        conversation with your partner about the topic. You will find some \
        information about the topic to the left of the chat.'
AFTER_PARTNER_PICK_TOPIC_WIZARD_MSG = 'Your partner has selected the topic. \
        Please look to the left to find the relevant information for this topic.'

STOPWORDS = {
    '', 'yourself', 'no', 'them', 'a', 're', 'want', 'have', 'am',
    'hasn', 'off', 'there', 'your', 'will', 'such', 'or', 'ma', 'her', 'their',
    'can', 'himself', 'weren', 'during', 'should', 'shouldn', 'they', 'on',
    'doing', 'once', 'few', 'these', 'again', 'yourselves', 'myself', 'if',
    't', 'didn', 'before', 'needn', 'but', 'does', 've', 'y', '*', 'me',
    'further', 'hers', 'couldn', 'what', 'because', 'down', 'very', 'are',
    'theirs', 'who', 'most', 'don', "'d", 'only', 'is', 'had', 'above', 'so',
    'until', 'hadn', 'ours', 'also', 'ourselves', 'has', 'under', 'his', 'both',
    'than', 'aren', '.', 'by', 'now', "n't", 'really', "'ve", 'did', 'after',
    'you', 'all', 'whom', 'as', 'between', 'an', 'how', 'of', 'it', 'i', 'at',
    'doesn', 'more', 'about', 'from', 'been', 'not', 'each', 'm', "''", 'being',
    'to', 'we', 'own', 'for', 'here', 'while', 'why', 'then', 'this', 'that',
    'those', 'o', 'where', 'having', 'itself', 'him', "'m", 'with', 'mustn',
    'below', 'were', 'ain', 'see', 'and', 'know', 'which', 'through', 'our',
    'some', 'too', "'s", 'he', 'into', 'out', 'when', 'be', '--', 'd', 'wasn',
    '``', 'mightn', '?', 'was', 'wouldn', 'any', 'she', 'do', 'over', 'people',
    'the', 'nor', 'haven', 'won', 'my', 'yours', 'other', 'isn', 'against', 'up',
    'shan', 'herself', 'll', "'ll", ',', 'in', 'just', 'its', "'re", 'same', 's',
    'themselves'
}
