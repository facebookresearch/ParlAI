# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This preprocessing script was used to create ParlAI's version of the
data. It was run in a script called parse.py inside ijcnlp_dailydialog/ after
uncompressing the original directory and all subdirectories.
"""

import json
import os
import re
from collections import Counter

BADSENT = r'\.(\w)'

# set each of these and rerun the script
FOLD = "validation"
FOLD_OUT = "valid"
# FOLD = "train"
# FOLD_OUT = "train"
# FOLD = "test"
# FOLD_OUT = "test"

ACTS = [
    'no_act',
    'inform',
    'question',
    'directive',
    'commissive',
]

EMOTIONS = [
    'no_emotion',
    'anger',
    'disgust',
    'fear',
    'happiness',
    'sadness',
    'surprise',
]

TOPICS = [
    'no_topic',
    'ordinary_life',
    'school_life',
    'culture_and_educastion',
    'attitude_and_emotion',
    'relationship',
    'tourism',
    'health',
    'work',
    'politics',
    'finance',
]

ALL_COUNTS = Counter()


def cleanup_text(text):
    text = text.strip()

    # Prefer non-unicode special character
    SWITCH_LIST = [
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u201d", '"'),
        ("\u201c", '"'),
        ("\u2014", "--"),
        ("\u2013", "--"),
        ("\u3002", ". "),
        ("\u2032", "'"),
        ("\u3001", ", "),
    ]
    for before, after in SWITCH_LIST:
        text = text.replace(before, after)

    # fix some broken sentence tokenization
    text = re.sub(BADSENT, r' . \1', text)

    ALL_COUNTS.update([t for t in text.split() if len(t) == 1 and ord(t) > 127])
    return text.strip()


f_acts = open(os.path.join(FOLD, "dialogues_act_" + FOLD + ".txt"))
f_emotions = open(os.path.join(FOLD, "dialogues_emotion_" + FOLD + ".txt"))
f_texts = open(os.path.join(FOLD, "dialogues_" + FOLD + ".txt"))

topic_map = {}
with open('topicmap') as f_tm:
    for line in f_tm:
        text, topic = line.strip().split("\t")
        topic_map[text] = TOPICS[int(topic)]


f_out = open('out/' + FOLD_OUT + '.json', 'w')

for acts, emotions, raw_text in zip(f_acts, f_emotions, f_texts):
    acts = [ACTS[int(x)] for x in acts.strip().split()]
    emotions = [EMOTIONS[int(x)] for x in emotions.strip().split()]
    raw_text = raw_text.strip()
    texts = raw_text.split("__eou__")[:-1]
    assert len(acts) == len(emotions)
    assert len(texts) == len(emotions)
    # fix one topic lookup bug
    raw_text = raw_text.replace(
        "one here as well . I've been using",
        "one here as well . __eou__ I've been using"
    )
    if raw_text not in topic_map:
        continue
    record = {
        'fold': FOLD,
        'topic': topic_map[raw_text],
        'dialogue': [
            {
                'emotion': e,
                'act': a,
                'text': cleanup_text(t),
            }
            for e, a, t in zip(emotions, acts, texts)
        ],
    }
    outline = json.dumps(record)
    f_out.write(outline + '\n')

f_out.close()
