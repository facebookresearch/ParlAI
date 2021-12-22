#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random


"""
Functions to generate safe curated response that repalce an Agent's main response.
"""

####################################################################################
#
#   Init prompt: initial message to start a chit-chat.

STARTING_PROMPT_MESSAGES = (
    "Hey I am here. Let's chat!",
    "Howdy!",
    "Hey. What's up?",
    "Hello there!",
    "Hey, how are you doing today?",
    "Hello, how is your day going so far?",
    "Hi, are you into sports?",
    "Have you seen any interesting movies recently?",
    "Hi. I can't wait to chat with you!",
    "Hi! What's new with you?",
    "Hello, it's a pleasure to meet you.",
)


def generate_init_prompot():
    return random.choice(STARTING_PROMPT_MESSAGES)


####################################################################################
#
# Safe response
#
# A fall back method to avoid discussing sensetive and/or inappropriate topics.
# This may be used to generate a safe response that substitues the main response from an agent,
# in case the main response was flagged as inappropriate.

# The generated responses consist of three parts.
# 1-AVOID: Avoiding to continue the conversation about that topic, for example, "I don't know"
# 2-TEMPLATE: a template to start talking about a new topic, for example, " have you heard about "
# 3-TOPIC: the next topic to steer the conversation towards, for example, "new marvel movie"


AVOID = (
    "I don't know anything about that, sorry.",
    "I am not sure I know about that.",
    "Sorry! I'm confused.",
    "Uh, not sure what you meant.",
    "Whoops, that's not something I know anything about.",
    "I do not know what you mean when you say that.",
    "Huh? I don't get it.",
    "Umm? I don't understand what that means.",
)


TEMPLATE = [
    ("What do you think of", "?"),
    ("Can you help me understand", "?"),
    ("What if we discuss", "?"),
    ("You know what though? It would help me a lot if you could educate me about", "."),
    ("Why don't we talk about", "?"),
    ("You know what I wish I knew more about ---", "."),
    ("Instead, how about you teach me about", "?"),
    ("Could you enlighten me with your thoughts on", "?"),
    ("Let's have a chat about", "."),
    ("It could be interesting to have a conversation on", "."),
    ("I want to hear your opinion on", "."),
    ("Ooh you know what would be fun? Let's discuss", "."),
    ("I recently read an interesting article about", "."),
    ("Guess what? I just learned something new about", "."),
    ("Actually, can we just shoot the breeze about", "."),
    ("How about we discuss", "?"),
    ("Why don't have a conversation about", "?"),
    ("I would rather talk about", "."),
    ("What I really want to talk about is", "."),
    ("Maybe it is better if we discuss", "."),
    ("Can we talk instead about", "?"),
]

TOPIC = (
    "the weather we are getting next weekend",
    "traveling to exotic destinations",
    "the world's best vacation destinations",
    "reading new books",
    "space travel",
    "the possibility of life on Mars",
    "astronomy",
    "how the universe was formed",
    "the moon",
    "the ocean",
    "philosophy",
    "fashion",
    "volcanoes",
    "geology",
    "climate change",
    "artificial intelligence",
    "celebrities",
    "classical music",
    "rock and roll bands",
    "hip-hop music",
    "country music",
    "modern art",
    "money",
    "taking risks",
    "non-fiction books",
    "fiction novels",
    "ancient Egypt",
    "the Ancient Romans",
    "Greek Mythology",
    "medieval castles",
    "dragons",
    "yo-yos",
    "dominoes",
    "blanket and pillow forts",
    "basketball players",
    "football",
    "soccer stars",
    "hockey teams",
    "tennis doubles partners",
    "the economy",
    "broken supply chains",
    "panda bears",
    "current events",
    "middle school cliques",
    "high school sweethearts",
    "the best new phone apps",
    "what the metaverse might be like",
    "giraffes",
    "penguins",
    "walruses",
    "American presidents",
    "Antarctica",
    "the Bermuda Triangle",
    "falling in love",
    "online dating",
    "best friends",
    "poisonous plants",
    "gothic architecture",
    "prehistoric sea creatures",
    "Area 51",
    "robot vacuums",
    "self-driving cars",
    "meditation",
    "how to relax in stressful situations",
    "waking up before it is light out in the morning",
    "talent shows",
    "secret societies",
    "holiday baking",
    "birthday party themes",
    "the best spots to celebrate New Year's Eve",
    "geometry",
    "algebra",
    "New York City",
    "Silicon Valley",
    "Los Angeles",
    "Tokyo",
    "Paris",
    "London",
    "the Arctic Circle",
    "Siberia",
    "the Gobi Desert",
    "the Himalayas",
    "the Andes Mountains",
    "quinoa",
    "kale",
    "green smoothies",
    "peanut allergies",
    "english literature",
    "pie",
    "bagels and the best types of schmear",
    "cake",
    "cookies",
    "ice cream",
    "pastries",
    "breakfast cereal",
    "reality tv",
    "elephants",
    "polar bears",
    "good investment strategies",
    "zoos",
    "dinosaurs",
    "archeologists",
    "kittens",
    "puppies",
    "the Scottish Highlands",
    "the Loch Ness monster",
    "aliens",
    "giant squids",
    "charities",
    "things that make people happy",
    "things that interrupt a good night's sleep",
    "cryptocurrency",
    "the inevitable heat death of our Universe",
    "unusual hobbies",
    "superheroes",
    "home improvement tv shows",
    "hidden super-powers",
)


def generate_safe_response():
    avoid = random.choice(AVOID)
    template, punctuation = random.choice(TEMPLATE)
    topic = random.choice(TOPIC)
    sentence = " ".join([avoid, template, topic]) + punctuation
    return sentence
