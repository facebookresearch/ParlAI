import random


APOLOGIES = [
    'Sorry! ',
    'Oops! ',
    'Oops, my bad. ',
    "Oh, my mistake! ",
]

REQUESTS = [
    "What would have been a better response for me to give?",
    "What would have been a good thing to say instead?",
    "What could I have said instead?",
    "What could I have said that would have made more sense?",
    "What should I have said instead?",
    "Could you tell me what a better response might have been?",
    "Could you give an example of what a better response would have been?",
    "Could you suggest something I could have said instead?",
    "How could I have responded better?",
]

# ACKNOWLEDGMENTS = {
#     ISAID:      "I can be pretty forgetful sometimes. ",
#     NOTSENSE:   "I messed up. ",
#     UM:         "I got confused. ",
#     YOUWHAT:    "I thought we were talking about something else. ",
#     WHATYOU:    "I thought we were talking about something else. ",
#     WHATDO:     "I guess I wasn't as on topic as I thought I was. ",
# }

thank_yous = [
    "Got it! ",
    "Thanks! ",
    "Great--I'll use your feedback to get better. ",
    "Ok, I'll remember that. ",
]

new_topics = [
    "Now can you pick another topic for us to talk about?",
    "How about you get us started on a new topic now?",
    "Let's talk about something else now. What do you want to talk about?",
    "What should we talk about now?",
]


class QuestionGenerator(object):
    pass

class QuestionGeneratorTemplates(QuestionGenerator):
    def generate(self, observation):
        apo = random.choice(APOLOGIES)
        ack = '' # random.choice(ACKNOWLEDMENTS)
        # req = random.choice(REQUESTS)
        req = ("Can you write an example (inside quotation marks) of something I could "
            "have said instead?")
        question = apo + ack + req
        return question