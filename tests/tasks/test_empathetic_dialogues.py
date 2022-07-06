#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from parlai.core.opt import Opt
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.utils import testing as testing_utils


EPISODE_COUNTS = {
    'train_experiencer_only': 19531,
    'train_both_sides': 39057,
    'valid': 2769,
    'test': 2547,
}

EXAMPLE_COUNTS = {
    'train_experiencer_only': 40254,
    'train_both_sides': 64636,
    'valid': 5738,
    'test': 5259,
}


class TestEDTeacher(unittest.TestCase):
    """
    Basic tests to count the number of examples/episodes and to check a few utterances.

    # Counting num episodes (from the original internal copy of the data)
    cat /checkpoint/parlai/tasks/empathetic_dialogues/train.csv | grep -E '^hit:[0-9]+_conv:[0-9]+,2' | wc  # 19531
    cat /checkpoint/parlai/tasks/empathetic_dialogues/train.csv | grep -E '^hit:[0-9]+_conv:[0-9]+,(2|3)' | wc  # 39057
    cat /checkpoint/parlai/tasks/empathetic_dialogues/valid_random_cands.csv | grep -E '^hit:[0-9]+_conv:[0-9]+,2' | wc  # 2769
    cat /checkpoint/parlai/tasks/empathetic_dialogues/test_random_cands.csv | grep -E '^hit:[0-9]+_conv:[0-9]+,2' | wc  # 2547
    # We count the number of lines with turn_idx=2 because this means that we have at
    # least one full utterance in the conversation. For train_experiencer_only==False,
    # we also include turn_idx=3 to count the Listener-based conversations in the same
    # manner.

    # Count num examples (from the original internal copy of the data)
    grep -E 'hit:[0-9]+_conv:[0-9]+,(2|4|6|8|10|12),' /checkpoint/parlai/tasks/empathetic_dialogues/train.csv | wc  # 40254
    grep -E 'hit:[0-9]+_conv:[0-9]+,(2|3|4|5|6|7|8|9|10|11|12),' /checkpoint/parlai/tasks/empathetic_dialogues/train.csv | wc  # 64636 (--train-experiencer-only False)
    grep -E 'hit:[0-9]+_conv:[0-9]+,(2|4|6|8|10|12),' /checkpoint/parlai/tasks/empathetic_dialogues/valid_random_cands.csv | wc  # 5738
    grep -E 'hit:[0-9]+_conv:[0-9]+,(2|4|6|8|10|12),' /checkpoint/parlai/tasks/empathetic_dialogues/test_random_cands.csv | wc  # 5259
    """

    def test_counts(self):

        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir

            # Check EmpatheticDialoguesTeacher, with multiple examples per episode
            opts_episodes_and_examples = [
                (
                    {'datatype': 'train'},
                    EPISODE_COUNTS['train_both_sides'],
                    EXAMPLE_COUNTS['train_both_sides'],
                ),  # Test the default mode
                (
                    {'datatype': 'train', 'train_experiencer_only': True},
                    EPISODE_COUNTS['train_experiencer_only'],
                    EXAMPLE_COUNTS['train_experiencer_only'],
                ),
                (
                    {'datatype': 'train', 'train_experiencer_only': False},
                    EPISODE_COUNTS['train_both_sides'],
                    EXAMPLE_COUNTS['train_both_sides'],
                ),
                (
                    {'datatype': 'valid'},
                    EPISODE_COUNTS['valid'],
                    EXAMPLE_COUNTS['valid'],
                ),
                ({'datatype': 'test'}, EPISODE_COUNTS['test'], EXAMPLE_COUNTS['test']),
            ]
            for teacher_class in [EmpatheticDialoguesTeacher]:
                for opt, num_episodes, num_examples in opts_episodes_and_examples:
                    full_opt = Opt({**opt, 'datapath': data_path})
                    teacher = teacher_class(full_opt)
                    self.assertEqual(teacher.num_episodes(), num_episodes)
                    self.assertEqual(teacher.num_examples(), num_examples)

    def test_check_examples(self):

        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir

            # Check EmpatheticDialoguesTeacher
            opts_and_examples = [
                (
                    {'datatype': 'train', 'train_experiencer_only': True},
                    {
                        'situation': ' i used to scare for darkness',
                        'emotion': 'afraid',
                        'text': 'dont you feel so.. its a wonder ',
                        'labels': [
                            'I do actually hit blank walls a lot of times but i get by'
                        ],
                        'episode_done': False,
                    },
                ),
                (
                    {'datatype': 'train', 'train_experiencer_only': False},
                    {
                        'situation': 'I remember going to the fireworks with my best friend. There was a lot of people, but it only felt like us in the world.',
                        'emotion': 'sentimental',
                        'text': 'Where has she gone?',
                        'labels': ['We no longer talk.'],
                        'episode_done': True,
                    },
                ),
                (
                    {'datatype': 'valid'},
                    {
                        'situation': 'I was walking through my hallway a few week ago, and my son was hiding under the table and grabbed my ankle. I thought i was got. ',
                        'emotion': 'surprised',
                        'text': 'I may have let out a scream that will have him question my manhood for the rest of our lives, lol. ',
                        'labels': ['I would probably scream also.'],
                        'episode_done': True,
                        'label_candidates': [
                            "That really does make it special. I'm glad you have that. ",
                            "It must've have been. Glad they are okay now.",
                            "Well sometimes companies make mistakes. I doubt it's anything you did.",
                            "Oh no, I'm so so sorry. I've always had at least one pet throughout my life, and they're truly part of the family.",
                            'Wow. That must suck. Do you like the band incubus? I missed them a couple of times but I saw them this year',
                            "I can't play those kinds of games. Too spooky for me.",
                            'I think your boss should give you more recognition in that case!',
                            "That's always a good thing. It means you should get on great with your neighbors.",
                            "Yeah, I had my Commodore 64 and Amiga in the late 80's. Still, the games were great when they worked!",
                            "That's ok, you did the right thing. It probably happens to lots of people.",
                            "That's good. Now you don't have to worry about it.",
                            'Hopefully one day you will be willing to explore a relationship in a serious way.',
                            "I'm sorry, things will get better.",
                            'Oh, okay. Maybe you should ask your teacher for some extra help or find a study buddy. i hope you do better next time.',
                            'Why? What did she do?',
                            'I do enjoy the zoo and the animals. I think they could be just as good.',
                            'Well at least you managed to save him!',
                            'That sucks, how much is it?',
                            'Yeah, that is a hard one to deal with.  Maybe you should give it back so you will not feel bad about yourself.',
                            'HAve you been practicing? Do you have note cards?',
                            "That's good news at least. I hope you are feeling better now. And don't be hard on yourself, accidents happen.",
                            'Oops. I hate when that happens. Did they say anything to you?',
                            'no its not',
                            'Yes, my friends are coming with me. :)',
                            "Oh my gosh! I'm sorry! haha Thats funny. Atleast you have them a story to always remember.;)",
                            'I am so happy for you! All of your hard work paid off!',
                            'Wow, thats a nice car',
                            "Does it make you feel like you're living in an alternate reality?",
                            'glade all was well',
                            'ah, crisis averted! that could have been a lot worse',
                            "Maybe if we weren't so attached to being creatures of comfort. Some things we just can't let go of, wouldn't exist without some poor shmuck having to do the dirty work. I guess we're all that shmuck to someone, someway or another.",
                            "That's awesome! You're going to be rolling in the dough with those skills",
                            "Don't worry, from what you said it doesn't sound like you almost ruined it. It wasn't something on purpose at least.",
                            'Have you tried yoga? It can help in the meanwhile till you get a proper vacation.',
                            "I wish my insurance would give me something like that! It's good to go anyways.",
                            'I bet you are pretty anxious and excited at the same time.',
                            'Do you honk at them?',
                            "That's a bad supervisor. Did you call him/her out on it?",
                            "Geniuses don't do chores my friend.",
                            'Which country? that sounds fun, are you guys doing anything fun there?',
                            'oh that is so exciting!!! good for you man!',
                            'Wow! Any way they can get out? Did they call someone?',
                            'I love that nostalgic feeling. ',
                            'Congratulations. You have done great!',
                            'hahaha I definitely admire your courage to have done that.',
                            'wait til she leaves and continue',
                            'I do too.  I am so sorry you are going through this',
                            'That is awesome. Congratulations. Im sure you earned every penny.',
                            'I want some of whatever you had for breakfast. You seem very happy.',
                            'Oh wow! I am so sorry that happened to you.',
                            'Well, hopefully there will be nothing but great things for him in his future.',
                            'Oh that was so nice of them! I bet you were relieved!',
                            'how was it ?',
                            "Nice! Why do you like it more than other places you've lived?",
                            'It must be difficult, do you think she will come back ever?',
                            "That's so messed up! Why was he doing that?",
                            'Did you try therapy at all or counseling?',
                            'Did you reconnect and promise to keep in touch?',
                            'I am so sorry for you. Perhaps you can hang with her after your workdays?',
                            "That's good that you found happiness. That's what were all in the pursuit of right?",
                            'I hope these feelings last for you!',
                            'you have eyes too',
                            "Wow, that's rude! He won't last long...",
                            "Hopefully the person learned what they had to do so they don't hold the line up in the future.",
                            'Oh no that must have been a terrible experience, I hope no one got hurt from the shattered glass.',
                            "Congrats!, I'm sure you must be very happy!",
                            "That's good to know!  You all have a lot to be thankful for!",
                            'It depends, if you love her, you could try to work it out.  Or you could cheat on her too',
                            "I'm sorry to hear that, I'm pretty terrified of the dentist myself. Ask for gas! Good luck, I'm sure everything will be just fine.",
                            'That makes sense, you are a good older sibling!',
                            'They say dogs are mans best friend. ',
                            'I would probably scream also.',
                            'Well I hope he gets to felling better.',
                            "If I had a bag of M&M's right now I would eat them for sure!",
                            'Yep. Happy and healthy is a blessing',
                            'Wow was it a scam or was it legit?',
                            'that is good to hear, it was a motivation to succeed, a great character building ',
                            'Its not time to get over it, you arent doing any wrong its okay to "feel" things. I hope people around you give you a lot of love! ',
                            'Awww. Did you keep it?',
                            "Oh I see.. Well that's a pretty positive talent then, huh? Maybe you should encourage him to keep doing it. Maybe he misses it. You could get him a present for his birthday or Christmas that was related to drawing tools/pencils and all that.",
                            "You learn a lot about someone when you move in with them, so if you feel comfortable in your relationship I think that's actually rather prudent.",
                            'That is terrible. How long have you had this pet?',
                            'Oh that sucks...did she explain herself yet',
                            "8 Miles!? That's very impressive.  I bet I could barely make it a mile!",
                            'That stuff is pretty expensive. Maybe you can sell it on eBay or something.',
                            'Its horrible to have to got through things like thaty',
                            'Oh god.. so sorry to hear that.. May i ask how did Tom pass?',
                            'Like a paranormal type fear or a human with intent to harm type fear?',
                            'I bet you cant wait.  WHere are going for your vacation?',
                            'Aw, that sucks. Did you give her a proper burial?',
                            'Awesome! What are you going to see?',
                            'What kind of food does it serve? Sounds wonderful!',
                            "Oh no! What's wrong with your dad?",
                            'oh god yes i know what you mean, any ideas what you wanna do ?',
                            "Hopefully you'll able to get it all sorted out soon.  I'm sure when it's done it'll be a beautiful house.",
                            'That would be bad you should double check before you leave',
                            'I hope he continues to do well.',
                            "You can only do so much.  Next time I'd just let him drink on his own.",
                            'I am sure you will meet them',
                            'Wow thats nice.  What do you drive?',
                        ],
                    },
                ),
                (
                    {'datatype': 'test'},
                    {
                        'situation': 'My mother stopped by my house one day and said she saw 3 dogs on the road, down from our house. They were starving, with ribs showing, and it was a mother dog and her two small puppies. Of course, my daughter wanted to bring them to our house, so we could feed and help them. We did, and my heart went out to them, as they were so sweet, but really were in a bad shape.',
                        'emotion': 'caring',
                        'text': "Oh my goodness, that's very scary! I hope you are okay now and the drunk driver was punished for his actions?",
                        'labels': ['Yeah he was punished hes in jail still'],
                        'episode_done': True,
                        'label_candidates': [
                            "Are you really able to work from home? Finding a gig is so difficult, I'm glad that it is working for you.",
                            "Oh no. That's quite unfortunate for the deer. Did you just drive past it?",
                            'Wow, you must have felt jealous',
                            'I can only imagine! How is he now?',
                            'Oh goodness, what happened for the past 3 weeks?',
                            'LOL i hate that',
                            'I love a warm fire outside while camping! Sounds like a great time.',
                            'Yeah he was punished hes in jail still',
                            'Was he upset?',
                            "Wow that's awesome! Are you on a team?",
                            'Oh man that is just crazy! Feel bad for the person who has to clean it.',
                            'im sorry, thats awful. its a shame his parents arent being more supportive',
                            'I bet that was scary. Did he surprise you with something?',
                            'That sounds pretty stressful. Are you moving soon?',
                            "Well, if I were you, I'd keep it up, whether or not my spouse laughed at me, or a new girlfriend/boyfriend, whatever. It's not childish to me. Life is stressful enough. Let us snuggle what we want.",
                            "That's hilarious! Is he usually down for a good prank?",
                            'Oh I love seeing kids achieve things! Adorable. Good for her! :) ',
                            'that makes two of us! i am terrified of all snakes',
                            'that is dangerous, glad that both of your are okay',
                            "That's good to hear. I hope I meet someone that will do that for me.",
                            "Well that's good.",
                            'We need more people like you in the world.  Theres always someone out there who needs a helping hand and could use a friend.',
                            'How ever so exciting! is this your first cruise?',
                            'Do you feel any less nervous?  Job interviews are always nerve-wracking ',
                            'Maybe you could try to better that?',
                            "That's what matters most, that you had a good time and made memories!",
                            'Oh man! I hear you. I rescue animals and it is VERY hard to potty train them!',
                            'Hopefully they will give him a better shift.',
                            "That's a big step. I hope it works out for you.",
                            "Hiking is probably a tough environment to meet people! LA is real nice, but I hear the people there aren't/",
                            'I hope things turn out better for you. Keep fighting.',
                            "please don't lol i'm a man, I appreciate what you women go through when you're pregnant or on your period but i'm okay with not knowing details",
                            'I wish refrigerators would have a warranty that replaced food when they went bad. ',
                            'Seeing old friends that you have not contacted in so long is a nice feeling. ',
                            'Cool. Will you leave it there forever?',
                            'Oh wow. How far away did you move??',
                            'So inconsiderate! It reminds me of my neighbours doing building work one morning at 6AM!',
                            'Oh no, did they do something embarrasing?',
                            'That is awesome!  Is there a particular reason you are so happy?',
                            'Did you buy all the essential items the dog will need? ',
                            'Fantastic, now do you have a job lined up?',
                            'Better luck next time!  I love to scratch!',
                            'Thats neat. Do you guys make a lot of money?',
                            'I would be furious.  What did you do?',
                            "Well hopefully you're able to familiarize yourself quickly. Good luck!",
                            'Oh thats good for your friend, but it sounds like you really would like to live there! I can imagine feeling jealous',
                            "That's unfortunate.  What are you doing now?",
                            'Oh no. I rent also so I know your pain. My last landlord was awful.  How did your landlord react?',
                            'Im sorry to hear that, how long did you have him?',
                            'Lovely.  What did you do together?',
                            'Have you thought about getting another dog? ',
                            'Oh yeah?  Do you still look awesome like you did back then?',
                            'Do you dress up when the new movies come out also?',
                            "That's a shame. I hate it when a place doesn't live up to the hype.",
                            "Sometimes life isn't very fair. I like to think of it as motivation to get a better job.",
                            'Well at least you have a plan. Are you planning to start the renovation soon?',
                            "Kids pick those things up quickly.  And it'll help with her hand-eye coordination, reading - all sorts of things!  ",
                            'did you enjoy yourself ',
                            'Haha, how did she feel when she found out?',
                            'that would really help if it was a permanent solution',
                            "Wow that must have been frustrating.  I hope it didn't cost too much.",
                            'How nice of her. You must have been so happy to see her.',
                            "I know it's hard, but practice makes perfect! Keep trying and I am sure you will get it!",
                            'Do they live in a different state than you?',
                            "It reallyi s the best way to do things. That way even if you forget something you've got time to remember and remedy the situation",
                            'Wow, must have been rather frightening. Glad you are ok!',
                            'That was really nice!  What a wonderful surprise!  This act of kindness helps to restore my faith in humanity.',
                            "It's located in a small farming town in Vermont. I went on a tour of their factory once and it was interesting to see the cheese being made.",
                            "Poor guy was just nervous, I'm sure the more you take him the more he will venture away from you and have some fun!",
                            'How did he scare you?',
                            "Isn't that what sisters are for?  What were you guys upset about?",
                            'That is a long time to be in the car for sure.',
                            "I'm glad to hear... Weddings can be stressful",
                            "That sounds amazing! How'd you guys meet?",
                            "Getting out of town and way from all of everday life's struggles always sounds like a great time.  Did you leave you cell phone at home while you weer away to 'really' get away from everything for a minute?",
                            'Man that is scary! Granted i like to hear things about that. ',
                            'Yikes! Was anything damaged? ',
                            'Ouch, I would try and wear something on your neck next time you go in there.',
                            'awesome! was it hold em?',
                            'Not me! haha I love them all!',
                            "Oh that's nice, I love doing that. Did the cat seem happy?",
                            "Yeah, I can imagine. At least it's only one week!",
                            'Ew, I hate spiders. We are in the process of getting them out of our garage.',
                            "That's great news. I don't know what I would do if my mom passed.",
                            'Is that like the child equivalent of under the bed?',
                            "That's really fantastic!  I'm glad to hear you turned your life around.  ",
                            'What kind of work do you do?',
                            'Ah ok I undestand.',
                            'Very sad to hear. You have a good heart and are very caring, that is something to atleast be proud of!',
                            'Man that sounds really stressful...',
                            'You are so strong!  Please thank your husband for his service and thank you for being his support, no matter the miles between you.  Take care of yourself and get out with friends when you can!',
                            'I see. Is it your favorite food now? :p',
                            'YAY! good job! He/she is going to be beautiful',
                            'Nothing went wrong, we just have different lives in different places. I go visit every now and then.',
                            "A spelling bee - what fun! I'm sure you will win - I bet you've worked hard toward your goal.",
                            'You should install security cameras outside your house.',
                            'Border collie. She was great!',
                            'Oh dear me.. So sorry to hear that! what did you do?',
                            'Praise God man! He really is amazing and we should always be grateful for we have ',
                            'Oh no.. Did he pass away?',
                        ],
                    },
                ),
            ]
            for opt, example in opts_and_examples:
                full_opt = Opt({**opt, 'datapath': data_path})
                teacher = EmpatheticDialoguesTeacher(full_opt)
                self.assertEqual(teacher.get(episode_idx=1, entry_idx=1), example)


if __name__ == '__main__':
    unittest.main()
