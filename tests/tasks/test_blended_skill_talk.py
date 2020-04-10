#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import parlai.utils.testing as testing_utils
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.scripts.display_data import setup_args


class TestBlendedSkillTalkTeacher(unittest.TestCase):
    """
    Basic tests to count the number of examples/episodes and to check a few utterances.

    Manual check of the number of episodes and examples:
    >>> import json
    >>> for datatype in ['train', 'valid', 'test']:
    >>>    with open(f'data/blended_skill_talk/{datatype}.json') as f:
    >>>        data = json.load(f)
    >>>    num_episodes = len(data)
    >>>    num_examples = sum([len(d['dialog']) // 2 for d in data])
    >>>    print(f'Number of episodes: {num_episodes:d}')
    >>>    print(f'Number of examples: {num_examples:d}')

    Output of manual check:
    Number of episodes: 4819
    Number of examples: 27018
    Number of episodes: 1009
    Number of examples: 5651
    Number of episodes: 980
    Number of examples: 5482
    """

    def test_counts(self):

        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir

            opts_episodes_and_examples = [
                ({'datatype': 'train'}, 4819, 27018),
                ({'datatype': 'valid'}, 1009, 5651),
                ({'datatype': 'test'}, 980, 5482),
            ]
            for kwargs, num_episodes, num_examples in opts_episodes_and_examples:
                all_kwargs = {
                    **kwargs,
                    'task': 'blended_skill_talk',
                    'datapath': data_path,
                }
                parser = setup_args()
                parser.set_defaults(**all_kwargs)
                opt = parser.parse_args([])
                agent = RepeatLabelAgent(opt)
                teacher = create_task(opt, agent).get_task_agent()
                self.assertEqual(teacher.num_episodes(), num_episodes)
                self.assertEqual(teacher.num_examples(), num_examples)

    def test_check_examples(self):

        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir

            # Check the first entry (entry_idx==0) of the second episode for the train
            # set, in order to check the context for an episode that has a WoW topic
            # string
            train_opt_and_example = (
                {'datatype': 'train'},
                {
                    'text': "your persona: i just bought a new house with my partner.\nyour persona: i like to make my own coffee.\nLasagne\nOh, I love lasagne. I make my own noodles as well as the sauce. \nWow.  That's amazing.  I read where lasagne originated in Italy during the Middle Ages.  \nOh really!? That is interesting. I am actually italian myself.",
                    'labels': [
                        "Awesome. Me and my partner just bought a house. I can't wait to cook in my kitchen."
                    ],
                    'context_dataset': 'wizard_of_wikipedia',
                    'free_turker_message': 'Oh really!? That is interesting. I am actually italian myself.',
                    'guided_turker_chosen_suggestion': ' ',
                    'episode_done': False,
                },
            )
            all_kwargs = {
                **train_opt_and_example[0],
                'task': 'blended_skill_talk',
                'datapath': data_path,
            }
            parser = setup_args()
            parser.set_defaults(**all_kwargs)
            opt = parser.parse_args([])
            agent = RepeatLabelAgent(opt)
            teacher = create_task(opt, agent).get_task_agent()
            self.assertEqual(
                teacher.get(episode_idx=1, entry_idx=0), train_opt_and_example[1]
            )

            # Check the second entry (entry_idx==1) of the second episode for each dataset
            opts_and_examples = [
                (
                    {'datatype': 'train'},
                    {
                        'text': 'Moving in a new place can be a lot of fun. Are you a good cook?',
                        'labels': [
                            'I like to think so. I love to make coffee for an after dinner treat too.'
                        ],
                        'context_dataset': 'wizard_of_wikipedia',
                        'free_turker_message': 'Moving in a new place can be a lot of fun. Are you a good cook?',
                        'guided_turker_chosen_suggestion': ' ',
                        'episode_done': False,
                    },
                ),
                (
                    {'datatype': 'valid'},
                    {
                        'text': 'I like to go mountain biking with my friends.',
                        'labels': [
                            "I have never done that.  Not really the physical activity type, but I'd be willing to give it a try, I guess"
                        ],
                        'context_dataset': 'empathetic_dialogues',
                        'free_turker_message': 'I like to go mountain biking with my friends.',
                        'guided_turker_chosen_suggestion': '',
                        'label_candidates': [
                            'i work as a vet so no days off over here!',
                            'I love the book 1984',
                            "Well it's always right before Friday so that is always good! Do you travel a lot. ",
                            'gardening.  i am a newbie at gardening.',
                            'Thats great I love tea made from fesh leaves',
                            'I hope for your sake as well! Lack of sleep can really affect the brain and cognitive functions.',
                            'I am not looking forward to the cold weather. It snowed here today. I live in Maine and winter is no picnic.',
                            'lol. what kind of cat is he?',
                            "It's pretty fun. The pay isn't great, but it keeps me busy. Decent work for now.",
                            'thats super luckily, because my mom hoard a lots off hello kitty things',
                            'I do I love movies. I mostly like documentaries though. What is your favorite genre of movies?',
                            'that is very impressive you should visit her..',
                            'Now the real grind starts; the everyday work week. Do you plan to go into teaching? Or more of a specialized role in education?',
                            'Yes I love veggies but I love my meats more. Lol',
                            'Oh, I bet. French and Arabic are my second and third languages, and I can understand those better than accents in the deep south haha.',
                            'My favorite movie is the last of the mohicans',
                            "Thank you that'll help me so much",
                            'Yes, it is great exercise for everyone. Do you like the show silicon valley?',
                            "I've never had a bird.  Are they easy to take care of?",
                            'My favorites are Rolling Stones and Police. ',
                            "I'm in school to be a computer engineer.  Interior design must be very interesting.",
                            'how long is he away?',
                            'Nice. There are some good violin players in japan that  I would like to learn from',
                            'Yea, no doubt we all need a little push once in a while ',
                            "Yeah. I'm an only child. So we're tight.",
                            'I have tried brussel sprouts, they are not my favorite! I am more of a broccoli, carrot type of person',
                            'Totally, we have too many passwords to remember!! How much does it cost you to do a full body painting?',
                            "that is very impressive. i hope i'm successful like you when i grow up. :-) ",
                            'That makes sense =)  I want to eat healthy, but do indulge in fried food sometimes - this could help that.',
                            "Yeah, it's good that it is always on December 25th, so that I don't forget. hahaha.",
                            'the greatest muscian ever to have lived, playboi carti',
                            'I believe it! I hope he made it perfect for you.',
                            "Good for you. Family's real important. I try to get out to visit mine whenever I can even if I'm busy",
                            'Oh yeah, tomato sauce is a staple on pizza.',
                            'Fantasia, no doubt! I like the old classics, but am looking forward to the live version of The Little Mermaid',
                            'The building itself is white, but is illuminated purple by lights. It is amazing.',
                            "Yes, I have a dog.  He's a rescue and the best dog ever, in my opinion. LOL  ",
                            'No that was sony! haha',
                            "Me too! I used to travel all the time but it's gotten a bit harder to do with three kids",
                            "Yeah, those crowds get crazy. I wouldn't take a Bowie knife because I work out a lot. I really want to be a football player when I grow up. ",
                            'I understand. Balance has to be the most difficult thing to regain.',
                            'It is. Have you tried yoga and meditation?',
                            "s Spider doesn't like to get off her couch but Puppy loves to play fetch with a little rope toy he's had forever",
                            "heh, me too--I'm very good at both of those things now!  So what do you do for fun?",
                            "Well I don't know all of Italy's regions, but there seems to be some major differences in the food between the north (alps) and the south of the Italian peninsula.",
                            'Is it by yourself or are you with other people?',
                            '<ovie sound tracks are good as well. Which one did you go see?',
                            '35 mph. But I look really cool on it. My mom says so, anyway.',
                            'He worked doing dry wall. Still required thinking, but of a different type',
                            'mostly comics! what are you into',
                            "Wow, that sounds amazing. I'll have to check that out when I'm in France. Do you like to travel?",
                            'Are they green iguanas?',
                            'I love reddit, it is fun',
                            'Thats awesome, what kind of performance do you do?',
                            'Swing looks fun too!  I like watching lol, but if I had to pick anything it would probably be swing as well.',
                            "Oh that's great. I love fruit so we have 10 different trees in our garden!",
                            'I think Jennifer Lopez and Shakira are playing the show this year.',
                            "Thank you, I couldn't have done it without an amazing team!",
                            'thats a good hobby to have too, whatever you enjoy',
                            "That doesn't sound nice! Have you recovered from the Keto Flu yet?",
                            'No, sadly, sports is not my thing. My son like to play a pick-up football game when he can. ',
                            'So which one is your favourite then?',
                            ' I got the cat without asking my parents. My dad got upset and I had to give it away to my friend.',
                            'i like a lot but listen to country music too. Recently I have been liking soundscapes',
                            "No, I don't think Coke would help me right now, I probably should see a doctor.",
                            'Thats amazing! How much do the passes cost?',
                            "Sorry for your loss.  I'm glad she was there to provide you support.",
                            "The summer. I can't stand the heat and humidity. What is yours?",
                            'I have 2. Mine like to wake me up too. Ive started drinking a Dr. Pepper in the morning to help me wake up. Its been too hot for coffee.',
                            'that is so insane!',
                            'Good to see your interest. I too fond of this movie.',
                            'You should check it out , if you have time. Its actually one of my favourite shows.',
                            'He got me a Dell and I love it',
                            "Water rides are pretty cool most of them don't have as steep of drops so they tend to be easier on me.",
                            "that's great , i wish i had a view of the mountains .",
                            'Is there any particular program she is going to study? ',
                            'I fish off of a pier. I cant swim so a boat is out of the question for me. ',
                            'Where is the first place you are looking forward to going?',
                            'Yes! Did you do anything similar?',
                            'Are you done with college?',
                            'I really do. How about you?',
                            'I was not paying attention and i tripped over a big piece of wood.',
                            'Okay! It was nice chatting with you!',
                            "I have never done that.  Not really the physical activity type, but I'd be willing to give it a try, I guess",
                            'BC?  Is that British Columbia?',
                            "They always should be! There isn't any reason for a woman who cares about what she is doing not to do it. ",
                            "I'm sure you will.  Children are the best, but a handful.  Are you married?",
                            "I'm a baller. Not quite like LeBron yet, but I'm getting there.",
                            'You are absolutely right there.  If its truly a passion then you should go for it.  Life is too short to be stuck in a rut.',
                            'extreme couponing is such a weird and annoying show, really dont care for it.',
                            "That's really important.. and, ESPECIALLY in the medical field, where that 'bedside manner' is a major factor in the healing process.",
                            'oh Im sorry to hear to that your hands are unsteady, you may choose powder based makeup and use eyeshadow for liner making it easier.',
                            'Writing is great, keeps the creative juices going',
                            'Yeah, I agree.  I would drink some beer, eat some pizza, and make some comics...that would be a perfect night to me.',
                            "Yeah, it's tough balancing work and pets at times",
                            'are you left handed or right handed?',
                            "Don't you just love the band One Direction?",
                            'Has he been with you his entire life?',
                            'that is too bad.',
                            'And what else? ',
                        ],
                        'episode_done': False,
                    },
                ),
                (
                    {'datatype': 'test'},
                    {
                        'text': "He eats insects, leaves and sun flower seeds. It's easy. They don't need walking and cleanup is simple. Do you have any pets?",
                        'labels': [
                            'No, not at the moment.  I have 3 girls and they are enough trouble! LOL'
                        ],
                        'context_dataset': 'empathetic_dialogues',
                        'free_turker_message': "He eats insects, leaves and sun flower seeds. It's easy. They don't need walking and cleanup is simple. Do you have any pets?",
                        'guided_turker_chosen_suggestion': '',
                        'label_candidates': [
                            "Wow, engineering, sounds impressive.  I'm sure the income will be awesome.",
                            'Do you own lots of things related to The Beatles?',
                            'Yeah, you should probably look into that more. She sounds shady.',
                            'In the past the umbrella term was billiards',
                            "No, not indoors yet. I'll try. I don't know how you can stay so calm? I'm about ready to get in my car and leave.",
                            "It's in the works, believe me. This is just my job while I'm in school. I have a semester left! What do you do?",
                            "You're welcome.  I wish you the best! what do you do for a living ?",
                            'That would have been all bad. Can you imagine explaining that to the grieving family?',
                            "I'm a junior in college, my 3rd year. I hope to travel for my career as I love the beach and surfing. ",
                            'Oh sure, I love dressing up! Did you dress up for Halloween? I was a space alien.',
                            "No, my mom was a great role model.  Couldn't have done it without her",
                            'It started to rub off on her students and the day went better! ',
                            "Nah. I live in the south. We rarely get snow. I'm glad though because we can't ski. I'm so afraid of heights.",
                            'Folly beach in SC they have a great steak and seafood house there. I love my meat so they have the best of both worlds.',
                            'I agree. The best thing about summer is that I can read my mystery novels outside in the sun. Do you like mystery books?',
                            'Truly! I prefer a smaller venue setting as opposed to Music Festivals though, how about you?',
                            'you should give it a go , what do you do for work ?',
                            'Aww, same thing happens here on the holidays.',
                            'We both love riding motorcycles and playing put-put.',
                            'I also like to watch the walking dead. What about you?',
                            'Yeah, the HS rules keep players from being able to play year round. But one day we will be as good as they are, I know.',
                            'i love horror movies . i love to be afraid !',
                            'Yes, that is usually how it works. Are there any other hotels hiring for a day shift near you?',
                            "I did: my daughter. But she's still to young to hold the cup steady.",
                            "That's sad, how long ago was it?",
                            'I have a friend who has one of those! Very very smart and playful. I am looking to get a dog soon, for now we just have 2 cats.',
                            'I love that - Have you ever heard of "Walking tacos"? This is where you take an individual bag of doritos or fritos, and have all of the fixings.',
                            'Wow, that is so sweet!  Are you guys still close?',
                            "It's much easier now. I'm always planning trips in my mind -- I'm planning every trip I'm going to take for the next five years",
                            'Yes the travel is only part of it though,  i used to live in Nevada, ',
                            "Interesting. I'm not entirely sure, I'd probably say Bright Eyes because of the lyrics in their later albums.",
                            'Especially in the capital, Montgomery!  One of the state pastimes of Alabama is getting drunk in the street.',
                            'Yeah, My dad always my hero. I am prayer god to recover your dad.',
                            'exactly, a nice day of relaxing is just right. ',
                            "I'm planning on being an engineer so it's a bit of both. ",
                            "White belt. I've just started.",
                            'We never seem to be on the same page sometimes.',
                            'We are very close',
                            'When I have my restaurant, I can hire you :)',
                            'She had a skin cancer.',
                            'i think i am lucky that i dont like to drink.',
                            'Yes. Any type of physical exercise will help you maintain fitness and overall health and wellness.',
                            'It is exhausting! But at least I get a lot of holiday time to recover haha! Do you have a job?',
                            "I love my job. It's not easy but I enjoy being around kids. What do you study?",
                            "Some thoughts you just have to capture before they're gone, right?  ",
                            'I love little doggies whats your preference',
                            'Sadly, I can only read about Chinese history in English, but I can speak the other languages. They are so fun1',
                            'I have only seen the first 5 minutes of the the first show. Do you like game of thrones?',
                            'man , you have got an all around good deal !',
                            'I actually play for the detroit lions',
                            "Well. The US government won't say anything, but I have come across some interesting leads in a small Russian newspaper, which lead my investigation to a library in Kabul.",
                            "Ok - thanks for the tip.  I'll definitely have to check that out.  I want to stay healthy.  I hope it helps out your friend",
                            'I am! We are pregnant with twins! :) Due to minor complications the Dr. currently has me on bed rest.',
                            "Yeah he works at the local Church and he did some work with missionaries. But the little ones don't let him go",
                            'I was thinking the same thing, chocolate covered strawberries are the best!',
                            "I like Almond milk chocolate.  I'm glad chocolate has health benefits as well.",
                            "I've traveled around quite a bit and I always feel like the countries that legalized weed have themselves figured out much more.",
                            'I really liked chemistry because it allowed me to really try and understand life at a small level.',
                            "I think that's a great idea!Make sure you stretch first!",
                            'Yeah, my kids love pizza, I like to make it myself rather than order it, at least then I can save some money ',
                            "Same.  I'm in Naples, FL",
                            'No. I just wondered what you meant by saying you would play against the Lakers. I am not athletic at all.',
                            "It might sound silly, but I go to Reddit for life advice, lol. :) There's a lot of wise people on there!",
                            "My favorite place I've ever been in Pensacola FL. Warm, lovely beaches. Alcohol and seafood!",
                            'Most teller jobs need experience with handling cash and a high school diploma. so you would have to have some expirience in the field',
                            "What's your age compared to your roomates? ",
                            'I love museums as well, they really help teach the public about different things we may not otherwise know',
                            "Yeah. I'm from Texas. I used to fish for bass all the time. Only on vacations now.",
                            'Ribeye. Filet has better texture but ribeye has the supreme taste. Do you take pride in cooking? I take inspiration from the shows',
                            'Yes me too, I have been sick with cold going on 4 weeks.',
                            'True, I have to go to the eye doctor soon to get some more contacts this saturday but then sunday going to the beach, what about You?',
                            "I definitely do. It's amazing. My mom owns a shoe store and has over 500 pairs!",
                            "Cool.  I haven't done much clothes shopping in a while.  I miss it",
                            'That sounds awesome! I love tzaziki. Never had it with fries before!',
                            'What does your turtle eat?  Is it hard to take care of a turtle?',
                            'No, not at the moment.  I have 3 girls and they are enough trouble! LOL',
                            'i love dogs, i love cats more!!',
                            "Unfourtnally I can't grow anything.",
                            'So do I.  Anything with avocado is great. I love the freshness it brings.',
                            'Yeah - and, if you put a basket on it, you can go pick up some groceries as well.',
                            "yeah, i understand that hope you get new one pretty soon it's fun",
                            'I listen to absolutely anything, as long as it sounds good to me. I love blues though',
                            'Thank goodness you found each other. Hopefully you are both benefitting from resources that enhance your lives.',
                            'Walking in nature is amazing and there are so many health benefits from walking!',
                            'The season is early and lots can happen',
                            "We need more musicians like him - he was really about the music. There's way too much commercial garbage on the radio these days.",
                            'We should carve stone monuments so that when we freeze to death, the next dominant species can learn about us in the future.',
                            "That's quite interesting.  I need to learn more about cryptocurrency.  I thought that I had missed the opportunity.",
                            "Might check it out sometime! I'm actually in a small band at the moment although not many people watch us it's still pretty fun to perform",
                            'i like star trek',
                            "Lol I hear it's better to ask for forgiveness than permission.",
                            'I enjoy a lot of great vegetarian recipes. do you have any favorites?',
                            "i'm glad . always think positive .",
                            'What kind of music do you like?i work in a factory so my music is machine niose ',
                            'Are you a TB Lightning fan?',
                            'He is a very great actor. I also like him in Gangs of New York.',
                            'Oh god its a so sad news. how it happended its due to traffic collision.',
                            'Yeah, i need a vacation from all my work. I should use the vacation to do some charity...',
                            'do you know the song hunting fishing loving every day?',
                            'but the worst part is you have to clean every day and keep the flat tidy all the time.  ',
                        ],
                        'episode_done': False,
                    },
                ),
            ]
            for kwargs, example in opts_and_examples:
                all_kwargs = {
                    **kwargs,
                    'task': 'blended_skill_talk',
                    'datapath': data_path,
                }
                parser = setup_args()
                parser.set_defaults(**all_kwargs)
                opt = parser.parse_args([])
                agent = RepeatLabelAgent(opt)
                teacher = create_task(opt, agent).get_task_agent()
                self.assertEqual(teacher.get(episode_idx=1, entry_idx=1), example)


class TestPersonaTopicifierTeachers(unittest.TestCase):
    """
    Test the outputs of the first example of teachers used for all-in-one training,
    including those that add ConvAI2 personas and WoW topics to contexts.
    """

    def test_check_examples(self):

        # Define all pairs of task strings and examples
        no_header_normal = [
            (
                'empathetic_dialogues',
                {
                    'situation': 'I remember going to the fireworks with my best friend. There was a lot of people, but it only felt like us in the world.',
                    'emotion': 'sentimental',
                    'text': 'I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world.',
                    'labels': [
                        'Was this a friend you were in love with, or just a best friend?'
                    ],
                    'prepend_ctx': None,
                    'prepend_cand': None,
                    'deepmoji_ctx': None,
                    'deepmoji_cand': None,
                    'episode_done': False,
                    'label_candidates': [],
                },
            ),
            (
                'wizard_of_wikipedia',
                {
                    'id': 'WizardDialogKnowledgeTeacher',
                    'text': 'Science fiction',
                    'labels': [
                        "I think science fiction is an amazing genre for anything. Future science, technology, time travel, FTL travel, they're all such interesting concepts."
                    ],
                    'chosen_topic': 'Science fiction',
                    'episode_done': False,
                    'label_candidates': [],
                    'knowledge': 'Science fiction Science fiction (often shortened to SF or sci-fi) is a genre of speculative fiction, typically dealing with imaginative concepts such as futuristic science and technology, space travel, time travel, faster than light travel, parallel universes, and extraterrestrial life.\nScience fiction Science fiction often explores the potential consequences of scientific and other innovations, and has been called a "literature of ideas".\nScience fiction It usually avoids the supernatural, unlike the related genre of fantasy.\nScience fiction Historically, science-fiction stories have had a grounding in actual science, but now this is only expected of hard science fiction.\nScience fiction Science fiction is difficult to define, as it includes a wide range of subgenres and themes.\nScience fiction Hugo Gernsback, who suggested the term "scientifiction" for his "Amazing Stories" magazine, wrote: "By \'scientifiction\' I mean the Jules Verne, H. G. Wells and Edgar Allan Poe type of story—a charming romance intermingled with scientific fact and prophetic vision... Not only do these amazing tales make tremendously interesting reading—they are always instructive.\nScience fiction They supply knowledge... in a very palatable form... New adventures pictured for us in the scientifiction of today are not at all impossible of realization tomorrow...\n',
                    'title': 'Science fiction',
                    'checked_sentence': 'Science fiction (often shortened to SF or sci-fi) is a genre of speculative fiction, typically dealing with imaginative concepts such as futuristic science and technology, space travel, time travel, faster than light travel, parallel universes, and extraterrestrial life.',
                },
            ),
        ]
        persona_and_topic_normal = [
            (
                "blended_skill_talk:ConvAI2PersonaTopicifier",
                {
                    'text': "your persona: i like to remodel homes.\nyour persona: i like to go hunting.\nyour persona: i like to shoot a bow.\nyour persona: my favorite holiday is halloween.\nNicholas Sparks\nhi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .",
                    'labels': (
                        'you must be very fast . hunting is one of my favorite hobbies .',
                    ),
                    'reward': 0,
                    'label_candidates': (
                        'my mom was single with 3 boys , so we never left the projects .',
                        'i try to wear all black every day . it makes me feel comfortable .',
                        'well nursing stresses you out so i wish luck with sister',
                        'yeah just want to pick up nba nfl getting old',
                        'i really like celine dion . what about you ?',
                        'no . i live near farms .',
                        "i wish i had a daughter , i'm a boy mom . they're beautiful boys though still lucky",
                        'yeah when i get bored i play gone with the wind my favorite movie .',
                        "hi how are you ? i'm eatingdinner with my hubby and 2 kids .",
                        'were you married to your high school sweetheart ? i was .',
                        'that is great to hear ! are you a competitive rider ?',
                        "hi , i'm doing ok . i'm abanker . how about you ?",
                        "i'm 5 years old",
                        'hi there . how are you today ?',
                        'i totally understand how stressful that can be .',
                        'yeah sometimes you do not know what you are actually watching',
                        'mother taught me to cook ! we are looking for an exterminator .',
                        'i enjoy romantic movie . what is your favorite season ? mine is summer .',
                        'editing photos takesa lot of work .',
                        'you must be very fast . hunting is one of my favorite hobbies .',
                    ),
                    'episode_done': False,
                },
            ),
            (
                "blended_skill_talk:EDPersonaTopicifier",
                {
                    'situation': 'I remember going to the fireworks with my best friend. There was a lot of people, but it only felt like us in the world.',
                    'emotion': 'sentimental',
                    'text': 'your persona: people hate that i obsess about the poor.\nyour persona: i like to make cellphone apps that would help heal our world.\nyour persona: i like to watch people pray together.\nyour persona: people don t like me too much but i like them anyways.\nAndroid (operating system)#Applications\nI remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world.',
                    'labels': [
                        'Was this a friend you were in love with, or just a best friend?'
                    ],
                    'prepend_ctx': None,
                    'prepend_cand': None,
                    'deepmoji_ctx': None,
                    'deepmoji_cand': None,
                    'episode_done': False,
                    'label_candidates': [],
                },
            ),
            (
                "blended_skill_talk:WoWPersonaTopicifier",
                {
                    'id': 'WizardDialogKnowledgeTeacher',
                    'text': "your persona: not a day goes by that i don't drink four mountain dews.\nyour persona: i enjoy movies about aliens invading the earth.\nyour persona: my favorite hobby is chess.\nyour persona: i just dyed my hair hot pink with purple highlights.\nScience fiction\n",
                    'labels': [
                        "I think science fiction is an amazing genre for anything. Future science, technology, time travel, FTL travel, they're all such interesting concepts."
                    ],
                    'chosen_topic': 'Science fiction',
                    'episode_done': False,
                    'label_candidates': [],
                    'knowledge': 'Science fiction Science fiction (often shortened to SF or sci-fi) is a genre of speculative fiction, typically dealing with imaginative concepts such as futuristic science and technology, space travel, time travel, faster than light travel, parallel universes, and extraterrestrial life.\nScience fiction Science fiction often explores the potential consequences of scientific and other innovations, and has been called a "literature of ideas".\nScience fiction It usually avoids the supernatural, unlike the related genre of fantasy.\nScience fiction Historically, science-fiction stories have had a grounding in actual science, but now this is only expected of hard science fiction.\nScience fiction Science fiction is difficult to define, as it includes a wide range of subgenres and themes.\nScience fiction Hugo Gernsback, who suggested the term "scientifiction" for his "Amazing Stories" magazine, wrote: "By \'scientifiction\' I mean the Jules Verne, H. G. Wells and Edgar Allan Poe type of story—a charming romance intermingled with scientific fact and prophetic vision... Not only do these amazing tales make tremendously interesting reading—they are always instructive.\nScience fiction They supply knowledge... in a very palatable form... New adventures pictured for us in the scientifiction of today are not at all impossible of realization tomorrow...\n',
                    'title': 'Science fiction',
                    'checked_sentence': 'Science fiction (often shortened to SF or sci-fi) is a genre of speculative fiction, typically dealing with imaginative concepts such as futuristic science and technology, space travel, time travel, faster than light travel, parallel universes, and extraterrestrial life.',
                },
            ),
        ]
        all_tasks_and_messages = no_header_normal + persona_and_topic_normal

        for task_string, desired_message in all_tasks_and_messages:

            # Get message
            kwargs = {'task': task_string, 'datatype': 'train:ordered'}
            parser = setup_args()
            parser.set_defaults(**kwargs)
            opt = parser.parse_args([])
            agent = RepeatLabelAgent(opt)
            teacher = create_task(opt, agent).get_task_agent()
            actual_message = teacher.get(episode_idx=0, entry_idx=0)

            print(f'\nChecking {task_string}:')
            for key in desired_message.keys():
                if key in ['label_candidates']:
                    # These are often created randomly and thus will vary
                    continue
                print(key)
                self.assertEqual(desired_message[key], actual_message[key])
            print('')


if __name__ == '__main__':
    unittest.main()
