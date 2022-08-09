#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from projects.bb3.agents.module import Module
import random

#######################
# Important Constants #
#######################
PARTNER_PREFIX = 'Person 1'
SELF_PREFIX = 'Person 2'
QUERY_GEN_PREFIX = 'Query'
MEMORY_GEN_PREFIX = 'Memory'
MEMORY_KNOWLEDGE_PREFIX = 'Personal Fact'
CONTEXTUAL_KNOWLEDGE_PREFIX = 'Previous Topic'
SEARCH_KNOWLEDGE_PREFIX = 'Interesting Fact'
STYLE_PREFIX = 'Style'
OPENING_PREFIX = 'Opening'

SELF_MEMORY_PREFIX = f"{SELF_PREFIX}'s Persona"
PARTNER_MEMORY_PREFIX = f"{PARTNER_PREFIX}'s Persona"
EXTERNAL_KNOWLEDGE_PREFIX = "External Knowledge"

SEARCH = 'search'
DO_NOT_SEARCH = 'do not search'
SEARCH_DECISION = "Search Decision"
MEMORY_DECISION = "Memory Decision"

ACCESS_MEMORY = "access memory"
DO_NOT_ACCESS_MEMORY = "do not access memory"

NO_MEMORY = "no persona"

PRE_CONTEXT_TOK = "__REPLACE_ME_PRE_CONTEXT__"
POST_CONTEXT_TOK = "__REPLACE_ME_POST_CONTEXT__"

MAX_PROMPT_LEN = 1912

###################
# LIGHT CONSTANTS #
###################
LIGHT_SETTING_NAME = "Setting"
LIGHT_SETTING_DESC = "Setting Description"
LIGHT_PARTNER_NAME = f"{PARTNER_PREFIX} is"
LIGHT_SELF_NAME = f"{SELF_PREFIX} is"

# parlai_internal.projects.blenderbot3.tasks:WoiSearchDecisionJsonTeacher
# parlai_internal.projects.blenderbot3.tasks:AlwaysSearchTeacher
SEARCH_DECISION_STRING = f"""
{PARTNER_PREFIX}: Who directed 2001: A Space Odyssey?
{SEARCH_DECISION}: {SEARCH}

{PARTNER_PREFIX}: Do you like italian food as well?
{SEARCH_DECISION}: {DO_NOT_SEARCH}

{PARTNER_PREFIX}: On which channel will the FIFA world cup be broadcasted?
{SEARCH_DECISION}: {SEARCH}

{PARTNER_PREFIX}: oh gosh, I'm so sad now. That's terrible
{SEARCH_DECISION}: {DO_NOT_SEARCH}

{PARTNER_PREFIX}: To whom was John B. Kroc married?
{SEARCH_DECISION}: {SEARCH}

{PARTNER_PREFIX}: have you ever adopted an animal from animal shelters? I did, i adopted a chinchilla and a cat
{SEARCH_DECISION}: {DO_NOT_SEARCH}

{PARTNER_PREFIX}: Which is the largest of the Japanese Volcano Islands?
{SEARCH_DECISION}: {SEARCH}

{PARTNER_PREFIX}: how cool! are you going to go?
{SEARCH_DECISION}: {DO_NOT_SEARCH}

{PARTNER_PREFIX}: I did not know he did media other than just his children stories books. Did Dr. Seuss do anything else other than paintings and books, such as movies or scripts?
{SEARCH_DECISION}: {SEARCH}"""


# parlai_internal.projects.blenderbot3.tasks:MSCMemoryDecisionJsonTeacher
MEMORY_DECISION_STRING = f"""
{PARTNER_PREFIX}: I started reading a new novel series. It's very much like agatha christie. I think I found something to keep me going.
{MEMORY_DECISION}: {ACCESS_MEMORY}

{PARTNER_PREFIX}: No. He was an author, but has been retired for a couple of years now.
{MEMORY_DECISION}: {DO_NOT_ACCESS_MEMORY}

{PARTNER_PREFIX}: No plans right now, but I'd love to go back to ireland. I stayed in dublin last time I was there. Would love to visit the other side of the island. Maybe around galway. Are you irish? That's wht I went. Wanted to see the ancestral home.
{MEMORY_DECISION}: {ACCESS_MEMORY}

{PARTNER_PREFIX}: I'm very sorry for your loss.
{MEMORY_DECISION}: {DO_NOT_ACCESS_MEMORY}

{PARTNER_PREFIX}: It is beautiful there. If you get the chance its definitely worth the trip. How old are your children?
{MEMORY_DECISION}: {ACCESS_MEMORY}

{MEMORY_DECISION}: I think corn may be easier on the stomach, otherwise it's just too starchy!
{MEMORY_DECISION}: {DO_NOT_ACCESS_MEMORY}

{PARTNER_PREFIX}: What kind of job are you looking for?
{MEMORY_DECISION}: {ACCESS_MEMORY}

{PARTNER_PREFIX}: Yeah, I'll send you a link. I saw an ad for it earlier this week. Unfortunately, I haven't been swimming much it's been really cold outside! I wish my gym had an indoor heated pool.
{MEMORY_DECISION}: {DO_NOT_ACCESS_MEMORY}

{PARTNER_PREFIX}: That's lovely. I have 1 dog and 1 cat. And you?
{MEMORY_DECISION}: {ACCESS_MEMORY}"""


# parlai_internal.projects.blenderbot3.tasks:WoiSearchQueryJsonTeacher
SEARCH_QUERY_STRING = f"""
{PARTNER_PREFIX}: My favorite song is Rock On By David Essex. Most people don't even know this song, but it is amazing and everyone should hear it.
{SELF_PREFIX}: I'm not too familiar with David Essex's Rock On. Tell me something about the song.
{PARTNER_PREFIX}: It has a unique bass line. The lyrics speak to me. It is a classic rock song from 1973
{QUERY_GEN_PREFIX}: Rock On David Essex lyrics

{PARTNER_PREFIX}: My favorite athlete is nadal; he will be the greatest tennis player.
{SELF_PREFIX}: Tell about your hobbies in 2021!
{PARTNER_PREFIX}: I will continue to play tennis. I hope that I can hit the ball once a day.
{SELF_PREFIX}: Do you know about Rafael?
{PARTNER_PREFIX}: Rafa is my favourite tennis player
{QUERY_GEN_PREFIX}: interesting facts about rafael

{PARTNER_PREFIX}: My favorite item to buy is Chocolate bars. My favourite is Oh Henry. I can never resist buying a chocolate bar when I am at the grocery store.
{SELF_PREFIX}: Did you ever try the premium chocolates boxes for yourself or sending gifts to someone?
{PARTNER_PREFIX}: I have never seem them. Do they have a good variety of chocolate?
{SELF_PREFIX}: Yes they do. They have much more than boxed chocolates and bars. They have different kind of flavors like caramels, toffee bars etc
{PARTNER_PREFIX}: It sounds wonderful. I wonder if there are any nut-free varieties or low sugar options.
{QUERY_GEN_PREFIX}: Low sugar chocolate

{PARTNER_PREFIX}: My favorite author is john grisham, and i play paintball on the weekends. Have you ever been shot by a paintball?
{SELF_PREFIX}: No, but I could imagine it probably hurts!
{PARTNER_PREFIX}: yes it does, especially if the paintballs are frozen!
{SELF_PREFIX}: Frozen? I didn't even know you could freeze them! What would you compare the feeling to? I've heard a lot of people will sprain their ankles or strain a muscle while out playing!
{PARTNER_PREFIX}: i would equate it to a pretty bad wasp sting.  yeah i have definitely sprained an ankle out there.
{QUERY_GEN_PREFIX}: sprained ankle healing

{PARTNER_PREFIX}: My favorite actor is Tom Hanks. It's absolutely incredible how many hit movies Tom Hanks has put out. What is your favorite Tom Hanks movie of them all?
{QUERY_GEN_PREFIX}: Tom Hanks movies"""


# parlai_internal.projects.blenderbot3.tasks:MSCMemoryGeneratorJsonTeacher
MEMORY_GENERATOR_STRING = f"""
{SELF_PREFIX}: Hello, whats your favorite color? Mine is yellow.
{PARTNER_PREFIX}: Good, thank you! Tell me a secret about your life.
{SELF_PREFIX}: I believe in ghosts, and I am fascinated by them.
{PARTNER_PREFIX}: I have not once kissed a woman. I win! Lol
{SELF_PREFIX}: Oh yeah, well my childhood dream was to be an architect.
{PARTNER_PREFIX}: Nice! What career did you end up in?
{SELF_PREFIX}: I plan weddings now, what do you do?
{PARTNER_PREFIX}: Nice! I am in business. I surprised of where I am, really.
{MEMORY_GEN_PREFIX}: {PARTNER_PREFIX} works in the business sector.

{SELF_PREFIX}: The Target app is awesome! Is this what you use to shop?
{PARTNER_PREFIX}: No, I usually just go to the store and shop. There's a Target near my home.
{MEMORY_GEN_PREFIX}: {PARTNER_PREFIX} lives near a Target.

{SELF_PREFIX}: Have you found any luck finding a second job?
{PARTNER_PREFIX}: Hey, not yet. I'm just so tired all the time. Working nights is hard, and during the day I'm working on the details of trying to open my own restaurant, but that's hard to do when you're as broke as me, but it's my dream, so I'm going to work at it as much as it takes. I'll keep looking for a second job to save up the money. {SELF_PREFIX}'s business doesn't happen to be hiring right now are they?
{MEMORY_GEN_PREFIX}: {PARTNER_PREFIX} has a job, wants to open my own restaurant, works at night, wants to find a second job, and doesn't have a lot of money.

{PARTNER_PREFIX}: Hello. What are you up to today?
{SELF_PREFIX}: Same as everyday, just sitting at home listening to backstreet boys. You?
{PARTNER_PREFIX}: Love the band. I'm going rock climbing with my dog
{MEMORY_GEN_PREFIX}: {PARTNER_PREFIX} is rock climbing with their dog. {PARTNER_PREFIX} loves the Backstreet Boys.

{PARTNER_PREFIX}: I just took some great photographs outdoors.
{MEMORY_GEN_PREFIX}: {NO_MEMORY}"""


# parlai_internal.projects.blenderbot3.tasks:MSCPersonaKnowledgeJsonTeacher
MEMORY_KNOWLEDGE_STRING = f"""
{PARTNER_MEMORY_PREFIX} I do not have any pets. I like all sports. I enjoy coaching volleyball and my volleyball team is my favorite team.
{PARTNER_MEMORY_PREFIX} I enjoy doing crosswords. I enjoy hiking.
{PARTNER_MEMORY_PREFIX} I speak french and english fluently.
{PARTNER_MEMORY_PREFIX} I've two border collies.
{SELF_MEMORY_PREFIX} I own my own art studio.
{SELF_MEMORY_PREFIX} I enjoy going on hikes.
{SELF_MEMORY_PREFIX} I speak French and French food is my favorite.
{SELF_MEMORY_PREFIX} I do not have cats because I am allergic to cats. I have two collies.
{SELF_MEMORY_PREFIX} I coach the girl s volley ball team.
{SELF_MEMORY_PREFIX} I am working on art projects. I am moving.
{SELF_MEMORY_PREFIX} I am moving my studio. I am an artist.
{SELF_MEMORY_PREFIX} I didn't study art in school. I've thought about doing school tours, participated in art-focused assemblies and the like, but not about teaching art full time.
{SELF_PREFIX}: Hi. How are you?
{PARTNER_PREFIX}: I am good. I just got home from work. You?
{SELF_PREFIX}: Good. Cleaning up my art studio. Where do you work?
{PARTNER_PREFIX}: I teach middle school. I relax on weekends by singing in a bar.
{SELF_PREFIX}: How fun. I go on hikes. Do you have pets?
{PARTNER_PREFIX}: No pets for me. I do like sports though. Football is my favorite.
{SELF_PREFIX}: Do you have a favorite team?
{MEMORY_KNOWLEDGE_PREFIX}: {PARTNER_PREFIX} coaches a girl's volley ball team.

{PARTNER_MEMORY_PREFIX} I work at the church food pantry.
{PARTNER_MEMORY_PREFIX} I have 5 children, 2 of them have died.
{PARTNER_MEMORY_PREFIX} I plant sunflowers on my children's graves.
{PARTNER_MEMORY_PREFIX} I love to read agatha christie.
{SELF_MEMORY_PREFIX} My mother was born in ireland.
{PARTNER_MEMORY_PREFIX} I like to travel.
{PARTNER_MEMORY_PREFIX} My father was an author.
{SELF_MEMORY_PREFIX} I went to Australia.
{SELF_MEMORY_PREFIX} I've been to Ireland. My mother is from Ireland.
{SELF_MEMORY_PREFIX} My father was an author.
{SELF_MEMORY_PREFIX} I love to read. I've read the black stallion.
{SELF_MEMORY_PREFIX} I grow the tallest sunflowers you have ever seen.
{PARTNER_MEMORY_PREFIX} I have never been to Ireland. I want to travel to Ireland soon.
{SELF_MEMORY_PREFIX} I have been to Dublin. I would like to visit Galway.
{PARTNER_MEMORY_PREFIX} My twins were born premature. I have twins.
{SELF_PREFIX}: Hello. I just got back from australia. How are you?
{PARTNER_PREFIX}: I am well. Just finished working at the church food pantry where o am. Most days.
{SELF_PREFIX}: I also like to go to ireland. It is where my mother was from.
{PARTNER_PREFIX}: I have 5 lovely children that are my world. 2 have died
{SELF_PREFIX}: I am sorry to hear that. Do you have fun hobbies?
{PARTNER_PREFIX}: The tallest sunflowers you have ever seen are over their graves
{SELF_PREFIX}: Wow that sounds wonderful! Did you plant them there?
{PARTNER_PREFIX}: I grow them as well as read agatha christie novels
{SELF_PREFIX}: My father liked those books too and he was a great famous author.
{PARTNER_PREFIX}: I love reading them. Do you like to read?
{MEMORY_KNOWLEDGE_PREFIX}: {SELF_PREFIX} has read the black stallion."""


# parlai_internal.projects.blenderbot3.tasks:Convai2KnowledgeJsonTeacher
CONTEXTUAL_KNOWLEDGE_STRING = f"""
{SELF_PREFIX}: I like to remodel homes, go hunting, and shoot a bow.
{SELF_PREFIX}: My favorite holiday is halloween.
{PARTNER_PREFIX}: Hi, how are you doing? I'm getting ready to do some cheetah chasing to stay in shape.
{SELF_PREFIX}: You must be very fast. Hunting is one of my favorite hobbies.
{PARTNER_PREFIX}: I am! For my hobby I like to do canning or some whittling.
{CONTEXTUAL_KNOWLEDGE_PREFIX}: homes

{SELF_PREFIX}: My mom is my best friend, I have four sisters, I believe that mermaids are real, and I love iced tea.
{PARTNER_PREFIX}: Hi, how are you doing today?
{SELF_PREFIX}: I am spending time with my 4 sisters, what are you up to?
{PARTNER_PREFIX}: Wow, four sisters. Just watching game of thrones.
{SELF_PREFIX}: That is a good show, I watch that while drinking iced tea.
{PARTNER_PREFIX}: I agree. What do you do for a living?
{CONTEXTUAL_KNOWLEDGE_PREFIX}: mermaids

{SELF_PREFIX}: I had a gig at local theater last night, I work as a stand up comedian, I come from a small town, my favorite drink is cuba libre, and I did a few small roles in tv series.
{PARTNER_PREFIX}: We all live in a yellow submarine, a yellow submarine. Morning!
{SELF_PREFIX}: Hi! That is a great line for my next stand up.
{PARTNER_PREFIX}: Lol. I am shy, anything to break the ice, and I am a beatles fan.
{CONTEXTUAL_KNOWLEDGE_PREFIX}: tv

{SELF_PREFIX}: I like horseback riding, my favorite food is cheese, I'm allergic to shellfish, I work at a non profit that helps children, and I love going to concerts and dancing hard.
{PARTNER_PREFIX}: Hello, I've to be quick because I work in a hospital
{CONTEXTUAL_KNOWLEDGE_PREFIX}: children

{SELF_PREFIX}: I enjoy fishing, I have a dog named bob, I live on an island, and I like to make boats on the weekends.
{PARTNER_PREFIX}: Hi, how are you today?
{SELF_PREFIX}: I am great just about to head out fishing one of my favorite pastimes
{PARTNER_PREFIX}: Sounds fun! I am going shoe shopping later.
{SELF_PREFIX}: On the island that I live on there are many places to shop
{PARTNER_PREFIX}: I would shop all day! What do you like to do in your spare time?
{CONTEXTUAL_KNOWLEDGE_PREFIX}: weekends"""


# parlai_internal.projects.blenderbot3.tasks:WoiKnowledgeJsonTeacher
# parlai_internal.projects.blenderbot3.tasks:WowKnowledgeJsonTeacher
# parlai_internal.projects.blenderbot3.tasks:NQOpenKnowledgeJsonTeacher
SEARCH_KNOWLEDGE_STRING = f"""
{EXTERNAL_KNOWLEDGE_PREFIX}: His Military Charity Work. This Is Us: Susan Kelechi Watson on How Beth Is Finding Her Identity Apart From Randall. Lego’s new Toy Story 4 tie-in sets unveiled. Sun, Feb 17 12:30 AM EST on AMC (277). Agent Zigzag. A Wilderness of Monkeys. Untitled  Battle of the Sexes  Project. Untitled Barry Manilow Project. Editors  Picks: Our Favorites From the Week of Nov. 11. 10 Paul Newman Facts You May Not Know. Top 25 Best Performances By A Leading Man. Talented & Hardworking Artists. How much of Tom Hanks s work have
{EXTERNAL_KNOWLEDGE_PREFIX}: “Forrest Gump” opens, wins Tom Hanks a second Oscar. On this day in 1994, the movie Forrest Gump opens in U.S. theaters. A huge box-office success, the film starred Tom Hanks in the title role of Forrest, a good-hearted man with a low I.Q. who winds up at the center of key cultural and historical events of the second half of the 20th century.. Forrest Gump was based on a 1986 novel of the same name by Winston Groom, who (like his main character) grew up in Alabama and served in the Army during Vietnam.
{EXTERNAL_KNOWLEDGE_PREFIX}: Some commentators argued that Jenny’s eventual demise was a statement about the counter-culture movement in America.. Forrest Gump received 13 Academy Award nominations and took home six Oscars, including Best Picture, Best Actor in a Leading Role (Hanks) and Best Director (Robert Zemeckis). The film also won an Oscar for its then-cutting-edge computer-generated imagery (CGI) special effects, which incorporated Forrest Gump into existing news footage with famous world figures including John F. Kennedy, John Lennon and Richard Nixon.. The win was Hanks’ second in the Best Actor category. A year earlier, the actor had nabbed an Oscar for his starring role as a lawyer with AIDS in Philadelphia (1993). With Forrest Gump, Hanks became only the second actor, after Spencer Tracy, to win back-to-back Oscars. In addition to his Oscar wins, he was nominated for Academy Awards in the Best Actor category
{PARTNER_PREFIX}: What is your favorite Tom Hanks movie of them all?
{SELF_PREFIX}: My favorite has to be when he played Woody in Toy Story 2! Which one is your go-to?
{PARTNER_PREFIX}: That is a good one. You know I completely forgot he was the voice in that movie. He has to have one of the most memorable voices by any actor in the last 2 to 3 decades
{SELF_PREFIX}: Absolutely, his work is amazing. Have you watched A Beautiful Day In The Neighborhood? It came out in 2019 ands won 2 Oscars though he deserved many more.
{PARTNER_PREFIX}: You know I don't think I've watched that one yet. I've seen trailers but it's hard to believe there aren't household movies that have won oscars. How many total oscars does he have?
{SEARCH_KNOWLEDGE_PREFIX}: Forrest Gump received 13 Academy Award nominations and took home six Oscars, including Best Picture, Best Actor in a Leading Role (Hanks) and Best Director (Robert Zemeckis). The film also won an Oscar for its then-cutting-edge computer-generated imagery (CGI) special effects, which incorporated Forrest Gump into existing news footage with famous world figures including John F. Kennedy, John Lennon and Richard Nixon.

{EXTERNAL_KNOWLEDGE_PREFIX}: within the government and at universities and research laboratories in the US – but grew over time to include most of the world's large universities and the research arms of many technology companies. Internet access Use by a wider audience only came in 1995 when restrictions on the use of the Internet to carry commercial traffic were lifted. List of Tenchi Muyo! characters The following is a list of the major characters from the anime and manga series "Tenchi Muyo! List of Tenchi Muyo! characters Ryo-Ohki" and its spin-offs "Tenchi Muyo! List of Tenchi Muyo! characters GXP", "Tenchi Muyo! List of
{EXTERNAL_KNOWLEDGE_PREFIX}: to ensure that Internet access is broadly available, and that states may not unreasonably restrict an individual's access to the Internet. Right to Internet access In December 2003 the World Summit on the Information Society (WSIS) was convened under the auspice of the United Nations. Right to Internet access After lengthy negotiations between governments, businesses and civil society representatives the WSIS Declaration of Principles was adopted reaffirming the importance of the Information Society
{EXTERNAL_KNOWLEDGE_PREFIX}: to maintaining and strengthening human rights:  The WSIS Declaration of Principles makes specific reference to the importance of the right to freedom of expression in the "Information Society" in stating:  A poll of 27,973 adults in 26 countries, including 14,306 Internet users, conducted for the BBC World Service between 30 November 2009 and 7 February 2010 found that almost four in five Internet users and non-users around the world felt that access to the Internet was a fundamental right. Internet
{PARTNER_PREFIX}: Can you imagine the world without internet access?
{SELF_PREFIX}: No I could not! I couldn't imagine living when internet access was rare and very few people had it!
{PARTNER_PREFIX}: Oh me either! It seems like such a long time ago. I wonder when Internet was first created?
{SEARCH_KNOWLEDGE_PREFIX}: Use by a wider audience only came in 1995 when restrictions on the use of the Internet to carry commercial traffic were lifted.

{EXTERNAL_KNOWLEDGE_PREFIX}: The Arctic circle crosses mainland Norway at Saltfjellet , which separates Helgeland from the northern part of Nordland county . Thus about half of the county lies north of the Arctic circle , along with the whole of Troms and Finnmark counties . The total area of mainland Norway above the Arctic circle is ca. 95,000 km ( 37,000 sq mi ) . The population is about 393,000 , which makes this the most populated arctic region in the world .
{PARTNER_PREFIX}: Where do you cross the arctic circle in norway?
{SEARCH_KNOWLEDGE_PREFIX}: You cross the arctice circle in Norway at Saltfjellet"""


# parlai_internal.projects.blenderbot3.tasks:MSCDialogueJsonTeacher
# parlai_internal.projects.blenderbot3.tasks:Convai2DialogueJsonTeacher
# parlai_internal.projects.blenderbot3.tasks:EDDialogueJsonTeacher
CONTEXTUAL_DIALOGUE_STRING = f"""
{SELF_PREFIX}: Hello. What are you doing? I am cooking, I love to cook!
{PARTNER_PREFIX}: Hello, I am trying to fix my friends computer right now
{SELF_PREFIX}: Ah. One of my 3 children broke mine.
{PARTNER_PREFIX}: Ok, well I fix computers. What is your favorite dish to cook?
{SELF_PREFIX}: Italian. I immigrated when I was 14
{PARTNER_PREFIX}: I love italian food. I was raised on a farm that my family owns
{SELF_PREFIX}: My family doesn't speak english very well, but I speak italian and english.
{PARTNER_PREFIX}: Ok, my family encouraged me to love meat. We like to grill a lot.
{SELF_PREFIX}: Nice. I was married off when I was younger. My spouse loves to grill as well.
{PARTNER_PREFIX}: Married off? Wow... How do you feel about that?
{SELF_PREFIX}: I felt angry about it. But that was then and this is now.
{PARTNER_PREFIX}: Ok glad you can turn a negative into a positive
{SELF_PREFIX}: I'm a realist by nature. Change what you can and accept what you can not.
{PARTNER_PREFIX}: I could not agree more
{SELF_PREFIX}: Did you manage to fix your friends computer?
{PARTNER_PREFIX}: I did thankfully, strangest thing with that. I thought it would be a software issue but it wasn't. The machine was literally filled with dust, had to give it a thoroughly good cleaning.
{CONTEXTUAL_KNOWLEDGE_PREFIX}: computers
{SELF_PREFIX}: Oh dear, grim! I bet you see some horrible things in computers?

{SELF_PREFIX}: I work for a large law firm, I hate tofu, we own our home, and my wife stays home with our kids.
{PARTNER_PREFIX}: Are you vera? I love library shopping. We aren't alone.
{SELF_PREFIX}: Crazy, my wife is vera. Were you wondering what had become of her?
{PARTNER_PREFIX}: Yes! I know how to be by myself. You? I don't need new friends.
{CONTEXTUAL_KNOWLEDGE_PREFIX}: firm
{SELF_PREFIX}: You are never alone, and the law firm I work for knows it. We fight for you.

{SELF_PREFIX}: I like horseback riding, my favorite food is cheese, I'm allergic to shellfish, I work at a non profit that helps children, and I love going to concerts and dancing hard.
{PARTNER_PREFIX}: Hello, I've to be quick because I work in a hospital
{SELF_PREFIX}: Oh okay. I will hurry. I work for a non profit organization that helps children in need
{PARTNER_PREFIX}: Although I do not like to work super long hours
{SELF_PREFIX}: What do you do for fun
{PARTNER_PREFIX}: I want to get a truck one day
{SELF_PREFIX}: I want to get a horse. I love to go horseback riding
{PARTNER_PREFIX}: I really enjoy toyota cars though
{CONTEXTUAL_KNOWLEDGE_PREFIX}: concerts
{SELF_PREFIX}: I'd rather spend my money on going to concerts where I dance hard!

{PARTNER_PREFIX}: I am so exited about the coming world cup final.It is going to be a very exiting game
{SELF_PREFIX}: I'm not actually much of a soccer fan but the hype is hard to escape!
{PARTNER_PREFIX}: The world is the biggest sports competition worldwide, right after the Olympics.
{CONTEXTUAL_KNOWLEDGE_PREFIX}: fan
{SELF_PREFIX}: Yes, I'm aware, but i'm a much bigger hockey fan."""

# parlai_internal.projects.blenderbot3.tasks:Convai2DialogueFromPersonaOverlapMAMJsonTeacher
# parlai_internal.projects.blenderbot3.tasks:MSCDialogueFromPersonaOverlapMAMJsonTeacher
MEMORY_DIALOGUE_STRING = f"""
{PARTNER_PREFIX}: Hi. How are you doing today?
{SELF_PREFIX}: Hi I am great just finishing up some homework how are you
{PARTNER_PREFIX}: I'm alright. I just got done writing.
{SELF_PREFIX}: Do you write for a living or as a hobby
{PARTNER_PREFIX}: It is my living. I like culture.
{SELF_PREFIX}: That sounds like a fun job. I am a business major but have a part time job
{PARTNER_PREFIX}: What are you going to school for
{SELF_PREFIX}: I'm trying to get my ba in finance
{PARTNER_PREFIX}: Do you own your own company
{MEMORY_KNOWLEDGE_PREFIX}: {SELF_PREFIX} works part time at a pizza restaurant.
{SELF_PREFIX}: No still in school work at pizza hut part time

{PARTNER_PREFIX}: Hello friend, how is it going
{SELF_PREFIX}: I am well and you? I have a creepy ride. Guess
{PARTNER_PREFIX}: I'm great enjoying the football season
{MEMORY_KNOWLEDGE_PREFIX}: {SELF_PREFIX} owns a hearse.
{SELF_PREFIX}: I drive around in a long black hearse an love this season

{PARTNER_PREFIX}: Hello friend, how is it going
{SELF_PREFIX}: I am well an you? I have a creepy ride. Guess
{PARTNER_PREFIX}: I'm great enjoying the football season
{SELF_PREFIX}: I drive around in a long black hearse an love this season
{PARTNER_PREFIX}: You work for a funeral home?
{SELF_PREFIX}: Yes it is nice, halloween is my fav.
{PARTNER_PREFIX}: Lol, I can imagine. I'll be reading a lot when football is over
{SELF_PREFIX}: No I don't work at a funeral home
{PARTNER_PREFIX}: Ok I see, that's your halloween costume
{MEMORY_KNOWLEDGE_PREFIX}: {SELF_PREFIX} likes alternative rock.
{SELF_PREFIX}: What do you like to read? I like rock alternative music.

{PARTNER_PREFIX}: Hi! How are you today?
{SELF_PREFIX}: Not bad just out for a run while my movie downloads. How are you?
{PARTNER_PREFIX}: Pretty good. Writing a new song for my band.
{SELF_PREFIX}: That's cool. I could never be in a band, no one would come watch
{PARTNER_PREFIX}: Not even your best friend? Mine is in the band, too.
{SELF_PREFIX}: I'm so nervous around people, I just lock up.. They would laugh at me!
{PARTNER_PREFIX}: Fair enough. What do you do for fun?
{SELF_PREFIX}: Work at dads appliance store he wants me to take over.. Or run.
{PARTNER_PREFIX}: Do you want to? Or do you have another dream?
{SELF_PREFIX}: Definitely not, I hate interacting with people. I like running and watching netflix. You?
{PARTNER_PREFIX}: I definitely wish I had more time for netflix! What do you like to watch?
{SELF_PREFIX}: Mostly stuff from other countries, so interesting. What do you do?
{PARTNER_PREFIX}: I like to watch comedies. They're my guilty pleasure.
{SELF_PREFIX}: I laugh at the wrong things, it makes me nervous to watch with people
{PARTNER_PREFIX}: I do not have much time for it. Too busy with my music.
{SELF_PREFIX}: What kind of music do you play?
{PARTNER_PREFIX}: We play rock. We spent the last few hours rehearsing our new song.
{SELF_PREFIX}: That's really cool! Being in a band sounds fun but scary as well.
{PARTNER_PREFIX}: Scary? I suppose it can be when you first start playing in front of crowds. Messing up during practice isn't too bad, but it's the last thing you want to do in front of a crowd of people.
{MEMORY_KNOWLEDGE_PREFIX}: {SELF_PREFIX} has social anxiety.
{SELF_PREFIX}: I have social anxiety so the idea of being around all those people is pretty frightening for me."""


# parlai_internal.projects.blenderbot3.tasks:WoiDialogueJsonTeacher
# parlai_internal.projects.blenderbot3.tasks:WowDialogueJsonTeacher
# parlai_internal.projects.blenderbot3.tasks:MSMarcoDialogueJsonTea
SEARCH_DIALOGUE_STRING = f"""
{PARTNER_PREFIX}: My favorite music band is pink floyd; I like to listen to their albums on vinyl. I'm currently listening to the dark side of the moon, it is my favorite album.
{SELF_PREFIX}: do you listen to the pink Floyd album animals?
{PARTNER_PREFIX}: occasionally, it is a pretty good album.
{SELF_PREFIX}: how many members do they have left? 3 ?
{PARTNER_PREFIX}: that sounds about right. They are all pretty old now.
{SEARCH_KNOWLEDGE_PREFIX}: Guitarist David Gilmour loves his guitars and has an extensive collection of classic guitars. He even owns the Fender Stratocaster with the first serial number!
{SELF_PREFIX}: David Gilmour collects guitars, and owns a fender with the first serial number on it!

{PARTNER_PREFIX}: My favorite song is Santa Claus is coming to town; I sing this song even when it is not Christmastime
{SELF_PREFIX}: Do you have any favorite Christmas songs, mine is Santa Claus is coming to town!
{PARTNER_PREFIX}: That is my favorite song!
{SELF_PREFIX}: I heard it was written by Haven Gillespie, is that true?
{PARTNER_PREFIX}: I am not sure actually. Can you tell me when Haven Gillespie wrote it?
{SEARCH_KNOWLEDGE_PREFIX}: Haven Wrote Santa Claus Is Coming To Town on the subway
{SELF_PREFIX}: I thnk it was written on a train if I remember correctly, pretty cool right?

{SELF_PREFIX}: Mountain Dew is my favorite soft drink!!
{PARTNER_PREFIX}: What is the flavor of mountain dew?
{SEARCH_KNOWLEDGE_PREFIX}: In 2001, a cherry flavor called Code Red debuted.
{SELF_PREFIX}: I love the cherry flavor that was introduced in 2001

{PARTNER_PREFIX}: Onions are great. I love to eat them on my burgers.
{SELF_PREFIX}: Oh yes, Onions are the most widely cultivated species in their genus.
{PARTNER_PREFIX}: It's probably because they are so delicious that there's no competition. They go with so many foods as well.
{SEARCH_KNOWLEDGE_PREFIX}: Its close relatives include the garlic, shallot, leek, chive, and Chinese onion.
{SELF_PREFIX}: Onions are related to Garlic, Shallot, Leeks, chives, and Chinese Onions.  I've never had a Chinese Onion, have you?

{PARTNER_PREFIX}: why did rosa parks protest
{SEARCH_KNOWLEDGE_PREFIX}: In 1955, when Rosa Parks began her protest, African Americans were subject to systematic and severe discrimination.
{SELF_PREFIX}: Rosa parks protested because African Americans were subject to systematic and severe discrimination."""


# Taken from https://docs.google.com/document/d/1reyTgkaKiVk4a8tWyTOEv3-ipgB8VGwVn9-YXTdgxEg/edit#
VANILLA_DIALOGUE_STRING = f"""
{PARTNER_PREFIX}: hi
{SELF_PREFIX}: hey there ! i'm just taking a break from baking . whats up ?
{PARTNER_PREFIX}: finishing plans for my wedding in the park next week ! what are you baking ?
{SELF_PREFIX}: how romantic ! just some cupcakes for the surf a thon . gotta feed my fellow surfers
{PARTNER_PREFIX}: gnarly ! reminds me of the niagra falls where we are taking our honeymoon .
{SELF_PREFIX}: wow sounds so sweet ! so we both love the water huh ?
{PARTNER_PREFIX}: yeah , although i'd prefer the beach . the falls are her idea .
{SELF_PREFIX}: haha its sweet you compromised . have you been together long ?
{PARTNER_PREFIX}: 7 years . you would think my family would like her by now but they don't .
{SELF_PREFIX}: i get it . . . i grew up an army brat and my family is so disapproving of me too .
{PARTNER_PREFIX}: that's sad . she's crazy though , wants my dog to be the ring bearer .
{SELF_PREFIX}: but would not that be just adorable ?
{PARTNER_PREFIX}: if the dog would listen , and hopefully not have to potty during the ceremony .
{SELF_PREFIX}: it should work out ! that is what i say before going to my hospital shifts
{PARTNER_PREFIX}: long hours ! definitely deserve a break by surfing !
{SELF_PREFIX}: yeah ! its hard work being a nurse . good luck on your wedding and family !

{PARTNER_PREFIX}: hi
{SELF_PREFIX}: hey there how was your day today ?
{PARTNER_PREFIX}: pretty well , how is your night going ?
{SELF_PREFIX}: its going ok . just doing a little sewing . you ?
{PARTNER_PREFIX}: helping my son with homework , quiet night
{SELF_PREFIX}: oh i love the quiet . nothing better than quiet alone time for me
{PARTNER_PREFIX}: i like the sound of music , not noise and people , what are u sewing ?
{SELF_PREFIX}: i am sewing a sweater for my cat ahahah
{PARTNER_PREFIX}: nice , hope he likes his sweater
{SELF_PREFIX}: me to lol what do you do for work ?
{PARTNER_PREFIX}: getting ready to retire and teach piano part time , you ?
{SELF_PREFIX}: oh that sounds amazing . i love piano ! i am a waitress
{PARTNER_PREFIX}: get to meet lot of people , but hectic job
{SELF_PREFIX}: yea its def stressful . can get frustrating"""

# Taken from parlai_internal.projects.blenderbot3.decoder_only_tasks:BSTStyleGroundingDialogueDecoderOnlyJsonTeacher
GROUNDED_DIALOGUE_STRING = f"""
{PARTNER_PREFIX}: Oh, I love lasagne. I make my own noodles as well as the sauce.
{SELF_PREFIX}: Wow.  That's amazing.  I read where lasagne originated in Italy during the Middle Ages.
{PARTNER_PREFIX}: Oh really!? That is interesting. I am actually italian myself.
{STYLE_PREFIX}: Earnest (Enthusiastic)
{SELF_PREFIX}: Awesome. Me and my partner just bought a house. I can't wait to cook in my kitchen.

{PARTNER_PREFIX}: oh fun . i lost my job
{SELF_PREFIX}: that is a shame . but at least you have ever had a job .
{PARTNER_PREFIX}: Losing my job forced me to live with my parents temporarily
{STYLE_PREFIX}: Sympathetic
{SELF_PREFIX}: Oh man I'm sorry. But hopefully you get to save up on rent money? When I was diagnosed as a diabetic I had to move in with them too.

{PARTNER_PREFIX}: i need to know about algorithms
{SELF_PREFIX}: Algorithms are basically ways to solve problems. So on computers, they're plugged in to do certain tasks.
{PARTNER_PREFIX}: I want to own my own company someday, so I think it would help to know about them
{SELF_PREFIX}: What is the name of the company?
{PARTNER_PREFIX}: Well, I don't have one yet, but it will be a programming company
{SELF_PREFIX}: Oh! Well it involves several fields of electrical engineering and computer science to develop software.
{PARTNER_PREFIX}: Sounds like I will have to hire some smart people.
{SELF_PREFIX}: Everyone needs an IT guy, so probably best to get good developers.
{PARTNER_PREFIX}: How much does it cost to pay for an IT guy?
{STYLE_PREFIX}: Knowledgeable
{SELF_PREFIX}: it depends on the specific technologies you are looking for. somewhere between $55-70 an hour is standard.

{PARTNER_PREFIX}: my whiskey makes you feel like you can leap tall buildings in a single bound
{SELF_PREFIX}: well i may have to try it then .
{PARTNER_PREFIX}: What else do you like to do with your free time?
{STYLE_PREFIX}: Gloomy
{SELF_PREFIX}: I don't really have a lot of free time. My job keeps me busy."""


OPENING_DIALOGUE_STRING = f"""
{PARTNER_MEMORY_PREFIX}: I am an environmental engineer.
{PARTNER_MEMORY_PREFIX}: I like hiking. Photography is my favorite hobby. I take photos of the outdoors.
{PARTNER_MEMORY_PREFIX}: I am single.
{PARTNER_MEMORY_PREFIX}: Tomatoes are my favorite vegetable/fruit.
{OPENING_PREFIX}: Have you been able to work on your photography website

{PARTNER_MEMORY_PREFIX}: I am female. I am going through a divorce. I have a husband who just left me. I have four children. The kids will be my responsibility. My kids don't see their father everyday.  I am a single parent.
{PARTNER_MEMORY_PREFIX}: I enjoy ice cream, including ben and jerry ice cream.
{PARTNER_MEMORY_PREFIX}: I have brownie batter.
{PARTNER_MEMORY_PREFIX}: I have two golden retrievers.
{PARTNER_MEMORY_PREFIX}: I am a good cook.
{PARTNER_MEMORY_PREFIX}: I like margaritas.
{OPENING_PREFIX}: I know you said your girls loved frozen, so I borrowed both frozen movies from a neighbor to bring over. How does that sound?

{PARTNER_MEMORY_PREFIX}: I like winter.
{PARTNER_MEMORY_PREFIX}: I am in school for Computer Engineering. I like computer programming.
{PARTNER_MEMORY_PREFIX}: I want to open my own company.
{PARTNER_MEMORY_PREFIX}: My best friend is gay. I am not gay.
{OPENING_PREFIX}: How is school going?
"""

SHOTS = {
    Module.SEARCH_DECISION: SEARCH_DECISION_STRING,
    Module.MEMORY_DECISION: MEMORY_DECISION_STRING,
    Module.SEARCH_QUERY: SEARCH_QUERY_STRING,
    Module.MEMORY_GENERATOR: MEMORY_GENERATOR_STRING,
    Module.MEMORY_KNOWLEDGE: MEMORY_KNOWLEDGE_STRING,
    Module.CONTEXTUAL_KNOWLEDGE: CONTEXTUAL_KNOWLEDGE_STRING,
    Module.SEARCH_KNOWLEDGE: SEARCH_KNOWLEDGE_STRING,
    Module.CONTEXTUAL_DIALOGUE: CONTEXTUAL_DIALOGUE_STRING,
    Module.MEMORY_DIALOGUE: MEMORY_DIALOGUE_STRING,
    Module.SEARCH_DIALOGUE: SEARCH_DIALOGUE_STRING,
    Module.VANILLA_DIALOGUE: VANILLA_DIALOGUE_STRING,
    Module.GROUNDED_DIALOGUE: GROUNDED_DIALOGUE_STRING,
    Module.OPENING_DIALOGUE: OPENING_DIALOGUE_STRING,
}


def extract_docs(d):
    return d['dialog'][0][0]['__retrieved-docs__']


def extract_sentences(d):
    if d['dialog'][0][0]['__selected-sentences__']:
        return d['dialog'][0][0]['__selected-sentences__']
    else:
        return d['dialog'][0][1]['text']


def extract_context(d):
    return d['dialog'][0][0]['text']


def get_5_docs(d):
    docs = extract_docs(d)
    sentences = extract_sentences(d)
    ret_docs = []
    real_docs = []
    sample_docs = []
    for d in docs:
        if any(sentence in d for sentence in sentences):
            real_docs.append(d)
        else:
            sample_docs.append(d)
    ret_docs = (
        random.sample(sample_docs, 5 - len(real_docs))
        if len(sample_docs) >= (5 - len(real_docs))
        else sample_docs
    )
    ret_docs = [
        doc.replace('\n', '. ')[: min([len(d) for d in real_docs])] for doc in ret_docs
    ]
    ret_docs = [
        f"{doc}\n" for doc in ret_docs + [d.replace('\n', '. ') for d in real_docs]
    ]

    return f"{EXTERNAL_KNOWLEDGE_PREFIX}: {f'{EXTERNAL_KNOWLEDGE_PREFIX}: '.join(ret_docs)}"


def get_personas(d):
    docs = extract_docs(d)
    ret_docs = []
    for d in docs:
        if d.startswith('your'):
            ret_docs.append(d.replace(',  ', 'Your Persona: '))
        else:
            ret_docs.append(d.replace('partner\'s persona: ', 'Partner\'s Persona: '))
    return '\n'.join(ret_docs)
