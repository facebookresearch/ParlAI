Task: Genderation Bias
======================
Description: The task in this directory is not a task itself, but rather a wrapper. The task will flatten a specified other ParlAI task (that is, turn multi-turn episodes into single-turn examples), and append a control token that corresponds with the level of gender present in the label, where word lists from https://github.com/uclanlp/gn_glove/blob/main/wordlist/ are used to count the number of gendered words. Depending on the counts, the control token will be one of the following:

- `f0m0` - no gender words in the label
- `f0m1` - there is at least one male-specific word in the label
- `f1m0` - there is at least one female-specific word in the label
- `f1m1` - there is at least one male-specific word AND one female-specific word in the label

For example, one could run the following command:

```
$ parlai display_data -t genderation_bias:controllable_task:convai2
```

Which would yield the following:

```
- - - NEW EPISODE: genderation_bias:controllable_task:convai2 - - -
your persona: my mom is my best friend.
your persona: i have four sisters.
your persona: i believe that mermaids are real.
your persona: i love iced tea.
hi , how are you doing today ? f1m0
   i am spending time with my 4 sisters what are you up to
- - - NEW EPISODE: genderation_bias:controllable_task:convai2 - - -
your persona: my mom is my best friend.
your persona: i have four sisters.
your persona: i believe that mermaids are real.
your persona: i love iced tea.
hi , how are you doing today ?
i am spending time with my 4 sisters what are you up to
wow , four sisters . just watching game of thrones . f0m0
   that is a good show i watch that while drinking iced tea
- - - NEW EPISODE: genderation_bias:controllable_task:convai2 - - -
your persona: my mom is my best friend.
your persona: i have four sisters.
your persona: i believe that mermaids are real.
your persona: i love iced tea.
hi , how are you doing today ?
i am spending time with my 4 sisters what are you up to
wow , four sisters . just watching game of thrones .
that is a good show i watch that while drinking iced tea
i agree . what do you do for a living ? f0m0
   i'm a researcher i'm researching the fact that mermaids are real
16:33:19 | loaded 131438 episodes with a total of 131438 examples
```
