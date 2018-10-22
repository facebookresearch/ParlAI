# FromFile Task Agent

If you have a dialogue, QA or other text-only dataset that you can put in a text file in the format we will now describe, you can just load it directly from there, with no extra code!
Use the fromfile agent:

python parlai/scripts/display_data.py -t fromfile:parlaiformat --fromfile_datapath /tmp/data.txt
<.. snip ..>
[creating task(s): fromfile:parlaiformat]
[loading parlAI text data:/tmp/data.txt]
[/tmp/data.txt]: hello how are you today?
[labels: i'm great thanks! what are you doing?]
   [RepeatLabelAgent]: i'm great thanks! what are you doing?
~
[/tmp/data.txt]: i've just been biking.
[labels: oh nice, i haven't got on a bike in years!]
   [RepeatLabelAgent]: oh nice, i haven't got on a bike in years!
- - - - - - - - - - - - - - - - - - - - -
~
EPOCH DONE
[ loaded 1 episodes with a total of 2 examples ]
