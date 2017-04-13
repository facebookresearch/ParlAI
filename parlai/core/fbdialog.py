# Copyright 2004-present Facebook. All Rights Reserved.
"""This module provides access to data in the Facebook Dialog format.

The way FB Dialog data is set up is as follows:

1 Sam went to the kitchen.
2 Pat gave Sam the milk.
3 Where is the milk?<TAB>kitchen<TAB>1<TAB>hallway|kitchen|bathroom
4 Sam went to the hallway
5 Pat went to the bathroom
6 Where is the milk?<TAB>hallway<TAB>1<TAB>hallway|kitchen|bathroom

Lines 1-6 represent a single episode, with two different examples: the first
    example is lines 1-3, and the second is lines 4-6.
Lines 1,2,4, and 5 represent contextual information.
Lines 3 and 6 contain a query, a label, a reward for getting the question
    correct, and three candidates.
Since both of these examples are part of the same episode, the information
    provided in the first example is relevant to the query in the second example
    and therefore the agent must remember the first example in order to do well.
In general dialog in this format can be any speech, not just QA pairs:

1 Hi how's it going?<TAB>It's going great. What's new?
2 Well I'm working on a new project at work.<TAB>Oh me too!
3 Oh cool!<TAB>Tell me about yours.
etc.
"""

from .dialog import DialogTeacher


class FbDialogTeacher(DialogTeacher):
    """Subclasses DialogTeacher for functionality and provides an implementation
    of setup_data which iterates over datasets in the "fbdialog" format.
    """

    def __init__(self, opt, shared=None):
        self.cloze = opt.get('cloze', False)
        self.cands = self.load_cands(opt.get('cands_datafile', None))
        super().__init__(opt, shared)

    def candidates(self):
        return self.cands

    def load_cands(self, path):
        """Load global fixed set of candidate labels that the teacher provides
    every example (the true labels for a specific example are also added to this set,
    so that it's possible to get the right answer).
    """
        if path is None:
            return None
        cands = []
        lines_have_ids = False
        cands_are_replies = False
        cnt = 0
        with open(path) as read:
            for line in read:
                line = line.strip()
                if len(line) > 0:
                    cnt = cnt + 1
                    # If lines are numbered we stip them of numbers.
                    if cnt == 1 and line[0:2] == '1 ':
                        lines_have_ids = True
                    # If tabs then the candidates are all the replies.
                    if '\t' in line and not cands_are_replies:
                        cands_are_replies = True
                        cands = []
                    if lines_have_ids:
                        space_idx = line.find(' ')
                        line = line[space_idx + 1:]
                        if cands_are_replies:
                            sp = line.split('\t')
                            if len(sp) > 1 and sp[1] != '':
                                cands.append(sp[1])
                        else:
                            cands.append(line)
                    else:
                        cands.append(line)
        return cands

    def setup_data(self, path):
        """Reads data in the fbdialog format.
        Returns ((x,y,r,c), new_episode?) tuples.
        x represents a query, y represents the labels, r represents any reward,
        and c represents any candidates.

        The example above will be translated into the following tuples:

        x: 'Sam went to the kitchen\Pat gave Sam the milk\nWhere is the milk?'
        y: ['kitchen']
        r: '1'
        c: ['hallway', 'kitchen', 'bathroom']
        new_episode = True (this is the first example in the episode)

        x: 'Sam went to the hallway\nPat went to the bathroom\nWhere is the
            milk?'
        y: ['hallway']
        r: '1'
        c: ['hallway', 'kitchen', 'bathroom']
        new_episode = False (this is the second example in the episode)
        """
        print("[loading fbdialog data:" + path + "]")
        with open(path) as read:
            start = True
            x = ''
            reward = None

            for line in read:
                line = line.strip()
                if len(line) == 0:
                    continue

                # first, get conversation index -- '1' means start of episode
                space_idx = line.find(' ')
                conv_id = line[:space_idx]

                # split line into constituent parts, if available:
                # x<tab>y<tab>reward<tab>candidates
                # where y, reward, and candidates are optional
                split = line[space_idx + 1:].split('\t')

                # remove empty items and strip each one
                for i in range(len(split)):
                    word = split[i].strip()
                    if len(word) == 0:
                        split[i] = ''
                    else:
                        split[i] = word

                # now check if we're at a new episode
                if conv_id == '1':
                    x = x.strip()
                    if x:
                        yield [x, None, reward], start
                    start = True
                    # start a new episode
                    if self.cloze:
                        x = 'Fill in the blank in the last sentence.\n{x}'.format(
                            x=split[0]
                        )
                    else:
                        x = split[0]
                else:
                    if x:
                        # otherwise add current x to what we have so far
                        x = '{x}\n{next_x}'.format(x=x, next_x=split[0])
                    else:
                        if len(split) > 2:
                            reward = split[2]
                        x = split[0]

                if len(split) > 1 and split[1]:
                    # only generate an example if we have a y
                    split[0] = x
                    # split labels
                    split[1] = split[1].split('|')
                    if len(split) > 3:
                        # split candidates
                        split[3] = split[3].split('|')
                    if start:
                        yield split, True
                        start = False
                    else:
                        yield split, False
                    # reset x in case there is unlabeled data still left
                    x = ''
                    reward = None
