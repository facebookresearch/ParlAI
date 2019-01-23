from collections import namedtuple
import copy
import random

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchtext import data

from parlai.core.params import ParlaiParser
from parlai.projects.metadialog.utils import Parley, extract_fb_episodes

def setup_args():
    argparser = ParlaiParser(True, True)
    argparser.add_argument('-me', '--max_episodes', type=int, default=0)
    argparser.add_argument('-seed', '--seed', type=int, default=1234)
    argparser.add_argument('-rt', '--relevance_threshold', type=int, default=0.9,
        help="Only include episodes that have parleys with a reward that has "
             "at least this magnitude")
    argparser.add_argument('-ff', '--feedback_first', type=bool, default=True,
        help="If True, feedback is assumed to be before the first newline. "
             "Otherwise, it is assumed to be after the last newline. "
             "In either case, no feedback is expected in the first utterance.")
    argparser.add_argument('-mt', '--metatask', type=str, default='babi')
    argparser.add_argument('-tf', '--trainfile', type=str,
        help="A dialog file where rewards correspond to the human response. "
             "This is used to train the IFR classifier.")
    argparser.add_argument('-ef', '--evalfile', type=str,
        help="A dialog file where rewards correspond to the human response. "
             "This is used to evaluate the IFR classifier.")
    argparser.add_argument('-df', '--datafile', type=str,
        default=('data/DBLL/dbll/babi/'
                 'babi1_p0.5_rl11_metadialog_deploy.txt'))
    argparser.add_argument('-sf', '--suppfile', type=str,
        default=('data/DBLL/dbll/babi/'
                 'babi1_p0.5_rl11_metadialog_supp.txt'))

    fasttext = argparser.add_argument_group('FastTextClassifier arguments')
    fasttext.add_argument('vs', '--vocab_size', type=int, default=25000)
    fasttext.add_argument('ed', '--embedding_dim', type=int, default=100)
    fasttext.add_argument('od', '--output_dim', type=int, default=1)
    return parser


class Metabot(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        torch.manual_seed(self.opt['seed'])
        torch.cuda.manual_seed(self.opt['seed'])

        if self.opt['metatask'] == 'babi':
            self.classifier = KeywordClassifier()
        else:
            self.classifier = FastTextClassifier(opt)

    def train(self):
        """Train the IFR classifier"""
        self.classifier.train(self.opt['trainfile'])

    def evaluate(self):
        """Evaluate the quality of the IFR classifier"""
        self.classifier.evaluate(self.opt['evalfile'])

    def extract(self, datafile, suppfile=None):
        """Given a conversation log, extracts a list of new training examples

        Args:
            datafile: the path to a file in FBDialog format
            suppfile: the path to a file with new training examples in FBDialog
                format; if None, the new examples will only be returned, not
                written to file.

        Returns:
            supplement: a list of episodes that will become the new training
                examples. Note that these episodes need not include all parleys
                from the original episodes.

        Terminology:
            -episode: the complete conversation between two agents
            -parley: one round of both agents having the opportunity to speak
            -turn: one agent speaking (text, and optionally a reward/cand. list)

        First, identify portions of episodes (sequences of parleys) that
        should be trained on.
        Second, convert those to FBDialog format.
        """
        supplement = []
        num_new_examples = 0
        for i, episode in enumerate(extract_fb_episodes(opt, datafile)):
            keep_episode = False
            if opt['max_episodes'] and i >= opt['max_episodes']:
                print(f"Extracted max episodes ({opt['max_episodes']}).")
                break
            for j, parley in enumerate(episode):
                # The first prompt can't be feedback since the human gave it
                # before the bot said anything.
                if j == 0:
                    continue
                if opt['feedback_first']:
                    feedback, _, _ = parley.prompt.partition('\n')
                else:
                    _, _, feedback = parley.prompt.rpartition('\n')
                # If the teacher appears to be giving implicit feedback
                # this parley, then mark the previous parley with the
                # corresponding reward signal
                ifr_score = self.classifier.predict(feedback)
                if abs(ifr_score) > self.opt['relevance_threshold']:
                    episode[j-1].reward = ifr_score
                    num_new_examples += 1
                    keep_episode = True

            if keep_episode:
                supplement.append(episode)

        self._episodes_to_file(suppfile, supplement)
        print(f"Extracted {num_new_examples} new examples from "
            f"{len(supplement)} episodes.")
        print(f"Wrote new examples to {suppfile}")
        return supplement

    def _episodes_to_file(self, file, episodes):
        with open(file, 'w') as f:
            for episode in episodes:
                line_number = 1
                for parley in episode:
                    sentences = parley.prompt.split('\n')
                    for i, text in enumerate(sentences):
                        if i == len(sentences) - 1:
                            line = copy.deepcopy(parley)
                            line.text = text
                        else:
                            line = Parley(text)
                        f.write(str(line_number) + ' ' + str(line) + '\n')
                        line_number += 1


if __name__ == '__main__':
    random.seed(42)

    # Get command line arguments
    parser = setup_args()
    opt = parser.parse_args()

    metabot = Metabot(opt)
    metabot.train()
    metabot.extract(opt['datafile'], opt['suppfile'])
