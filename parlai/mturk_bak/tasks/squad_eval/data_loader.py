# Copyright 2004-present Facebook. All Rights Reserved.

import json
import random
from parlai.tasks.squad.build import build


class DataLoader():
    """
    Data loader for squad eval task, which loads the json squad data and randomly
    choose a wikipedia paragraph and its QAs as the conversation context and the 
    teacher question.
    """

    def __init__(self, opt):
        # self.datatype = opt['datatype']
        # build(opt)
        # if opt['datatype'].startswith('train'):  
        #     suffix = 'train'
        # else:
        #     suffix = 'dev'
        # datapath = (
        #     opt['datapath'] + 'SQuAD/' +
        #     suffix + '-v1.1.json')
        build(opt)
        # TODO: should we use both train and dev data here?
        datapath = opt['datapath'] + 'SQuAD/train-v1.1.json'
        self.data = self._setup_data(datapath)
        self.episode_idx = -1

    def _setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        self.len = 0
        self.examples = []
        for article_idx in range(len(self.squad)):
            article = self.squad[article_idx]
            for paragraph_idx in range(len(article['paragraphs'])):
                paragraph = article['paragraphs'][paragraph_idx]
                num_questions = len(paragraph['qas'])
                self.len += num_questions
                for qa_idx in range(num_questions):
                    self.examples.append((article_idx, paragraph_idx, qa_idx))

    def load_context(self, conversation_id):
        self.episode_idx = random.randrange(len(self.examples))
        article_idx, paragraph_idx, qa_idx = self.examples[self.episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        question = qa['question']
        answers = [a['text'] for a in qa['answers']]
        context = paragraph['context']

        return {
            'text': context + '\n' + question,
            'labels': answers
        }

        return context