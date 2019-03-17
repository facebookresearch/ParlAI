import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

import spacy
nlp = spacy.load('en_core_web_sm')

# from spacy.tokenizer import Tokenizer
# tokenizer = Tokenizer(nlp.vocab)
from spacy.lang.en import English
tokenizer = English().Defaults.create_tokenizer(nlp)




class IDFScorer(): 

    def my_spacy_tokenizer(self, doc):
        tokens = tokenizer(doc)
        return([token.text for token in tokens])
    
    def __init__(self, opt):
        
        self.vectorizer = TfidfVectorizer(lowercase=True, 
                                            tokenizer=self.my_spacy_tokenizer, 
                                            stop_words=None, 
                                            use_idf=True)


        datasetname = opt['task']
        train_filename = 'data/%s/%s/train.tsv' % (datasetname, datasetname)
        valid_filename = 'data/%s/%s/valid.tsv' % (datasetname, datasetname)


        convo_docs = []

        for filename in [train_filename, valid_filename]:
    
            lines = open(filename, 'r').readlines()
            convo_doc = ''
            prev_turn = -1

            for line in lines: 
                # '\t'.join([str(t), message, d, d_tilde, response, str(False), first_crisis]
                t, message, d, d_tilde, response, episode_end, first_crisis = line.split('\t')
        
        
                if int(t) < prev_turn: 
                    convo_docs.append(convo_doc)
                    convo_doc = ''
        
                if first_crisis: 
                    convo_doc += ' ' + first_crisis.strip('\n')
            
                if response: 
                    convo_doc += ' ' + response.strip('\n')
                    
                if message: 
                    convo_doc += ' ' + message.strip('\n')
            
            
                prev_turn = int(t)
        
            convo_docs.append(convo_doc)
            convo_doc = ''
    
        self.vectorizer.fit(convo_docs)


        print('Keys in dict: ', len(self.vectorizer.vocabulary_.keys()))

#         with open('data/%s/%s/tfidf_vectorizer.pkl'% (datasetname, datasetname), 'wb') as f:
#             pickle.dump(vectorizer, f)








