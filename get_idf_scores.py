import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

import spacy
nlp = spacy.load('en_core_web_sm')

# from spacy.tokenizer import Tokenizer
# tokenizer = Tokenizer(nlp.vocab)
from spacy.lang.en import English
tokenizer = English().Defaults.create_tokenizer(nlp)


def my_spacy_tokenizer(doc):
    tokens = tokenizer(doc)
    return([token.text for token in tokens])
    
    

vectorizer = TfidfVectorizer(lowercase=True, tokenizer=my_spacy_tokenizer, 
                                stop_words=None, use_idf=True)


datasetname = sys.argv[1]
train_filename = 'data/%s/%s/train.tsv' % (datasetname, datasetname)
valid_filename = 'data/%s/%s/valid.tsv' % (datasetname, datasetname)


crisis_docs = []

for filename in [train_filename, valid_filename]:
    
    lines = open(filename, 'r').readlines()
    crisis_doc = ''
    prev_turn = -1

    for line in lines: 
        # '\t'.join([str(t), message, d, d_tilde, response, str(False), first_crisis]
        t, message, d, d_tilde, response, episode_end, first_crisis = line.split('\t')
        
        
        if int(t) < prev_turn: 
            crisis_docs.append(crisis_doc)
            crisis_doc = ''
        
        if first_crisis: 
            crisis_doc += ' ' + first_crisis.strip('\n')
            
        if response: 
            crisis_doc += ' ' + response.strip('\n')
            
            
        prev_turn = int(t)
        
    crisis_docs.append(crisis_doc)
    crisis_doc = ''
    
vectorizer = vectorizer.fit(crisis_docs)


print('Keys in dict: ', len(vectorizer.vocabulary_.keys()))

with open('data/%s/%s/tfidf_vectorizer.pkl'% (datasetname, datasetname), 'wb') as f:
    pickle.dump(vectorizer, f)








