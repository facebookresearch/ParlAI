import numpy as np



def load_dict(filename): 
    loaded_dict = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            split = line.strip().split('\t')
            token = split[0]
            cnt = int(split[1]) if len(split) > 1 else 0
            
            if token not in loaded_dict.keys():
                loaded_dict[token] = cnt
            else: 
                if cnt > loaded_dict[token]:
                    loaded_dict[token] = cnt
                else:
                    print('token: %s already added with freq %s' % (token, loaded_dict[token]))
    return loaded_dict



def get_stats(filename, df_dict, tot_doc, modelname):
    mean_response_idf = []
    max_response_idf = []
    response_lens = []
    unigrams = []
    responses = []
    bigrams = []
    
    with open(filename, 'r') as f:    
        for line in f.readlines():
            line = line.strip()
            
            if modelname not in line:
                continue # skip lines without model output.
            
            if len(line.split(']')) == 2 and line.split(']')[-1] == '': 
                print('skipping: ', line)
                continue
            else:
                
                line = line.replace('person1 ', '').replace('person2 ', '')
                
                split = line.split(' ')
                
                if split[1] == '__start__': 
                    start = 2
                else:
                    start = 1
                    
                    
                response_idfs = [np.log(tot_doc / float(df_dict[split[i]]))
                                            for i in range(start, len(split))]
                mean_response_idf.append(np.mean(response_idfs))
                max_response_idf.append(np.max(response_idfs))
                response_lens.append(len(split))
                
                unigrams += split[start:]
                bigrams += ['%s %s' % (split[i], split[i+1]) for i in range(start, len(split)-1)]
                responses.append(' '.join(split[start:]))
                
    if len(mean_response_idf) > 0:    
        return np.mean(mean_response_idf), np.mean(max_response_idf), \
                np.mean(response_lens), float(len(set(unigrams)))/float(len(unigrams)), \
                float(len(set(bigrams)))/float(len(bigrams)), \
                float(len(set(responses)))/float(len(responses))
    else: 
        return 0, 0, 0, 0, 0, 0

    
models = ['seq2seq', 'transformer', 'language_model']
tasks = ['cornell_movie', 'dailydialog', 'empathetic_dialogues', 'personachat']

    
datasets = ['cornell_movie', 'dailydialog', 'empathetic_dialogues', 'personachat'] 
  
modelinfo = [('LanguageModel', 'language_model'), 
            ('LanguageModelWeighted', 'language_model_idf'), 
            ('LanguageModelWeighted', 'language_model_swapping'),
            ('Seq2Seq', 'seq2seq'), 
            ('Seq2SeqWeighted', 'seq2seq_idf'), 
            ('Seq2SeqWeighted', 'seq2seq_swapping'), #('FACE', 'face'), 
            ('TorchAgent', 'transformer'),
            ('TransformerWeighted', 'transformer_idf'),
            ('TransformerWeighted', 'transformer_swapping')
            ]
            

stats_format = '%s, %s, %.3f, %.3f, %.3f, %.5f, %.5f, %.5f, %s'


if __name__ == '__main__': 
    result_lines = ['===================================\n',]
    for dataset in datasets:
        dict_filename = 'tmp/%s/dict_minfreq_2.doc_freq' % dataset
        tot_doc_filename = 'tmp/%s/dict_minfreq_2.tot_doc' % dataset
        df_dict = load_dict(dict_filename)
        
        with open(tot_doc_filename, 'r') as f:
            tot_doc = float(f.readline())
        
        for (modelname, modelprefix) in modelinfo: 
            print(dataset, modelname, modelprefix)
            try: 
                filename = 'tmp/%s/%s_minfreq_2_test.out' % (dataset, modelprefix)
            
                outputs = get_stats(filename, df_dict, tot_doc, modelname)
                stats = stats_format % \
                            tuple([dataset, modelname] + list(outputs) + [filename,])
                result_lines.append(stats)
            
                if modelname=='FACE': 
                    filename = 'tmp/%s/%s_minfreq_2_greedy_test.out' % (dataset, modelprefix)
                    outputs = get_stats(filename, df_dict, tot_doc, modelname)
                    stats = stats_format % \
                                tuple([dataset, modelname+'_greedy'] + list(outputs) + [filename,])
                    result_lines.append(stats)
            except FileNotFoundError:
                result_lines.append('%s, %s,,,,,,,' % (dataset, modelprefix))
                
                
    print('datasetname, modelname, avg_mean_idf, avg_max_idf, avg_length, distinct-unigram-ratio, distinct-bigram-ratio, unique-response-ratio')        
    print('\n'.join(result_lines))
            
    
    
    
        
    
