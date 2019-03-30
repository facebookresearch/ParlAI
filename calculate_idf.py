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
    
    with open(filename, 'r') as f:    
        for line in f.readlines():
            line = line.strip()
            
            if modelname not in line:
                continue # skip lines without model output.
            
            if len(line.split(']')) == 2 and line.split(']')[-1] == '': 
                print('skipping: ', line)
                continue
            else:
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
                
                
    return np.mean(mean_response_idf), np.mean(max_response_idf), \
            np.mean(response_lens), float(len(set(unigrams)))/float(len(unigrams))


    
    
datasets = ['dailydialog', 'empathetic_dialogues', 'personachat'] 
  
modelinfo = [('Seq2Seq', 'seq2seq'), ('Seq2SeqRetriever', 'seq2seq_retriever_idf'), 
            ('Seq2SeqRetriever', 'seq2seq_retriever_idf_swapping'), ('FACE', 'face')]
            


if __name__ == '__main__': 
    result_lines = []
    for dataset in datasets:
        dict_filename = 'tmp/%s/dict_minfreq_2.doc_freq' % dataset
        tot_doc_filename = 'tmp/%s/dict_minfreq_2.tot_doc' % dataset
        df_dict = load_dict(dict_filename)
        
        with open(tot_doc_filename, 'r') as f:
            tot_doc = float(f.readline())
        
        for (modelname, modelprefix) in modelinfo: 
            filename = 'tmp/%s/%s_minfreq_2_test.out' % (dataset, modelprefix)
            
            outputs = get_stats(filename, df_dict, tot_doc, modelname)
            stats = '%s, %s, %.3f, %.3f, %.3f, %.3f, %s' % \
                        tuple([dataset, modelname] + list(outputs) + [filename,])
            result_lines.append(stats)
    print('datasetname, modelname, avg_mean_idf, avg_max_idf, avg_length, distinct-unigram-ratio')        
    print('\n'.join(result_lines))
            
    
    
    
        
    
