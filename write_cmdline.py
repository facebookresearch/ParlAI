from string import Template
import os, sys, stat

parameter_definition = '''\n\n\n\
### embedding ###\n\
embedding=\'glove\'\n\
embeddingsize=300\n\
\n\
### architecture ###\n\
hiddensize=1024\n\
historysize=2\n\
beamsize=5\n\
numlayers=1\n\
lmlayers=2\n\
bidirectional=True\n\
\n\
### optimization ###\n\
batchsize=32\n\
learningrate=.001\n\
sgdlearningrate=10\n\
sgdminlearningrate=.1\n\
optimizer=adam\n\
gradientclip=1.0\n\
lrschedulerdecay=.5\n\
lrschedulerpatience=3\n\
validationpatience=10\n\
validationeverynepochs=1\n\
validationmetric=\'loss\'\n\
validationmetricmode=\'min\'\n\
\n\
### Dictionary ###\n\
dictminfreq=2\n\
dictlower=True\n\
dicttokenizer=\'spacy\'\n\
dictfile=\'tmp/\'$taskname\'/dict_minfreq_$dictminfreq\'\n\
\n\
### logging ###\n\
tensorboardlog=False\n\
\n\
gpunum={GPUNUM}'''




lm_boiler = """\n\
####################################\n\
###########  train model ###########\n\
####################################\n\
\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/train_model.py \\\n\
-t $taskname \\\n\
-bs $batchsize \\\n\
--hiddensize $hiddensize \\\n\
-emb $embedding \\\n\
--numlayers $lmlayers \\\n\
--embeddingsize $embeddingsize \\\n\
--learningrate $sgdlearningrate \\\n\
--lr-factor $lrschedulerdecay \\\n\
--lr-patience $lrschedulerpatience \\\n\
--lr-minimum $sgdminlearningrate \\\n\
--validation-patience $validationpatience \\\n\
--validation-every-n-epochs $validationeverynepochs \\\n\
--validation-metric $validationmetric \\\n\
--validation-metric-mode $validationmetricmode \\\n\
--dict-minfreq $dictminfreq \\\n\
--dict-lower $dictlower \\\n\
--dict-tokenizer $dicttokenizer \\\n\
--dict-file 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \\\n\
-m $modelname \\\n\
%s\n\
-mf 'tmp/'$taskname'/%s_minfreq_'$dictminfreq \\\n\
> 'tmp/'$taskname'/%s_minfreq_'$dictminfreq'_train.out'\\\n\
\n\n\n """




other_boiler = """\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/train_model.py \\\n\
-t $taskname \\\n\
-bs $batchsize \\\n\
--hiddensize $hiddensize \\\n\
--history-size $historysize \\\n\
-emb $embedding \\\n\
--numlayers $numlayers \\\n\
--bidirectional $bidirectional \\\n\
--embeddingsize $embeddingsize \\\n\
--learningrate $learningrate \\\n\
--optimizer $optimizer \\\n\
--gradient-clip $gradientclip \\\n\
--lr-scheduler-decay $lrschedulerdecay \\\n\
--lr-scheduler-patience $lrschedulerpatience \\\n\
--validation-patience $validationpatience \\\n\
--validation-every-n-epochs $validationeverynepochs \\\n\
--validation-metric $validationmetric \\\n\
--validation-metric-mode $validationmetricmode \\\n\
--dict-minfreq $dictminfreq \\\n\
--dict-lower $dictlower \\\n\
--dict-tokenizer $dicttokenizer \\\n\
--dict-file 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \\\n\
-m $modelname \\\n\
%s\n\
-mf 'tmp/'$taskname'/%s_minfreq_'$dictminfreq \\\n\
> 'tmp/'$taskname'/%s_minfreq_'$dictminfreq'_train.out'\\\n\
\n\n\n """







eval_model="""\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/eval_model.py \\\n\
--datatype %s \\\n\
--history-size $historysize \\\n\
--beam-size $beamsize \\\n\
-m $modelname \\\n\
-t $taskname \\\n\
--display-examples 1 \\\n\
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \\\n\
-mf 'tmp/'$taskname'/%s_minfreq_'$dictminfreq \\\n\
> 'tmp/'$taskname'/%s_minfreq_'$dictminfreq'_valid.out'\\\n\
\n\n\n """



models = ['seq2seq', 'transformer', 'language_model']
tasks = ['cornell_movie', 'dailydialog', 'empathetic_dialogues', 'personachat']


if __name__ == '__main__': 
    
    for t, task in enumerate(tasks): 
        
        GPU_NUM = t + 3
        
        cmd_filename = 'cmd_%s.sh' % task
        with open(cmd_filename, 'w') as f: 
        
            f.write("taskname='%s'" % task)
            f.write(parameter_definition.format(GPUNUM=str(GPU_NUM)))
            f.write('\n\n\n\n\n')
        
        
            for basemodel in models:
        
                if basemodel == 'language_model':
                
                    f.write('\n\n\n\n\n')
                    f.write("modelname='%s'" % basemodel)
                    f.write('\n\n\n\n\n')
                
                    model_prefix = "'%s'" % basemodel
                    f.write(lm_boiler % ('\\', model_prefix, model_prefix))
                    f.write(eval_model % ('valid', model_prefix, model_prefix))
                    f.write(eval_model % ('test', model_prefix, model_prefix))
                
                
                    f.write('\n\n\n\n\n')
                    f.write("modelname='%s_weighted'" % basemodel)
                    f.write('\n\n\n\n\n')
                
                    model_prefix = "'%s_idf'" % basemodel
                    f.write(lm_boiler % ('--swap-criterion-train-eval False \\', model_prefix, model_prefix))
                    f.write(eval_model % ('valid', model_prefix, model_prefix))
                    f.write(eval_model % ('test', model_prefix, model_prefix))

                    model_prefix = "'%s_swapping'" % basemodel
                    f.write(lm_boiler % ('--swap-criterion-train-eval True \\', model_prefix, model_prefix))
                    f.write(eval_model % ('valid', model_prefix, model_prefix))
                    f.write(eval_model % ('test', model_prefix, model_prefix))
            
                else:
            
                    if basemodel == 'transformer':
                        f.write('\n\n\n\n\n')
                        f.write("modelname='%s/generator'" % basemodel)
                        f.write('\n\n\n\n\n')
                    else:
                        f.write('\n\n\n\n\n')
                        f.write("modelname='%s'" % basemodel)
                        f.write('\n\n\n\n\n')
                
                
                    model_prefix = "'%s'" % basemodel
                    f.write(other_boiler % ('\\', model_prefix, model_prefix))
                    f.write(eval_model % ('valid', model_prefix, model_prefix))
                    f.write(eval_model % ('test', model_prefix, model_prefix))

                
                    f.write('\n\n\n\n\n')
                    f.write("modelname='%s_weighted'" % basemodel)
                    f.write('\n\n\n\n\n')
                
                    model_prefix = "'%s_idf'" % basemodel
                    f.write(other_boiler % ('--swap-criterion-train-eval False \\', model_prefix, model_prefix))
                    f.write(eval_model % ('valid', model_prefix, model_prefix))
                    f.write(eval_model % ('test', model_prefix, model_prefix))

                    model_prefix = "'%s_swapping'" % basemodel
                    f.write(other_boiler % ('--swap-criterion-train-eval True \\', model_prefix, model_prefix))
                    f.write(eval_model % ('valid', model_prefix, model_prefix))
                    f.write(eval_model % ('test', model_prefix, model_prefix))
    
        
        # Assuming /tmp/foo.txt exists, Set a file execute by the group.
        os.chmod(cmd_filename, stat.S_IRWXU)





# #############################
# ##### FACE for training #####
# #############################
# modelname='face'
# fileprefix=$modelname
# 
# 
# ####################################
# ###########  train model ###########
# ####################################
# 
# CUDA_VISIBLE_DEVICES=$gpunum python examples/train_model.py \
# -t $taskname \
# -bs $batchsize \
# --hiddensize $hiddensize \
# --history-size $historysize \
# -emb $embedding \
# --numlayers $numlayers \
# --bidirectional $bidirectional \
# --embeddingsize $embeddingsize \
# --learningrate $learningrate \
# --optimizer $optimizer \
# --gradient-clip $gradientclip \
# --lr-scheduler-decay $lrschedulerdecay \
# --lr-scheduler-patience $lrschedulerpatience \
# --validation-patience $validationpatience \
# --validation-every-n-epochs $validationeverynepochs \
# --validation-metric $validationmetric \
# --validation-metric-mode $validationmetricmode \
# --dict-minfreq $dictminfreq \
# --dict-lower $dictlower \
# --dict-tokenizer $dicttokenizer \
# --dict-file 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
# --tensorboard-log $tensorboardlog \
# -m $modelname \
# -mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
# > 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_train.out'
# 
# 
# 
# ####################################
# ############ eval model ############
# ####################################
# 
# 
# ### EVAL model: modified (retriever) s2s ###
# CUDA_VISIBLE_DEVICES=$gpunum python examples/eval_model.py \
# --datatype valid \
# --history-size $historysize \
# --beam-size $beamsize \
# -m $modelname \
# -t $taskname \
# --display-examples 1 \
# -df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
# -mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
# > 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_valid.out'
# 
# 
# ### TEST model: modified (retriever) s2s ###
# CUDA_VISIBLE_DEVICES=$gpunum python examples/eval_model.py \
# --datatype test \
# --history-size $historysize \
# --beam-size $beamsize \
# -m $modelname \
# -t $taskname \
# --display-examples 1 \
# -df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
# -mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
# > 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_test.out'



# #############################
# ##### Greedy FACE #####
# #############################
# modelname='face'
# fileprefix=$modelname
# 
# ### TEST model: modified (retriever) s2s ###
# CUDA_VISIBLE_DEVICES=$gpunum python examples/eval_model.py \
# --history-size $historysize \
# --beam-size 1 \
# --datatype test \
# -m $modelname \
# -t $taskname \
# --display-examples 1 \
# -df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
# -mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
# > 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_greedy_test.out'




