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
s2sattn=\'general\'\n\
\n\
### transformer architecture ###\n\
ffnsize=300\n\
transformerlayers=2 \n\
transformerdropout=0.2 \n\
transformerheads=2 \n\
\n\
### optimization ###\n\
batchsize=32\n\
transformerbatchsize=32\n\
learningrate=.001\n\
sgdlearningrate=10\n\
sgdminlearningrate=.1\n\
transformerlearningrate=.0008\n\
transformeroptimizer=adamax\n\
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




lm_train_boiler = """\n\
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
--tensorboard-log $tensorboardlog \\\n\
-m $modelname \\\n\
%s\n\
-mf 'tmp/'$taskname'/%s_minfreq_'$dictminfreq \\\n\
> 'tmp/'$taskname'/%s_minfreq_'$dictminfreq'_train.out'\\\n\
\n\n\n """




s2s_train_boiler = """\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/train_model.py \\\n\
-t $taskname \\\n\
-bs $batchsize \\\n\
--hiddensize $hiddensize \\\n\
--history-size $historysize \\\n\
-emb $embedding \\\n\
--numlayers $numlayers \\\n\
--bidirectional $bidirectional \\\n\
--attention $s2sattn \\\n\
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
--tensorboard-log $tensorboardlog \\\n\
-m $modelname \\\n\
%s\n\
-mf 'tmp/'$taskname'/%s_minfreq_'$dictminfreq \\\n\
> 'tmp/'$taskname'/%s_minfreq_'$dictminfreq'_train.out'\\\n\
\n\n\n """



transformer_train_boiler = """\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/train_model.py \\\n\
-t $taskname \\\n\
-bs $transformerbatchsize \\\n\
--ffn-size $ffnsize \\\n\
--n-heads $transformerheads \\\n\
--n-layers $transformerlayers \\\n\
--history-size $historysize \\\n\
-emb $embedding \\\n\
--embedding-size $embeddingsize \\\n\
--dropout $transformerdropout \\\n\
--learningrate $transformerlearningrate \\\n\
--optimizer $transformeroptimizer \\\n\
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
--tensorboard-log $tensorboardlog \\\n\
-m $modelname \\\n\
%s\n\
-mf 'tmp/'$taskname'/%s_minfreq_'$dictminfreq \\\n\
> 'tmp/'$taskname'/%s_minfreq_'$dictminfreq'_train.out'\\\n\
\n\n\n """




eval_torchgen_model="""\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/eval_model.py \\\n\
--datatype %s \\\n\
--history-size $historysize \\\n\
--beam-size $beamsize \\\n\
-m $modelname \\\n\
-t $taskname \\\n\
--display-examples 1 \\\n\
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \\\n\
-mf 'tmp/'$taskname'/%s_minfreq_'$dictminfreq \\\n\
> 'tmp/'$taskname'/%s_minfreq_'$dictminfreq'_%s.out'\\\n\
\n\n\n """


eval_language_model="""\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/eval_model.py \\\n\
--datatype %s \\\n\
-m $modelname \\\n\
-t $taskname \\\n\
--display-examples 1 \\\n\
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \\\n\
-mf 'tmp/'$taskname'/%s_minfreq_'$dictminfreq \\\n\
> 'tmp/'$taskname'/%s_minfreq_'$dictminfreq'_%s.out'\\\n\
\n\n\n """



model_boilers = [
                (transformer_train_boiler, eval_torchgen_model, 'transformer'), 
#                 (s2s_train_boiler, eval_torchgen_model, 'seq2seq'), 
#                 (lm_train_boiler, eval_language_model, 'language_model')
                ]
tasks = ['cornell_movie', 'dailydialog', 'empathetic_dialogues', 'personachat']
# tasks = ['cornell_movie', ]
TRAIN = True
EVAL = True


if __name__ == '__main__': 
    
    for t, task in enumerate(tasks): 
        
        GPU_NUM = t + 2
#         if GPU_NUM == 5: # HACK, as opensubtitles is currently running on gpu5
#             GPU_NUM = 1
        
        cmd_filename = 'cmd_transformer_only_%s.sh' % task
        with open(cmd_filename, 'w') as f: 
            
            if task == 'opensubtitles':
                print('NEED TO IMPLEMENT')
                import sys; sys.exit()
            else: 
                f.write("taskname='%s'" % task)
                f.write(parameter_definition.format(GPUNUM=str(GPU_NUM)))
                f.write('\n\n\n\n\n')
            
        
        
            for train_boiler, eval_boiler, basemodel in model_boilers:
                
                if basemodel == 'transformer':
                    f.write('\n\n\n\n\n')
                    f.write("modelname='%s/generator'" % basemodel)
                    f.write('\n\n\n\n\n')
                elif basemodel == 'language_model': 
                    f.write('\n\n\n\n\n')
                    f.write("modelname='%s_emb'" % basemodel)
                    f.write('\n\n\n\n\n')
                else:
                    f.write('\n\n\n\n\n')
                    f.write("modelname='%s'" % basemodel)
                    f.write('\n\n\n\n\n')
                
                model_prefix = "'%s'" % basemodel
                if TRAIN:
                    f.write(train_boiler % ('\\', model_prefix, model_prefix))
                if EVAL: 
#                     f.write(eval_boiler % ('valid', model_prefix, model_prefix, 'valid'))
                    f.write(eval_boiler % ('test', model_prefix, model_prefix, 'test'))
            
            
                f.write('\n\n\n\n\n')
                f.write("modelname='%s_weighted'" % basemodel)
                f.write('\n\n\n\n\n')
            
                model_prefix = "'%s_idf'" % basemodel
                if TRAIN:
                    f.write(train_boiler % ('--swap-criterion-train-eval False \\', model_prefix, model_prefix))
                if EVAL: 
#                     f.write(eval_boiler % ('valid', model_prefix, model_prefix, 'valid'))
                    f.write(eval_boiler % ('test', model_prefix, model_prefix, 'test'))

                model_prefix = "'%s_swapping'" % basemodel
                if TRAIN:
                    f.write(train_boiler % ('--swap-criterion-train-eval True \\', model_prefix, model_prefix))
                if EVAL:
#                     f.write(eval_boiler % ('valid', model_prefix, model_prefix, 'valid'))
                    f.write(eval_boiler % ('test', model_prefix, model_prefix, 'test'))
            
            
            
            ##### FOR FACE VERSIONS #####
            for train_boiler, eval_boiler, basemodel in model_boilers:
                
                if basemodel == 'transformer':
                    f.write('\n\n\n\n\n')
                    f.write("modelname='newface%s'" % basemodel)
                    f.write('\n\n\n\n\n')
                elif basemodel == 'language_model': 
#                     f.write('\n\n\n\n\n')
#                     f.write("modelname='%s_emb'" % basemodel)
#                     f.write('\n\n\n\n\n')
                    continue
                else:
                    f.write('\n\n\n\n\n')
                    f.write("modelname='newface'")
                    f.write('\n\n\n\n\n')
                
                model_prefix = "'newface%s'" % basemodel
                if TRAIN:
                    f.write(train_boiler % ('\\', model_prefix, model_prefix))
                if EVAL: 
#                     f.write(eval_boiler % ('valid', model_prefix, model_prefix, 'valid'))
                    f.write(eval_boiler % ('test', model_prefix, model_prefix, 'test'))
            
        # change permissions to allow execute.
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




