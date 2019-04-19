from string import Template
import os, sys, stat

parameter_definition = '''\n\n\n\
### embedding ###\n\
embeddingsize=1000\n\
\n\
### architecture ###\n\
hiddensize=1000\n\
historysize=2\n\
beamsize=5\n\
numlayers=4\n\
lmlayers=8\n\
bidirectional=True\n\
s2sattn=\'general\'\n\
\n\
### transformer architecture ###\n\
ffnsize=300\n\
transformerlayers=4 \n\
transformerdropout=0.2 \n\
transformerheads=8 \n\
\n\
### optimization ###\n\
batchsize=256\n\
transformerbatchsize=32\n\
learningrate=.001\n\
sgdlearningrate=10\n\
sgdminlearningrate=.1\n\
transformerlearningrate=.0008\n\
transformeroptimizer=adamax\n\
optimizer=adam\n\
gradientclip=1.0\n\
eps=1 \n\
\n\
### Dictionary ###\n\
dictmaxtokens=25000\n\
dictlower=True\n\
dicttokenizer=\'nltk\'\n\
dictfile=\'tmp/\'$taskname\'/dict_maxtokens_$dictmaxtokens\'\n\
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
--numlayers $lmlayers \\\n\
--embeddingsize $embeddingsize \\\n\
--learningrate $sgdlearningrate \\\n\
-eps $eps \\\n\
--dict-maxtokens $dictmaxtokens \\\n\
--dict-lower $dictlower \\\n\
--dict-tokenizer $dicttokenizer \\\n\
--dict-file 'tmp/'$taskname'/dict_maxtokens_'$dictmaxtokens \\\n\
--tensorboard-log $tensorboardlog \\\n\
-m $modelname \\\n\
%s\n\
-mf 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens \\\n\
> 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens'_train.out'\\\n\
\n\n\n """




s2s_train_boiler = """\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/train_model.py \\\n\
-t $taskname \\\n\
-bs $batchsize \\\n\
--hiddensize $hiddensize \\\n\
--history-size $historysize \\\n\
--numlayers $numlayers \\\n\
--bidirectional $bidirectional \\\n\
--attention $s2sattn \\\n\
--embeddingsize $embeddingsize \\\n\
--learningrate $learningrate \\\n\
--optimizer $optimizer \\\n\
--gradient-clip $gradientclip \\\n\
-eps $eps \\\n\
--dict-maxtokens $dictmaxtokens \\\n\
--dict-lower $dictlower \\\n\
--dict-tokenizer $dicttokenizer \\\n\
--dict-file 'tmp/'$taskname'/dict_maxtokens_'$dictmaxtokens \\\n\
--tensorboard-log $tensorboardlog \\\n\
-m $modelname \\\n\
%s\n\
-mf 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens \\\n\
> 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens'_train.out'\\\n\
\n\n\n """



transformer_train_boiler = """\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/train_model.py \\\n\
-t $taskname \\\n\
-bs $transformerbatchsize \\\n\
--ffn-size $ffnsize \\\n\
--n-heads $transformerheads \\\n\
--n-layers $transformerlayers \\\n\
--history-size $historysize \\\n\
--embedding-size $embeddingsize \\\n\
--dropout $transformerdropout \\\n\
--learningrate $transformerlearningrate \\\n\
--optimizer $transformeroptimizer \\\n\
--gradient-clip $gradientclip \\\n\
-eps $eps \\\n\
--dict-maxtokens $dictmaxtokens \\\n\
--dict-lower $dictlower \\\n\
--dict-tokenizer $dicttokenizer \\\n\
--dict-file 'tmp/'$taskname'/dict_maxtokens_'$dictmaxtokens \\\n\
--tensorboard-log $tensorboardlog \\\n\
-m $modelname \\\n\
%s\n\
-mf 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens \\\n\
> 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens'_train.out'\\\n\
\n\n\n """



eval_torchgen_model="""\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/eval_model.py \\\n\
--datatype %s \\\n\
--history-size $historysize \\\n\
--beam-size $beamsize \\\n\
-m $modelname \\\n\
-t $taskname \\\n\
--display-examples 1 \\\n\
-df 'tmp/'$taskname'/dict_maxtokens_'$dictmaxtokens \\\n\
-mf 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens \\\n\
> 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens'_%s.out'\\\n\
\n\n\n """


eval_language_model="""\n\
CUDA_VISIBLE_DEVICES=$gpunum python examples/eval_model.py \\\n\
--datatype %s \\\n\
-m $modelname \\\n\
-t $taskname \\\n\
--display-examples 1 \\\n\
-df 'tmp/'$taskname'/dict_maxtokens_'$dictmaxtokens \\\n\
-mf 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens \\\n\
> 'tmp/'$taskname'/%s_maxtokens_'$dictmaxtokens'_%s.out'\\\n\
\n\n\n """



model_boilers = [
                (s2s_train_boiler, eval_torchgen_model, 'seq2seq'), 
                (transformer_train_boiler, eval_torchgen_model, 'transformer'), 
                (lm_train_boiler, eval_language_model, 'language_model')
                ]
                
tasks = ['opensubtitles', ]
TRAIN = True
EVAL = True

GPU_START = 0

if __name__ == '__main__': 
    
    for t, task in enumerate(tasks): 
        cmd_filename = 'cmd_%s.sh' % task
        cmd_idf_filename = 'cmd_%s_idf.sh' % task
        cmd_swap_filename = 'cmd_%s_swap.sh' % task
        cmd_face_filename = 'cmd_%s_face.sh' % task
        
        
        with open(cmd_filename, 'w') as f: 
            
            GPU_NUM = GPU_START
            
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
            
            
        with open(cmd_idf_filename, 'w') as f: 
            
            GPU_NUM = GPU_START + 1
            
            f.write("taskname='%s'" % task)
            f.write(parameter_definition.format(GPUNUM=str(GPU_NUM)))
            f.write('\n\n\n\n\n')
            
            for train_boiler, eval_boiler, basemodel in model_boilers:
            
                f.write('\n\n\n\n\n')
                f.write("modelname='%s_weighted'" % basemodel)
                f.write('\n\n\n\n\n')
            
                model_prefix = "'%s_idf'" % basemodel
                if TRAIN:
                    f.write(train_boiler % ('--swap-criterion-train-eval False \\', model_prefix, model_prefix))
                if EVAL: 
#                     f.write(eval_boiler % ('valid', model_prefix, model_prefix, 'valid'))
                    f.write(eval_boiler % ('test', model_prefix, model_prefix, 'test'))

        
        
        with open(cmd_swap_filename, 'w') as f: 
            
            GPU_NUM = GPU_START + 2
            
            f.write("taskname='%s'" % task)
            f.write(parameter_definition.format(GPUNUM=str(GPU_NUM)))
            f.write('\n\n\n\n\n')
        
            for train_boiler, eval_boiler, basemodel in model_boilers:
            
                model_prefix = "'%s_swapping'" % basemodel
                if TRAIN:
                    f.write(train_boiler % ('--swap-criterion-train-eval True \\', model_prefix, model_prefix))
                if EVAL:
#                     f.write(eval_boiler % ('valid', model_prefix, model_prefix, 'valid'))
                    f.write(eval_boiler % ('test', model_prefix, model_prefix, 'test'))
            
            
            
            
        with open(cmd_face_filename, 'w') as f: 

            GPU_NUM = GPU_START + 3
            
            f.write("taskname='%s'" % task)
            f.write(parameter_definition.format(GPUNUM=str(GPU_NUM)))
            f.write('\n\n\n\n\n')
        
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
            
        for filename in [cmd_filename, 
                    cmd_idf_filename, 
                    cmd_swap_filename, 
                    cmd_face_filename]:
                    
            # change permissions to allow execute.
            os.chmod(filename, stat.S_IRWXU)



