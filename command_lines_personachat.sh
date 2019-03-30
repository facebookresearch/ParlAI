
### embedding ###
embedding='glove'
embeddingsize=300

### architecture ###
hiddensize=1024
historysize=2
beamsize=5
numlayers=1
bidirectional=True

### optimization ###
batchsize=32
learningrate=.001
optimizer=adam
gradientclip=1.0
lrschedulerdecay=.5
lrschedulerpatience=3
validationpatience=10
validationeverynepochs=1
validationmetric='loss'
validationmetricmode='min'

### Dictionary ###
dictminfreq=2
dictlower=True
dicttokenizer='spacy'
dictfile='tmp/'$taskname'/dict_minfreq_$dictminfreq'

### logging ###
tensorboardlog=True
# gpunum=1


### task ###
taskname='personachat'




####################################
############ build dict ############
####################################
python examples/build_dict.py \
-t $taskname \
--dict-minfreq $dictminfreq \
--dict-lower $dictlower \
--dict-tokenizer $dicttokenizer \
--dict-file 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/dict_minfreq_'$dictminfreq'.out'





#####################################
############# idf model #############
#####################################
modelname='seq2seq_retriever'
fileprefix=$modelname'_idf'


####################################
###########  train model ###########
####################################

python examples/train_model.py \
-t $taskname \
-bs $batchsize \
--hiddensize $hiddensize \
--history-size $historysize \
-emb $embedding \
--numlayers $numlayers \
--bidirectional $bidirectional \
--embeddingsize $embeddingsize \
--learningrate $learningrate \
--optimizer $optimizer \
--gradient-clip $gradientclip \
--lr-scheduler-decay $lrschedulerdecay \
--lr-scheduler-patience $lrschedulerpatience \
--validation-patience $validationpatience \
--validation-every-n-epochs $validationeverynepochs \
--validation-metric $validationmetric \
--validation-metric-mode $validationmetricmode \
--dict-minfreq $dictminfreq \
--dict-lower $dictlower \
--dict-tokenizer $dicttokenizer \
--dict-file 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
--tensorboard-log $tensorboardlog \
-m $modelname \
--weight-criterion-idf True \
--swap-criterion-train-eval False \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_train.out'



####################################
############ eval model ############
####################################


### EVAL model: modified (retriever) s2s ###
python examples/eval_model.py \
--datatype valid \
--history-size $historysize \
--beam-size $beamsize \
-m $modelname \
-t $taskname \
--display-examples 1 \
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_valid.out'


### TEST model: modified (retriever) s2s ###
python examples/eval_model.py \
--datatype test \
--history-size $historysize \
--beam-size $beamsize \
-m $modelname \
-t $taskname \
--display-examples 1 \
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_test.out'





#################################
##### turn loss swapping on #####
#################################
modelname='seq2seq_retriever'
fileprefix=$modelname'_idf_swapping'


####################################
###########  train model ###########
####################################

python examples/train_model.py \
-t $taskname \
-bs $batchsize \
--hiddensize $hiddensize \
--history-size $historysize \
-emb $embedding \
--numlayers $numlayers \
--bidirectional $bidirectional \
--embeddingsize $embeddingsize \
--learningrate $learningrate \
--optimizer $optimizer \
--gradient-clip $gradientclip \
--lr-scheduler-decay $lrschedulerdecay \
--lr-scheduler-patience $lrschedulerpatience \
--validation-patience $validationpatience \
--validation-every-n-epochs $validationeverynepochs \
--validation-metric $validationmetric \
--validation-metric-mode $validationmetricmode \
--dict-minfreq $dictminfreq \
--dict-lower $dictlower \
--dict-tokenizer $dicttokenizer \
--dict-file 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
--tensorboard-log $tensorboardlog \
-m $modelname \
--weight-criterion-idf True \
--swap-criterion-train-eval True \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_train.out'



####################################
############ eval model ############
####################################


### EVAL model: modified (retriever) s2s ###
python examples/eval_model.py \
--datatype valid \
--history-size $historysize \
--beam-size $beamsize \
-m $modelname \
-t $taskname \
--display-examples 1 \
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_valid.out'


### TEST model: modified (retriever) s2s ###
python examples/eval_model.py \
--datatype test \
--history-size $historysize \
--beam-size $beamsize \
-m $modelname \
-t $taskname \
--display-examples 1 \
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_test.out'






###########################
##### vanilla seq2seq #####
###########################
modelname='seq2seq'
fileprefix=$modelname


####################################
###########  train model ###########
####################################

python examples/train_model.py \
-t $taskname \
-bs $batchsize \
--hiddensize $hiddensize \
--history-size $historysize \
-emb $embedding \
--numlayers $numlayers \
--bidirectional $bidirectional \
--embeddingsize $embeddingsize \
--learningrate $learningrate \
--optimizer $optimizer \
--gradient-clip $gradientclip \
--lr-scheduler-decay $lrschedulerdecay \
--lr-scheduler-patience $lrschedulerpatience \
--validation-patience $validationpatience \
--validation-every-n-epochs $validationeverynepochs \
--validation-metric $validationmetric \
--validation-metric-mode $validationmetricmode \
--dict-minfreq $dictminfreq \
--dict-lower $dictlower \
--dict-tokenizer $dicttokenizer \
--dict-file 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
--tensorboard-log $tensorboardlog \
-m $modelname \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_train.out'



####################################
############ eval model ############
####################################


### EVAL model: modified (retriever) s2s ###
python examples/eval_model.py \
--datatype valid \
--history-size $historysize \
--beam-size $beamsize \
-m $modelname \
-t $taskname \
--display-examples 1 \
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_valid.out'


### TEST model: modified (retriever) s2s ###
python examples/eval_model.py \
--datatype test \
--history-size $historysize \
--beam-size $beamsize \
-m $modelname \
-t $taskname \
--display-examples 1 \
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_test.out'











#############################
##### FACE for training #####
#############################
modelname='face'
fileprefix=$modelname


####################################
###########  train model ###########
####################################

python examples/train_model.py \
-t $taskname \
-bs $batchsize \
--hiddensize $hiddensize \
--history-size $historysize \
-emb $embedding \
--numlayers $numlayers \
--bidirectional $bidirectional \
--embeddingsize $embeddingsize \
--learningrate $learningrate \
--optimizer $optimizer \
--gradient-clip $gradientclip \
--lr-scheduler-decay $lrschedulerdecay \
--lr-scheduler-patience $lrschedulerpatience \
--validation-patience $validationpatience \
--validation-every-n-epochs $validationeverynepochs \
--validation-metric $validationmetric \
--validation-metric-mode $validationmetricmode \
--dict-minfreq $dictminfreq \
--dict-lower $dictlower \
--dict-tokenizer $dicttokenizer \
--dict-file 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
--tensorboard-log $tensorboardlog \
-m $modelname \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_train.out'



####################################
############ eval model ############
####################################


### EVAL model: modified (retriever) s2s ###
python examples/eval_model.py \
--datatype valid \
--history-size $historysize \
--beam-size $beamsize \
-m $modelname \
-t $taskname \
--display-examples 1 \
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_valid.out'


### TEST model: modified (retriever) s2s ###
python examples/eval_model.py \
--datatype test \
--history-size $historysize \
--beam-size $beamsize \
-m $modelname \
-t $taskname \
--display-examples 1 \
-df 'tmp/'$taskname'/dict_minfreq_'$dictminfreq \
-mf 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq \
> 'tmp/'$taskname'/'$fileprefix'_minfreq_'$dictminfreq'_test.out'










# 
# python examples/build_dict.py \
# -t personachat \
# --dict-minfreq 10 \
# --dict-lower True \
# --dict-tokenizer spacy \
# --dict-file tmp/personachat/dict_minfreq_10 \
# > tmp/personachat/dict_minfreq_10.out
# 
# 
# ###########################################
# # train seq2seq_retriever -- idf weighted #
# ###########################################
# 
# time python examples/train_model.py \
# -t personachat \
# --weight-criterion-idf True \
# --swap-criterion-train-eval False \
# -bs 128 \
# --hiddensize 1024 \
# -emb glove \
# --numlayers 1 \
# --bidirectional True \
# --embeddingsize 300 \
# --learningrate .001 \
# --optimizer adam \
# --gradient-clip 1.0 \
# --lr-scheduler-decay .5 \
# --lr-scheduler-patience 5 \
# --validation-patience 25 \
# --validation-every-n-epochs 1 \
# --validation-metric loss \
# --dict-minfreq 10 \
# --dict-lower True \
# --dict-tokenizer spacy \
# --dict-file tmp/personachat/dict_minfreq_10 \
# --tensorboard-log True \
# -m seq2seq_retriever \
# -mf tmp/personachat/seq2seq_retriever_model_idf_minfreq_10 \
# > tmp/personachat/seq2seq_retriever_model_idf_minfreq_10_train.out
# 
# 
# 
# ### EVAL model: modified (retriever) s2s ###
# time python examples/eval_model.py \
# --datatype valid \
# -m seq2seq_retriever \
# -t personachat \
# --display-examples 1 \
# -df tmp/personachat/dict_minfreq_10 \
# -mf tmp/personachat/seq2seq_retriever_model_idf_minfreq_10 \
# > tmp/personachat/seq2seq_retriever_model_idf_minfreq_10_valid.out
# 
# 
# 
# ### TEST model: modified (retriever) s2s ###
# time python examples/eval_model.py \
# --datatype test \
# -m seq2seq_retriever \
# -t personachat \
# --display-examples 1 \
# -df tmp/personachat/dict_minfreq_10 \
# -mf tmp/personachat/seq2seq_retriever_model_idf_minfreq_10 \
# > tmp/personachat/seq2seq_retriever_model_idf_minfreq_10_test.out
# 
# 
# 
# ############################################
# # train seq2seq_retriever -- idf, swapping #
# ############################################
# 
# time python examples/train_model.py \
# -t personachat \
# --weight-criterion-idf True \
# --swap-criterion-train-eval True \
# -bs 128 \
# --hiddensize 1024 \
# -emb glove \
# --numlayers 1 \
# --bidirectional True \
# --embeddingsize 300 \
# --learningrate .001 \
# --optimizer adam \
# --gradient-clip 1.0 \
# --lr-scheduler-decay .5 \
# --lr-scheduler-patience 5 \
# --validation-patience 25 \
# --validation-every-n-epochs 1 \
# --validation-metric loss \
# --dict-minfreq 10 \
# --dict-lower True \
# --dict-tokenizer spacy \
# --dict-file tmp/personachat/dict_minfreq_10 \
# --tensorboard-log True \
# -m seq2seq_retriever \
# -mf tmp/personachat/seq2seq_retriever_model_idf_swapping_minfreq_10 \
# > tmp/personachat/seq2seq_retriever_model_idf_swapping_minfreq_10_train.out
# 
# 
# 
# ### EVAL model ###
# time python examples/eval_model.py \
# --datatype valid \
# -m seq2seq_retriever \
# -t personachat \
# --display-examples 1 \
# -df tmp/personachat/dict_minfreq_10 \
# -mf tmp/personachat/seq2seq_retriever_model_idf_swapping_minfreq_10 \
# > tmp/personachat/seq2seq_retriever_model_idf_swapping_minfreq_10_valid.out
# 
# 
# ### TEST model ###
# time python examples/eval_model.py \
# --datatype test \
# -m seq2seq_retriever \
# -t personachat \
# --display-examples 1 \
# -df tmp/personachat/dict_minfreq_10 \
# -mf tmp/personachat/seq2seq_retriever_model_idf_swapping_minfreq_10 \
# > tmp/personachat/seq2seq_retriever_model_idf_swapping_minfreq_10_test.out
# 
# 
# 
# 
# 
# 
# 
# 
# #################################
# # train seq2seq w/o any weights #
# #################################
# 
# time python examples/train_model.py \
# -t personachat \
# -bs 128 \
# --hiddensize 1024 \
# -emb glove \
# --numlayers 1 \
# --bidirectional True \
# --embeddingsize 300 \
# --learningrate .001 \
# --optimizer adam \
# --gradient-clip 1.0 \
# --lr-scheduler-decay .5 \
# --lr-scheduler-patience 5 \
# --validation-patience 25 \
# --validation-every-n-epochs 1 \
# --validation-metric loss \
# --dict-minfreq 10 \
# --dict-lower True \
# --dict-tokenizer spacy \
# --dict-file tmp/personachat/dict_minfreq_10 \
# --tensorboard-log True \
# -m seq2seq \
# -mf tmp/personachat/seq2seq_model_minfreq_10 \
# > tmp/personachat/seq2seq_model_minfreq_10_train.out
# 
# 
# 
# ### EVAL model: s2s ###
# time python examples/eval_model.py \
# --datatype valid \
# -m seq2seq \
# -t personachat \
# --display-examples 1 \
# -df tmp/personachat/dict_minfreq_10 \
# -mf tmp/personachat/seq2seq_model_minfreq_10 \
# > tmp/personachat/seq2seq_model_minfreq_10_valid.out
# 
# 
# 
# ### TEST model: s2s ###
# time python examples/eval_model.py \
# --datatype test \
# -m seq2seq \
# -t personachat \
# --display-examples 1 \
# -df tmp/personachat/dict_minfreq_10 \
# -mf tmp/personachat/seq2seq_model_minfreq_10 \
# > tmp/personachat/seq2seq_model_minfreq_10_test.out
# 
# 
# 
