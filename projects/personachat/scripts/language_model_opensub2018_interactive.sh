# Interact with pre-trained model
# Language model trained on Opensubtitles 2018 dataset
# Run from ParlAI directory

if [ ! -d data/models/opensubtitles2018 ]; then
  mkdir -p data/models/opensubtitles2018
fi
if [ ! -f data/models/opensubtitles2018/languagemodel_esz512_hid1024_nl2.pt ];  then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/languagemodel_esz512_hid1024_nl2.pt -P data/models/opensubtitles2018
fi
if [ ! -f data/models/opensubtitles2018/opensubtitles2018.dict ]; then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/opensubtitles2018.dict -P data/models/opensubtitles2018
fi
python examples/interactive.py -mf 'data/models/opensubtitles2018/languagemodel_esz512_hid1024_nl2.pt' -m 'language_model' --dict-file 'data/models/opensubtitles2018/opensubtitles2018.dict' --sampling-mode True
