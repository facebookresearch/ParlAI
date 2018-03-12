# Interact with pre-trained model
# Language model trained on Opensubtitles 2018 dataset
# Run from ParlAI directory

if [ ! -f languagemodel_esz512_hid1024_nl2.pt ];  then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/languagemodel_esz512_hid1024_nl2.pt
fi
if [ ! -f opensubtitles2018.dict ]; then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/opensubtitles2018.dict
fi
python examples/interactive.py -mf 'languagemodel_esz512_hid1024_nl2.pt' -m 'language_model' --dict-file 'opensubtitles2018.dict' --sampling-mode True
