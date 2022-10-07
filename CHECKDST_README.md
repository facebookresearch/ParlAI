# Getting started (CheckDST)

1. Create a folder for data inside `ParlAI`: i.e. `mkdir ParlAI/data` 
1. Place the data from the [data section](#data) into `ParlAI/data` 
1. Create an environment: `conda create -n checkdst python=3.8` 
1. Follow steps in [ParlAI](ParlAI/README.md) to install `ParlAI` locally 
    ```
    cd ParlAI 
    python setup.py develop 
    ```
    1. (optional) Before installing `ParlAI`, it may be necessary to replace torch version in `requirements.txt` with one with CUDA support that is compatible with available GPUs, e.g. for a100 gpus: `torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 torchtext -f https://download.pytorch.org/whl/torch_stable.html`  
    1. If this results in dependency errors when running parlai commands, run `python setup.py develop` again. 
1. (optional) Train a ParlAI model: e.g. `parlai train_model --task multiwoz_checkdst` 
1. Evaluate on all CheckDST: `parlai eval_model --task multiwoz_checkdst:aug=orig,multiwoz_checkdst:aug=SDI,multiwoz_checkdst:aug=TPI,multiwoz_checkdst:aug=NED --model-file $MODELDIR -dt test` 
    1. Evaluate on individiual augmentation: `parlai eval_model --task multiwoz_checkdst --model-file $MODELDIR --aug $AUGTYPE -dt test` where `$AUGTYPE` is one of `[SDI, TPI, NED]`
    1. Evaluate on original validation set: `parlai eval_model --task multiwoz_checkdst --model-file $MODELDIR -dt valid`
1. CheckDST results are available as `model.train_report_$AUGTYPE.json` files in the model directory. 
1. Metrics can be computed with a separate script that is universal for both TripPy and ParlAI models. (TBD)


Refer to the ParlAI [docs](https://www.parl.ai/docs/) for additional customization. 


### Overview 

- Scripts are in `bash_scripts/core`. 
- Trained models are in `models`
- Data is in `data` 

### Finetuning on MultiWOZ

- `bash_scripts/core/bart_multiwoz_finetune.sh` has the template for submitting a slurm job for finetuning a BART model on MultiWOZ. 
- `bash_scripts/core/bart_submit_ft_multiwoz.py` contains the python script for submitting multiple jobs with various configurations. 

