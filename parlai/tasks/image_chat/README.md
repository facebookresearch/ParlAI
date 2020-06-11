Task: Image_Chat
=================
Description: 202k dialogues and 401k utterances over 202k images from the YFCC100m dataset(https://multimediacommons.wordpress.com/yfcc100m-core-dataset/)using 215 possible personality traitssee https://klshuster.github.io/image_chat/ for more information.

Tags: #Image_Chat, #All, #Visual, #ChitChat

Notes: If you have already downloaded the images, please specify with the `--yfcc-path` flag, as the image download script takes a very long time to run

If you just want to download data, run as `./parlai/tasks/image_chat/download_data.sh`. Change the required `$DATA_DIR` variable to where you want to save the file. Defaults to `/tmp`. It basically calls the wrapper `parlai/tasks/image_chat/download_data.py`  
