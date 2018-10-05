# Fairseq

This agent is a wrapper for Facebook AI Research's fairseq so it can be used an agent in ParlAI. Read more about fairseq [here](https://github.com/facebookresearch/fairseq).


## Basic Examples

Train a fairseq transformer model on the OpenSubtitles task. (*NOTE:* You will be prompted to install fairseq if you have not already.)
```bash
python examples/train_model.py -m fairseq --arch transformer -t opensubtitles -mf /tmp/fairseq_opensubtitles.mdl
```
