# Engaging Image Captioning via Personality

Kurt Shuster, Samuel Humeau, Hexiang Hu, Antoine Bordes, Jason Weston

Please see [Shuster et al. (CVPR 2019)](https://arxiv.org/abs/1810.10665) for more details.

## Abstract

Standard image captioning tasks such as COCO and Flickr30k are factual, neutral in tone and (to a human) state the obvious (e.g., "a man playing a guitar"). While such tasks are useful to verify that a machine understands the content of an image, they are not engaging to humans as captions. With this in mind we define a new task, Personality-Captions, where the goal is to be as engaging to humans as possible by incorporating controllable style and personality traits. We collect and release a large dataset of 201,858 of such captions conditioned over 215 possible traits. We build models that combine existing work from (i) sentence representations (Mazare et al., 2018) with Transformers trained on 1.7 billion dialogue examples; and (ii) image representations (Mahajan et al., 2018) with ResNets trained on 3.5 billion social media images. We obtain state-of-the-art performance on Flickr30k and COCO, and strong performance on our new task. Finally, online evaluations validate that our task and models are engaging to humans, with our best model close to human performance.

## Dataset

The Personality-Captions dataset can be accessed via ParlAI, with `-t personality_captions`.

Additionally, the ParlAI MTurk tasks for data collection and human evaluation
are [made available](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/personality_captions) in ParlAI.

## Leaderboards for Personality-Captions Task

### Retrieval Models

Model                                | Paper          | Test R@1
------------------------------------ | -------------- | --------
TransResNet, ResNeXt-IG-3.5B         | [Shuster et al. (2019)](https://arxiv.org/abs/1810.10665) | 77.5
TransResNet, ResNet152               | [Shuster et al. (2019)](https://arxiv.org/abs/1810.10665) | 51.7
TransResNet, No Images               | [Shuster et al. (2019)](https://arxiv.org/abs/1810.10665) | 25.8

### Generative Models

Model                                | Paper          | BLEU1 | BLEU4 | ROUGE-L | CIDEr | SPICE
------------------------------------ | -------------- | ----- | ----- | ------- | ----- | -----
UpDown, ResNeXt-IG-3.5B              | [Shuster et al. (2019)](https://arxiv.org/abs/1810.10665) | 44.0 | 8.0 | 27.4 | 16.5 | 5.2
ShowAttTell, ResNeXt-IG-3.5B         | [Shuster et al. (2019)](https://arxiv.org/abs/1810.10665) | 43.3 | 7.1 | 27.0 | 12.6 | 3.6
ShotTell, ResNeXt-IG-3.5B            | [Shuster et al. (2019)](https://arxiv.org/abs/1810.10665) | 38.4 | 7.3 | 24.3 | 9.6 | 1.6


## Pretrained Models

Pretrained models are forthcoming; this website will be updated when they are
available for download.

## Model Examples

<p align="center"><img width="85%" src="Examples.png" /></p>

## Citation

If you use the dataset or models in your own work, please cite with the following BibText entry:

    @inproceedings{shuster2019_personality_caps,
      author={Shuster, Kurt and Humeau, Samuel and Hu, Hexiang and Bordes, Antoine and Weston, Jason},
      title={Engaging Image Captioning via Personality},
      booktitle = {2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2019},
    }
