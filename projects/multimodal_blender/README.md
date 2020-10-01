# [TODO: change] Recipes for building an open-domain chatbot

[TODO: change] Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston

## Abstract

[TODO: change] Building open-domain chatbots is a challenging area for machine learning research.  While prior work has shown that scaling neural models in the number of parameters and the size of the data they are trained on gives improved results, we show that other ingredients are important for a high-performing chatbot.
Good conversation requires a number of skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to their partners, both asking and answering questions, and displaying knowledge, empathy and personality appropriately, depending on the situation.
We show that large scale models can learn these skills when given appropriate training data and choice of generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter neural models, and make our models and code publicly available. Human evaluations show our best models are superior to existing approaches in  multi-turn dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing failure cases of our models.

## Paper

[TODO: change] [Link](https://arxiv.org/abs/2004.13637)

## Example conversations

[TODO: change] <p align="center"><img width="50%" src="steve_jobs.png" /></p>
<hr />
<p align="center"><img width="50%" src="funguy.png" /></p>

## Training and safety

{{{TODO: say see here for details on training + safety, and give link}}}

## Citation

If you use the models in your own work, please cite with the following BibTex entry:

[[[TODO: change]]]
```
@inproceedings{roller2020recipes,
  author={Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston},
  title={Recipes for building an open-domain chatbot},
  journal={arXiv preprint arXiv:2004.13637},
  year={2020},
}
```
