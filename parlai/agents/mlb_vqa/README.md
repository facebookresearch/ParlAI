# MLB VQA
 The *MLB VQA* agent is an implementation of the Multimodal Low-rank Bilinear
 Attention Network outlined in https://arxiv.org/abs/1610.04325,
 that can be used with the VQA V1 and VQA V2 datasets.

 ## Basic Examples
 Train the agent on the VQA V1 dataset, using the `PytorchDataTeacher` for
 fast dataloading.
```bash
python examples/train_model.py -m mlb_vqa -pytd vqa_v1 -mf /tmp/mlb -bs 512 -im resnet152_spatial --image-size 448 --image-cropsize 448
```
 After training, load and evaluate that model on the VQA V2 test set.
```bash
python examples/eval_model.py -pytd vqa_v2 -mf /tmp/mlb -dt test
```
