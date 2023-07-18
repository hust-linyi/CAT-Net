# Masked Attention Transformer (CAT-Net)
This is the official code for our MICCAI 2023 paper:

> [Few Shot Medical Image Segmentation with Cross Attention Transformer](https://arxiv.org/abs/2303.13867) <br>
> Yi Lin*, Dong Zhang*, Xiao Fang, Yufan Chen, Kwang-Ting Cheng, Hao Chen

## Highlights
<p align="justify">
In this work, we propose a novel framework for few-shot medical image segmentation, termed CAT-Net, based on cross masked attention Transformer. Our proposed network mines the correlations between the support image and query image, limiting them to focus only on useful foreground information and boosting the representation capacity of both the support prototype and query features. We further design an iterative refinement framework that refines the query image segmentation iteratively and promotes the support feature in turn.

### Using the code
Please clone the following repositories:
```
git clone https://github.com/hust-linyi/CAT-Net
```

### Requirement
```
pip install -r MedISeg/requirements.txt
```

### Data preparation


### Training & Evaluation


## Citation
Please cite the paper if you use the code.
```bibtex
@article{lin2023few,
  title={Few Shot Medical Image Segmentation with Cross Attention Transformer},
  author={Lin, Yi and Chen, Yufan and Cheng, Kwang-Ting and Chen, Hao},
  journal={arXiv preprint arXiv:2303.13867},
  year={2023}}
```

## Acknowledgment 
Our code is partially based on [xxx](https://youtube.com). 
