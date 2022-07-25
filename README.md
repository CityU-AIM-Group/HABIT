# HABIT
This is an official PyTorch implementation of the paper "Hierarchical Bias Mitigation for Semi-Supervised Medical Image Classification"

## Introduction

We retrospect the pseudo labeling and identify three hierarchical biases: perception bias, selection bias and confirmation bias, at feature extraction, pseudo label selection and momentum optimization stages, respectively. In this regard, we propose a HierArchical BIas miTigation (HABIT) framework to amend these biases, which consists of three customized modules including Mutual Reconciliation Network (MRNet), Recalibrated Feature Compensation (RFC) and Consistency-aware Momentum Heredity (CMH). Firstly, in the feature extraction, MRNet is devised to jointly utilize convolution and permutator-based paths with a mutual information transfer module to exchanges features and reconcile spatial perception bias for better representations. To address pseudo label selection bias, RFC adaptively recalibrates the strong and weak augmented distributions to be a rational discrepancy and augments features for minority categories to achieve the balanced training. Finally, in the momentum optimization stage, in order to reduce the confirmation bias, CMH models the consistency among different sample augmentations into network updating process to improve the dependability of the model.

## Dependencies
All experiments use PyTorch library. We recommend installing following package versions:

* &nbsp;&nbsp; python==3.7 

* &nbsp;&nbsp; pytorch==1.6.0

* &nbsp;&nbsp; MedPy==0.4.0

* &nbsp;&nbsp; scipy==1.5.4

Dependency packages can be installed using following command:
```
pip install -r requirements.txt
```

## Quickstart

### Training
```python
python Train_HABIT.py 
    --gpu=0 
    --dataset=isic2018 
    --n-classes=7 
    --backbone=ConvMLP
    --n-labeled=350 
    --RFC=True
    --CMH=True
    -batchsize=12 
    --mu=3
    --thr=0.95 
    --n-epoches=256 
    --setting=ISIC_350
```


## Acknowledgement
* The implementation of baseline method is adapted from: [TEAR](https://github.com/QiushiYang/TEAR) and [MLP-Mixer](https://github.com/lucidrains/mlp-mixer-pytorch).
