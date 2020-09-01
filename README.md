# bias-gzsl

This is the source code for reproducing the experiments from the paper:  

[Bias-Awareness for Zero-Shot Learning the Seen and Unseen](https://arxiv.org/abs/2008.11185)  
**William Thong and Cees G.M. Snoek**  
British Machine Vision Conference (BMVC), 2020

**TL;DR** *We mitigate the classifier bias towards classes seen during training in generalized zero-shot learning (GZSL).
To achieve this, we calibrate the model with temperature and apply entropy regularization to balance out seen and unseen class probabilities.*

We validate the proposed approach on four benchmarks (CUB, SUN, AWA, and FLO) in two scenarios:  
(a) stand-alone classifier with only seen class features,  
(b) classifier with real seen class samples and generated unseen class features.

In other words, our bias-aware classifier behaves like a traditional compatibility functions for (G)ZSL and can integrate unseen class features from any existing generative model.

### Datasets

**Features** We rely on the features extracted by \[Xian, et al, CVPR 2018\] and use the same train/test splits. Sentences for CUB and FLO were collected by \[Reed et al, CVPR 2016\].  
* CUB, SUN, and AWA features are [here](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip).  
* CUB sentences are [here](https://www.dropbox.com/sh/btoc495ytfbnbat/AAAaurkoKnnk0uV-swgF-gdSa?dl=0).
* FLO features and sentences are [here](http://datasets.d2.mpi-inf.mpg.de/xian/cvpr18xian.zip).


##  Stand-alone classification with seen classes only

**GZSL** Run [runs/stand-alone.sh](runs/stand-alone.sh). This will train a bias-aware model with real seen features only. It will perform GZSL at the end of the training for each of the four benchmarks.

**Swapping representations** Run [runs/swapping.sh](runs/swapping.sh). This will train one model with *attributes* representations, and one with *sentence* representations. Then, it will extract the features and perform a least squares estimation for swapping the representations between seen and unseen classes.

##  Classification with both seen and unseen classes

**Generating unseen class features** We rely on existing models to generate unseen class features. We select [f-CLSWGAN](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning) and [cycle-CLSWGAN](https://github.com/rfelixmg/frwgan-eccv18), which are publicly available and reproducible.

We train both and extract the features to be used in our bias-aware classifier. Note that our bias-aware classifier is not limited to these two methods to generate features, and can be used for any other method that produces generated features! For this, you just need to save into an `.npz` file the following:
```
train_seen_X
train_seen_Y
train_unseen_X
train_unseen_Y
test_seen_X
test_seen_Y
test_unseen_X
test_unseen_Y
```
or alternatively with seen and unseen merged in the training set:
```
train_X  # includes both real and generated features
train_Y
test_seen_X
test_seen_Y
test_unseen_X
test_unseen_Y
```

For convenience, [here](https://isis-data.science.uva.nl/wthong/cyclegan.tar.gz) are features from cycle-CLSWGAN for each dataset (900MB file). Note that it corresponds to features for one seed point. In the paper we report the average over 3 runs, so numbers might be different.

**GZSL** Run [runs/with-generated.sh](runs/with-generated.sh). This will train a bias-aware model with real seen features and generated unseen features from cycle-CLSWGAN. It perform GZSL at the end of the training for each of the four benchmarks.

## Requirements

The source code relies on python 3.6 and pytorch 0.4:

```
python=3.6.9
pandas=0.25.2
numpy=1.17.3
pytorch=0.4.1
torchvision=0.2.1
scikit-learn=0.22.1
```

## Citation

If you find these scripts useful, please consider citing our paper:

```
@inproceedings{
    Thong2020BiasGZSL,
    title={Bias-Awareness for Zero-Shot Learning the Seen and Unseen},
    author={Thong, William and Snoek, Cees G.M.},
    booktitle={BMVC},
    year={2020},
    url={https://arxiv.org/abs/2008.11185}
}
```
