# MergedNET: A Simple Approach for One-Shot Learning in Siamese Networks based on Similarity Layers.
The project investigates similarity layers for the one-shot learning tasks and proposes two frameworks for merging these layers into a network called MergedNet. We show that MergedNet improved classification accuracy compared with the baselines on all the four datasets used in our experiment. Our MergedNet network trained on the miniImageNet dataset generalises to other datasets. 

<img src="architecture.png" />

# Dataset used in these experiments

<b>CUB-200-2011 </b> is a fine-grained dataset, which has 200 classes with a total of 11,788 images. We used a similar dataset split (100, 50 and 50 classes for training, validation and testing, respectively) to what was proposed in Chen et al. 2019.

The <b>Caltech256 </b> dataset contains 30,607 real-world images. It has 257 classes, but for few-shot learning, the clutter class is removed. We have used a dataset split similar to in Chen et al. 2019.

The <b>MiniImageNet</b> contains 100 randomly selected classes from the original ImageNet dataset with 600 images per class. We split the dataset according to in Chen et al. 2019.

The<b>CIFAR-100</b> dataset contains 100 classes with 600 images each. We followed the recommendation in in Chen et al. 2019 to split the dataset.

# Minimum dependencies required to use these codes:
Python 3.6.1
Keras 2.0.6
Tensorflow 1.3.0
Numpy 1.13.3
Pillow 5.1.0
Opencv 3.2.0
