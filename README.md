# MergedNET: A Simple Approach for One-Shot Learning in Siamese Networks based on Similarity Layers.
The project investigates similarity layers for the one-shot learning tasks and proposes two frameworks for merging these layers into a network called MergedNet. We show that MergedNet improved classification accuracy compared with the baselines on all the four datasets used in our experiment. Our MergedNet network trained on the miniImageNet dataset generalises to other datasets. 

<img src="architecture.png" />
# Dataset used in these experiments


# Minimum dependencies required to use these codes:
Python 3.6.1

Keras 2.0.6

Tensorflow 1.3.0

Numpy 1.13.3

Pillow 5.1.0

Opencv 3.2.0

# Running the codes:
Use train.py to train the baseline models and train_lite.py can be used to train the "Lite" models

Use evaluate.py to evaluate the baseline models and evaluate_lite.py to evaluate the "Lite" models.

Most network parameters can be changed in the parameter file.

All CNN models are contained in the Models folder

Dataset not included but can be downloaded from http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
