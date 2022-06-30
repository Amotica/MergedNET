# MergedNET: A Simple Approach for One-Shot Learning in Siamese Networks based on Similarity Layers.
The project investigates similarity layers for the one-shot learning tasks and proposes two frameworks for merging these layers into a network called MergedNet. We show that MergedNet improved classification accuracy compared with the baselines on all the four datasets used in our experiment. Our MergedNet network trained on the miniImageNet dataset generalises to other datasets. 

<img src="architecture.png" />

# Running the codes:
<ul>
<li>The files config.py and siameseModelv2.py are required to run the codes. The config.py contain configuration parameters, which we sometimes override in the siameseModelv2.py file. The siameseModelv2.py file contains all the class(es) and methods to train and evaluate the model.</li>
<li>To train the model ensure the parameter (eval = True) in the “if __name__ == '__main__':” is set to False.</li>
<li>You should also ensure the datasets are downloaded from the respective websites and placed in a dataset folder (self-explanatory from the source codes).</li>
</ul>



# Dataset used in these experiments

<b>CUB-200-2011 </b> is a fine-grained dataset, which has 200 classes with a total of 11,788 images. We used a similar dataset split (100, 50 and 50 classes for training, validation and testing, respectively) to what was proposed in Chen et al. 2019.

The <b>Caltech256 </b> dataset contains 30,607 real-world images. It has 257 classes, but for few-shot learning, the clutter class is removed. We have used a dataset split similar to in Chen et al. 2019.

The <b>MiniImageNet</b> contains 100 randomly selected classes from the original ImageNet dataset with 600 images per class. We split the dataset according to in Chen et al. 2019.

The<b>CIFAR-100</b> dataset contains 100 classes with 600 images each. We followed the recommendation in in Chen et al. 2019 to split the dataset.

# Minimum dependencies required to use these codes:
<ul>
  <li>python=3.6.12</li>
  <li>tensorboard=2.6.0</li>
  <li>tensorflow-datasets=4.2.0</li>
  <li>numpy=1.19.5</li>
  <li>matplotlib=3.3.3</li>
  <li>pillow=8.0.1</li>
  <li>keras=2.3.1</li>
  <li>keras-applications=1.0.8</li>
  <li>keras-efficientnets=0.1.7</li>
  <li>keras-preprocessing=1.1.2</li>
  <li>keras-resnet=0.1.0</li>
  </ul>
