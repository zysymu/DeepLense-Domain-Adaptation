# DeepLense Domain Adaptation
A PyTorch-based collection of Unsupervised Domain Adaptation methods applied to strong gravitational lenses!

This project was created for Google Summer of Code 2021 under the Machine Learning for Science (ML4Sci) umbrella organization.

### Description
A promising means to identify the nature of dark matter is to study it through dark matter halos, and strong gravitational lenses have seen encouraging results in detecting the existence of dark matter substructure. Unfortunately, there isn't a lot of data of strong gravitational lenses available, which means that, if we want to train a machine learning model to identify the different kinds of dark matter substructure, we'd need to use simulations. The problem though, is that a model trained on simulated data does not generalize well to real data, having a very bad performance. This project aims to fix this problem by using Unsupervised Domain Adaptation (UDA) techniques to adapt a model trained on simulated data to real data!

### Blog post
For more about the motivation behind the project and also my Google Summer of Code experience check out [this blog post](https://medium.com/@marcostidball/gsoc-2021-with-ml4sci-domain-adaptation-for-decoding-dark-matter-bf0380898aed).

# Algorithms
There are currently four different UDA algorithms supported:
- ADDA
- Self-Ensemble
- CGDM
- AdaMatch

There is also a normal supervised training algorithm

# Installation
This repo's code can be acessed through the `deeplense_domain_adaptation` package. To install it simply do:
```shell
pip install --upgrade deeplense_domain_adaptation
```

# Data
The data loading pipeline implemented here expects the image data to be in the form of a four dimensional numpy array of shape: `(number_of_images, 1, height, width)`. Label data is expected to have a shape: `(number_of_images, 1)`.

The dataset used for training and inference will be made available as soon as possible! It consists of a source and a target dataset, both were simulated using PyAutoLens according to the following paper: [Decoding Dark Matter Substructure without Supervision](https://arxiv.org/abs/2008.12731).

The paper's Model A is our source dataset (less complex simulations) and the paper's Model B is our target dataset (more complex simulations). We have three distinct classes: no dark matter substructure, spherical dark matter substructure and vortex dar matter substructure. On our training sets we have 30'000 images for the source and 30'000 images for the target; in both cases there are 10'000 images per class. On our test sets we have 7'500 images for the source and 7'500 images for the target; in both cases there are 2'500 images per class.

# How to use `deeplense_domain_adaptation`
For a tutorial on how to use the `deeplense_domain_adaptation` package check out `tutorial.ipynb`. If the file isn't loading properly on GitHub you can also check the Jupyter Notebook on nbviewer [here](https://nbviewer.jupyter.org/github/zysymu/DeepLense-Domain-Adaptation/blob/main/tutorial.ipynb). For more information on specific functions/classes check out the documentation available on the functions/classes definitions.

# Before and after UDA
### Equivariant Network model
- Supervised training on source infering on **source**: accuracy = 96.8400; AUROC = 0.9964.
- Supervised training on source infering on **target**: accuracy = 67.5333; AUROC = 0.8558.

- Applying UDA and infering on target:
| Algorithm             | Adda    | Self-Ensemble | CGDM    | AdaMatch |
|-----------------------|---------|---------------|---------|----------|
| Accuracy              | 91.4666 | 80.0933       | 74.8133 | 85.8133  |
| AUROC (macro-average) | 0.9798  | 0.9391        | 0.8891  | 0.9600   |

### ResNet-18
- Supervised training on source infering on **source**: accuracy = 97.0933; AUROC = 0.9959.
- Supervised training on source infering on **target**: accuracy = 59.1866; AUROC = 0.8797.

- Applying UDA and infering on target:
| Algorithm             | Adda    | Self-Ensemble | CGDM    | AdaMatch |
|-----------------------|---------|---------------|---------|----------|
| Accuracy              | 85.8400 | 76.7066       | 75.1866 | 75.5466  |
| AUROC (macro-average) | 0.9552  | 0.9174        | 0.9139  | 0.9195   |