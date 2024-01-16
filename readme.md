# $\beta$-Variational autoencoders and transformers for reduced-order modelling of fluid flow


## Introduction
The code in this repository features a Python implementation of reduced-order model (ROM) of turbulent flow using $\beta$-variational autoencoders and transformer neural network. More details about the implementation and results from the training are available in ["$\beta$-Variational autoencoders and transformers for reduced-order modelling of fluid flow",Alberto Solera-Rico, Carlos Sanmiguel Vila, M. A. Gómez, Yuning Wang, Abdulrahman Almashjary, Scott T. M. Dawson, Ricardo Vinuesa](https://arxiv.org/abs/2304.03571)

## Data availabilty
We share the down-sampled data in [zendo](https://zenodo.org/records/10501216). We also provide the pre-trained models of $\beta$-VAE, transformers and LSTM in this repository. The obtined results such as temporal and spatial modes are available.


## Structure

+ *data*: Dataset used for the present study 

+ *utils*: Functions

+ *conf*: Configurations for hyper parameters 

+ *nns*: The architecture of neural networks 

+ *res*: Storage of results and figures

+ 