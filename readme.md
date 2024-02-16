# $\beta$-Variational autoencoders and transformers for reduced-order modelling of fluid flow

## Introduction
The code in this repository features a Python implementation of reduced-order model (ROM) of turbulent flow using $\beta$-variational autoencoders and transformer neural network. More details about the implementation and results from the training are available in ["$\beta$-Variational autoencoders and transformers for reduced-order modelling of fluid flow",Alberto Solera-Rico, Carlos Sanmiguel Vila, M. A. Gómez, Yuning Wang, Abdulrahman Almashjary, Scott T. M. Dawson, Ricardo Vinuesa](https://doi.org/10.1038/s41467-024-45578-4)

## Data availabilty
1. We share the down-sampled data in [zenodo](https://zenodo.org/records/10501216). 

2. We share the pre-trained models of $\beta$-VAE, transformers and LSTM with this repository.

## Training and inference 

+ To train and inference the easy-attention-based transformer, please run: 

        python main.py -re 40 -m run -nn easy 


## Structure

+ *data*: Dataset used for the present study 

+ *lib*: The main code used in the present study

+ *utils*: Support functions for visualisation, etc.

+ *configs*: Configurations of hyper parameters for models 

+ *nns*: The architecture of neural networks 

+ *res*: Storage of prediction results 

+ *figs*: Storaging figures output 
