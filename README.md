# PacketGAN
Generate synthetic network attack packet flows using generative adversarial networks.

About
-----

The motivation for this project is to improve the capabilities of modern Machine Learning based 
Intrusion Detection Systems for detecting low-footprint attacks in modern networks. These attacks
are difficult to train ML systems on because by their very nature, they provide very little data
(attacks take place over few packets). This projects is to use Generative Adversarial Networks to
synthesize network "attack" packet flows which have the same characteristics as these low-footprint
attacks we wish to detect. In doing so, we could generate an arbitrary amount of training data
to use with ML-based IDS systems. 

This is a research project in progress. I (Brandon Foltz) worked on it as an undergraduate at 
Temple University during the 2019 Spring semester under the supervision of Professor Jamie Payton. 

Getting Started
---------------

This work is built on the PyTorch deep learning platform. The easiest way to get up and running is
by installing PyTorch in an Anaconda3 environment, since that will already have Jupyter Notebooks
available as well. 

See here for installing PyTorch: https://pytorch.org/get-started/locally/

The "real" packets dataset used in this project is the UNSW-NB15 dataset. 


You can find this dataset here: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/


The code in this repository provides a PyTorch dataset class and functions for loading this data
into an ML-useful form. You may need to do some data cleaning on the UNSW-NB15 data files prior
to using them, some of the port numbers were stored in hexadecimal which my code is not written 
to handle, since the vast majority of the data is stored as base-10 digits.

What files to use?
------------------

There are several Jupyter Notebooks in this repository. Some of them are relevant/up to date and some should probably be deleted. The plain .py Python files contain logic that is used by various Jupyter Notebooks, so that the notebooks don't become overly bloated with details. 

*PacketGAN.ipynb* : The meat-and-potatoes of this project. This file loads the dataset, trains the GAN, and runs evaluation metrics. 

*WaveGAN.ipynb* : A proof-of-concept notebook to show that a GAN can be built to generate sequential data, and validate the evaluation metrics.

*PyTorch_GAN_Tutorial_from_Medium.ipynb* : What it sounds like. A basic GAN tutorial showing that a normal distribution can be learned from a uniform distribution by adversarial training. 

*Latent Space Classifier.ipynb* : Attempting to validate methods for mapping sequential data into a latent vector space for classification and evaluation metric generation. 

*networks.py* : Contains definitions of various neural networks used throughout the project.

*feature_extraction.py* : Mapping UNSW-NB15 packet features into a form that can be used as input/output with a neural network.

*evaluation.py* : Various functions for calculating evaluation metrics.

*train.py* : Contains the GAN training loop used by PacketGAN and WaveGAN.

*benchmark_timer.py* : A basic timer for calculating elapsed time.

*progress_bar.py* : Render a progress bar in Jupyter Notebooks.

*log_stats.py* : Tools for accumulating/logging statistics such as averages on various tasks.

*unsw_nb15_dataset.py* : Pytorch Dataset class for loading the UNSW-NB15 dataset.

License
-------

This work is licensed under the MIT license, see LICENSE.txt for more information.
