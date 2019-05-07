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
