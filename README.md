# Adversarial Benchmarking - Final Project for CS523 Boston University

Benchmarking adversarial techniques effectiveness against well-known ImageNet classifiers.

## Group Members

- James Kunstle - jkunstle@bu.edu - [@JamesKunstle](https://github.com/JamesKunstle).

- Cameron Garrison - cgarriso@bu.edu - [@cam-garrison](https://github.com/cam-garrison).

## Citations

- "AlexNet", as introduced in the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al. [(original paper)](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

- "SimBA", as introduced in the paper "Simple Black-box Adversarial Attacks" by Chuan Guo et al. [(original paper)](https://arxiv.org/abs/1905.07121). 

- "Fast Gradient Sign Method (FGSM)" as intoduced in the paper "Explaining and Harnessing Adversarial Examples" by Ian J. Goodfellow et al. [(original paper)](https://arxiv.org/abs/1412.6572).


## **Usage**

### **Dependencies:**

We have provided `requirements.txt`. We recommend running 

```
pip install -r requirements.txt
```

in a python venv to get all the requisite package versions.

### **Reproducing Results**

We have selected a subset of the ILSVRC Imagenet 2012 validation dataset for our usage. 

To see how that subset of data was selected, and how SimBA was ran on it, see `experiments/simba_frozen.ipynb`. This notebook also includes how to replicate those results by downloading the validation set. 

That yields us with ~1000 images from the ImageNet dataset. 

To replicate the experiment results found with FGSM, simply run 
```
python experiments/fgsm_experiment.py
```

**FROM THE ROOT DIRECTORY OF THE REPOSITORY^**

This will repopulate the csv of metrics and results in `experiments/results/fgsm_results.csv`.

To view an image attacked with FGSM, run 

```
python experiments/test_fgsm.py
```

**FROM THE ROOT DIRECTORY OF THE REPOSITORY^**

## Structure:

    ├── experiments             
        ├── data     # Store label map, ImageNet data.
        ├── results  # Store subset of dataset, experiment output csvs. 
        ├── src      # Store attacks, alexet loader.
    ├── LICENSE
    ├── requirements.txt        # Standard python requirements file.
    └── README.md

