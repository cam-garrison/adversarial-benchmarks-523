# Adversarial Benchmarking - Final Project for CS523 Boston University

Benchmarking Adversarial Attacks against AlexNet

Presentation Link: https://docs.google.com/presentation/d/1UXHFWQ0afXJib2Vdh4D6hqRmEcu75T1tWALIkn3cmXk/edit?usp=sharing 

## Group Members

- James Kunstle - jkunstle@bu.edu - [@JamesKunstle](https://github.com/JamesKunstle).

- Cameron Garrison - cgarriso@bu.edu - [@cam-garrison](https://github.com/cam-garrison).

## Citations

- "AlexNet", as introduced in the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al. [(original paper)](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

- "SimBA", as introduced in the paper "Simple Black-box Adversarial Attacks" by Chuan Guo et al. [(original paper)](https://arxiv.org/abs/1905.07121). 

- "Fast Gradient Sign Method (FGSM)" as intoduced in the paper "Explaining and Harnessing Adversarial Examples" by Ian J. Goodfellow et al. [(original paper)](https://arxiv.org/abs/1412.6572).


## **Usage**

### **Dependencies:**

We have provided `requirements.txt`. We recommend running:

```
pip3 install -r requirements.txt
```

in a python venv to get all the requisite package versions.

### **Reproducing Results**

We attack the AlexNet model w.r.t 1000 images that the model correctly identifies. These images are chosen randomly from the ILSVRC 2012 Imagenet Validation set.

In our notebook `experiments/SimBA_frozen.ipynb` we sample sample the 1000 images and run our SimBA attack against them. This attack is implemented in `experiements/src/simba.py`.

The results from this attack are stored in the folder `experiments/results/` and includes the list of all images that the attack was successful on (used\_images.txt), their labels (used\_labels.txt), and 
the masks that were created from the attack (sorted\_outputs.p). This final file is too large to go on Github so it's at this link: 

[Drive Link](https://drive.google.com/drive/folders/1VJ9aDBnFXGI6_92b97ICLRIO9mg7WCAY?usp=sharing)

Our analysis of the data from the SimBA attack is in the notebook `experiments/analysis.ipynb`.

We attack these same images with FGSM.

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

Our analysis on the attack done by FGSM is done in the notebook `experiments/fgsm_analysis.ipynb`.

## Structure:

    ├── experiments             
        ├── data     # Store label map, metadata, ImageNet data.
        ├── results  # Store subset of dataset, experiment output csvs and data structures.
        ├── src      # Store attacks, alexet loader.
    ├── LICENSE
    ├── requirements.txt        # Standard python requirements file.
    └── README.md

