import torch
import torchvision
import numpy as np
from pathlib import Path
from glob import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
from tqdm import trange
import sys

sys.path.insert(0, "src")

from alexnet import model as alexnet_model, preprocess as alexnet_preprocess

from labels import labels, path_meta, path_synset_words
from main import get_alexnet_img, get_labels, get_paths

from fgsm import FGSM_Attack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device used: {device}")

"""
     Data loading from: 
        https://github.com/calebrob6/imagenet_validation/blob/master/\
        1.%20Preprocess%20ImageNet%20validation%20set.ipynb
"""

val_data_path = Path("./imagenet/val/ILSVRC2012_img_val/")
val_data_labels = Path(
    "./imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
)

if __name__ == "__main__":
    # send model to the device
    alexnet_model.to(device)
    # set model to eval mode (no dropout etc)
    alexnet_model.eval()

    for ep in [0.001, 0.02, 0.05]:
        # get the paths to all images
        img_paths = get_paths()

        # get the ground truth labels
        y_val = get_labels()
        y_val = torch.from_numpy(y_val)

        # init variable to count number of examples
        num_examples = 0  # total we test
        num_perturbed = 0  # total we successfuly perturb
        total_norm = 0

        # init attack object with input epsilon
        fgsm_att = FGSM_Attack(alexnet_model, epsilon=ep)

        # iterate thru
        # for i in trange(len(img_paths)):
        for i in trange(1000):
            # get image, put it in device
            img_path = img_paths[i]
            try:
                this_img = get_alexnet_img(img_path)  # preprocess it too
            except:
                continue
            this_img.to(device)

            # lets pass it into the model, and see if its right.
            logits = alexnet_model(this_img.view(-1, 3, 224, 224))
            preds = torch.nn.functional.softmax(logits, dim=1)
            argmax = int(torch.argmax(preds))

            # model is incorrect before we perturb, skip
            if argmax != y_val[i]:
                continue

            # let's do our attack
            perted_img, norm = fgsm_att.fgsm(this_img, y_val[i])

            # test if our attack worked and fooled it
            perted_logits = alexnet_model(perted_img.view(-1, 3, 224, 224))
            perted_preds = torch.nn.functional.softmax(perted_logits, dim=1)
            perted_argmax = int(torch.argmax(perted_preds))

            # update our examples and perturbed examples
            num_examples += 1
            if perted_argmax != y_val[i]:
                num_perturbed += 1

            # update our total norm
            total_norm += norm.item()

        print(
            "\n",
            "epsilon:",
            ep,
            "\n",
            num_perturbed / num_examples,
            "\n",
            total_norm / num_examples,
        )
