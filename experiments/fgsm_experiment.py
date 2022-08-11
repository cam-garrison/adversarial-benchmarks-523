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
import csv

from src.alexnet import model as alexnet_model, preprocess as alexnet_preprocess
from data.labels import labels, path_meta, path_synset_words
from src.fgsm import FGSM_Attack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loading in our 1,000 images used to test SimBA
# results/defined in SimBA_frozen.ipynb
labels_path = Path("experiments/results/used_labels.txt")
images_names_path = Path("experiments/results/used_images.txt")
images_subset_path = Path("experiments/results/subset_used/")


def get_names_labels():
    with open(labels_path, "r") as f:
        lines = f.readlines()
    labels = [line.rstrip() for line in lines]
    f.close()

    with open(images_names_path, "r") as f:
        lines = f.readlines()
    names = [line.rstrip() for line in lines]
    f.close()
    return labels, names


LABELS, NAMES = get_names_labels()
PATHS = ["experiments/results/subset_used/" + p for p in NAMES]

CSV_HEADER = [
    "epsilon",
    "pct_misclassified",
    "num_misclassified",
    "total_attempted",
    "avg_mask_norm",
]

with open("experiments/results/fgsm_results.csv", "w", encoding="UTF8", newline="") as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(CSV_HEADER)
f.close()


def get_alexnet_img(path):
    img = Image.open(path)
    return alexnet_preprocess(img)


if __name__ == "__main__":
    # set model to eval mode (no dropout etc)
    alexnet_model.eval()
    # send model to the device
    alexnet_model = alexnet_model.to(device)

    for ep in [0.001, 0.01, 0.02, 0.05]:
        # get the paths to all images
        img_paths = PATHS

        # get the ground truth labels
        y_val = np.array(LABELS, dtype=int)
        y_val = torch.from_numpy(y_val)

        # init variable to count number of examples
        num_examples = 0  # total we test
        num_perturbed = 0  # total we successfuly perturb
        total_norm = 0

        # init attack object with input epsilon
        fgsm_att = FGSM_Attack(alexnet_model, epsilon=ep)

        # iterate thru
        for i in trange(len(img_paths)):

            # get image, put it in device
            img_path = img_paths[i]
            try:
                this_img = get_alexnet_img(img_path)  # preprocess it too
            except:
                continue
            this_img = this_img.to(device)

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
        with open(
            "experiments/results/fgsm_results.csv", "a", encoding="UTF8", newline=""
        ) as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(
                [
                    ep,
                    num_perturbed / num_examples,
                    num_perturbed,
                    num_examples,
                    total_norm / num_examples,
                ]
            )
        f.close()
