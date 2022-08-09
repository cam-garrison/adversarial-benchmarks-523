import torch
import torchvision
import numpy as np
from pathlib import Path
from glob import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
from tqdm import trange
import sys

sys.path.insert(0, "src")

from alexnet import model as alexnet_model, preprocess as alexnet_preprocess

from labels import labels, path_meta, path_synset_words
from main import get_alexnet_img, get_labels, get_paths, disp_img

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
    alexnet_model.eval()

    # get the paths to all images
    img_paths = get_paths()

    # get the ground truth labels
    y_val = get_labels()
    y_val = torch.from_numpy(y_val)

    img_path = img_paths[10]

    this_img = get_alexnet_img(img_path)
    this_img.to(device)

    fgsm_att = FGSM_Attack(alexnet_model)

    logits = alexnet_model(this_img.view(-1, 3, 224, 224))
    preds = torch.nn.functional.softmax(logits, dim=1)
    argmax = int(torch.argmax(preds))
    if argmax == y_val[10]:
        print("correct pred")

    print(preds[0][argmax])

    # confidence threshold
    if preds[0][argmax] > 0.50:
        print("passed conf check, do attack")

        perted_img = fgsm_att.fgsm(this_img, y_val[10])
        perted_logits = alexnet_model(perted_img.view(-1, 3, 224, 224))
        perted_preds = torch.nn.functional.softmax(perted_logits, dim=1)
        perted_argmax = int(torch.argmax(perted_preds))
        if perted_argmax == y_val[10]:
            print("didnt change pred")
        else:
            print("prediction is now wrong")
