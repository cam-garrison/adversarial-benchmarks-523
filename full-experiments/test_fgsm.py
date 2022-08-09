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


def disp_img(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

    im = Image.open(path)

    # alexnet preprocessing
    im_prepro = alexnet_preprocess(im)
    plt.imshow(im_prepro.permute(1, 2, 0))
    plt.show()


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

    disp_img(img_path)

    this_img = get_alexnet_img(img_path)
    this_img.to(device)

    fgsm_att = FGSM_Attack(alexnet_model)

    logits = alexnet_model(this_img.view(-1, 3, 224, 224))
    preds = torch.nn.functional.softmax(logits, dim=1)
    argmax = int(torch.argmax(preds))
    if argmax == y_val[10]:
        print("correct pred\n")

    print("pred class", labels[y_val[10].item()], "with conf", preds[0][argmax])

    # confidence threshold
    if preds[0][argmax] > 0.50:
        print("passed conf check, do attack")

        perted_img, norm = fgsm_att.fgsm(this_img, y_val[10])
        print("norm is", norm)
        perted_logits = alexnet_model(perted_img.view(-1, 3, 224, 224))
        perted_preds = torch.nn.functional.softmax(perted_logits, dim=1)
        perted_argmax = int(torch.argmax(perted_preds))
        if perted_argmax == y_val[10]:
            print("didnt change pred")
        else:
            print("prediction is now wrong")
            print(
                "pred class",
                labels[y_val[perted_argmax].item()],
                "with conf",
                perted_preds[0][perted_argmax],
            )
            plt.imshow(perted_img.cpu().detach().permute(1, 2, 0))
            plt.show()
