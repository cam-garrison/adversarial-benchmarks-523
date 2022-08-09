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

from alexnet import model as alexnet_model, preprocess as alexnet_preprocess
from resnet import model as resnet_model, preprocess as resnet_preprocess
from googlenet import model as googlenet_model, preprocess as googlenet_preprocess
from labels import labels, path_meta, path_synset_words

from simba import SimBA_Attack

IMG_INDEX = 4 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device used: {device}")

"""
     Data loading from: 
        https://github.com/calebrob6/imagenet_validation/blob/master/\
        1.%20Preprocess%20ImageNet%20validation%20set.ipynb
"""

val_data_path = Path("./imagenet/val/ILSVRC2012_img_val/")
val_data_labels = Path(
    "./imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")


def get_paths():
    image_paths = sorted(glob(str(val_data_path/"*")))
    return image_paths


def get_labels():
    meta = scipy.io.loadmat(str(path_meta))
    original_idx_to_synset = {}
    synset_to_name = {}

    for i in range(1000):
        ilsvrc2012_id = int(meta["synsets"][i, 0][0][0][0])
        synset = meta["synsets"][i, 0][1][0]
        name = meta["synsets"][i, 0][2][0]
        original_idx_to_synset[ilsvrc2012_id] = synset
        synset_to_name[synset] = name

    synset_to_keras_idx = {}
    keras_idx_to_name = {}
    with open(str(path_synset_words), "r") as f:
        for idx, line in enumerate(f):
            parts = line.split(" ")
            synset_to_keras_idx[parts[0]] = idx
            keras_idx_to_name[idx] = " ".join(parts[1:])

    def convert_original_idx_to_keras_idx(
        idx): return synset_to_keras_idx[original_idx_to_synset[idx]]

    with open(str(val_data_labels), "r") as f:
        y_val = f.read().strip().split("\n")
        y_val = np.array([convert_original_idx_to_keras_idx(int(idx))
                         for idx in y_val])

    return y_val


def disp_img(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

    im = Image.open(path)

    # alexnet preprocessing
    im_prepro = alexnet_preprocess(im)
    im_re = im_prepro.reshape(224, 224, 3)
    plt.imshow(im_re)
    plt.show()

    # resnet preprocessing
    im_prepro = resnet_preprocess(im)
    im_re = im_prepro.reshape(224, 224, 3)
    plt.imshow(im_re)
    plt.show()

    # googlenet preprocessing
    im_prepro = googlenet_preprocess(im)
    im_re = im_prepro.reshape(224, 224, 3)
    plt.imshow(im_re)
    plt.show()


def get_alexnet_img(path):
    img = Image.open(path)
    return alexnet_preprocess(img)


def get_resnet_img(path):
    img = Image.open(path)
    return resnet_preprocess(img)


def get_googlenet_img(path):
    img = Image.open(path)
    return googlenet_preprocess(img)


if __name__ == "__main__":

    # send model to the device
    alexnet_model.to(device)

    # get the paths to all images
    img_paths = get_paths()

    # get the ground truth labels
    y_val = get_labels()

    correct = 0
    miss = 0
    for i in trange(50000):
        img_path = img_paths[i]

        try:
            this_img = get_alexnet_img(img_path)
            this_img.to(device)

            with torch.no_grad():
                logits = alexnet_model(this_img.view(-1,3,224,224))

            preds = torch.nn.functional.softmax(logits, dim=1)
        except:
            miss += 1
            continue

        argmax = int(torch.argmax(preds))

        if argmax == y_val[i]:
            correct += 1    

            # confidence threshold
            if preds[0][argmax] > 0.85:
                print(f"running simba: {preds[0][argmax]}")

                attack = SimBA_Attack(alexnet_model)
                x, pert_history, last_probs, finished = attack.simba(this_img, y_val[i], epsilon=0.8, steps=20000)
                if finished:
                    plt.imshow(pert_history.reshape(224,224,3))
                    plt.show()

        if i % 5000 == 0 and i > 0:
            print(f"Miss: { miss }")
            print(f"Percentage correct {correct / (i - miss)}")

    print(f"Miss: {miss}")
    print(f"Percentage correct {correct / (50000 - miss)}")
