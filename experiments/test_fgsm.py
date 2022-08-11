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

from src.alexnet import model as alexnet_model, preprocess as alexnet_preprocess
from data.labels import labels, path_meta, path_synset_words
from src.fgsm import FGSM_Attack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device used: {device}")

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


# i/o helpers
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


def get_alexnet_img(path):
    img = Image.open(path)
    return alexnet_preprocess(img)


if __name__ == "__main__":
    # send model to the device
    alexnet_model.to(device)
    alexnet_model.eval()

    # get the paths to all images
    img_paths = PATHS

    # get the ground truth labels
    y_val = np.array(LABELS, dtype=int)
    y_val = torch.from_numpy(y_val)

    img_path = img_paths[42]

    disp_img(img_path)

    this_img = get_alexnet_img(img_path)
    this_img.to(device)

    fgsm_att = FGSM_Attack(alexnet_model, epsilon=0.02)

    logits = alexnet_model(this_img.view(-1, 3, 224, 224))
    preds = torch.nn.functional.softmax(logits, dim=1)
    argmax = int(torch.argmax(preds))
    if argmax == y_val[42]:
        print("correct pred\n")

    print("pred class", labels[y_val[42].item()], "with conf", preds[0][argmax])

    # confidence threshold
    if preds[0][argmax] > 0.50:
        print("passed conf check, do attack")

        perted_img, norm = fgsm_att.fgsm(this_img, y_val[42])
        print("norm is", norm)
        perted_logits = alexnet_model(perted_img.view(-1, 3, 224, 224))
        perted_preds = torch.nn.functional.softmax(perted_logits, dim=1)
        perted_argmax = int(torch.argmax(perted_preds))
        if perted_argmax == y_val[42]:
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
