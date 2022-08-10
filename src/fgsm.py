import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class FGSM_Attack(object):
    def __init__(self, model=None, epsilon=0.02):
        self.model = model
        self.epsilon = epsilon
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def perturb_image(self, image, image_grad):
        """
        actually attack image based on info found in fgsm()
        based on torch tutorial function 'fgsm_attack'
        """
        # find the sign of the gradient
        sign_img_grad = image_grad.sign()
        pert_img = image + self.epsilon * sign_img_grad  # change img
        return pert_img, torch.linalg.norm((self.epsilon * sign_img_grad))

    def fgsm(self, image, label):

        image, label = image.to(self.device), label.to(self.device)
        image.requires_grad = True

        label = Variable(torch.LongTensor([label.item()]), requires_grad=False)

        # get initial pred
        logits = self.model(image.view(-1, 3, 224, 224))
        preds = nn.functional.softmax(logits, dim=1)
        argmax = int(torch.argmax(preds))
        if argmax != label:
            return "shouldnt run, init is wrong"

        # get the loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, label)

        loss.backward(retain_graph=True)
        image_gradient = image.grad.data

        # call perturbing method
        return self.perturb_image(image, image_gradient)
