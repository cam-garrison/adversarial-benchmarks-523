"""
    Model gets Acc@1: 69.778, Acc@5: 89.53 w/ 6.6M params
"""
from torchvision.models import googlenet, GoogLeNet_Weights

# get model weights
weights = GoogLeNet_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

# load weights into model
model = googlenet(weights=weights)

# show model
print("GoogLeNet Loaded")
