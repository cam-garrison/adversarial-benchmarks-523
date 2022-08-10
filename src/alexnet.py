"""
    Model gets Acc@1: 56.522, Acc@5: 79.066 w/ 61.1M params
"""
from torchvision.models import alexnet, AlexNet_Weights

# get the pretrained model weights
weights = AlexNet_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

# load the weights into the model
model = alexnet(weights=weights)

print("AlexNet Loaded")
