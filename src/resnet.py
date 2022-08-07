"""
    Model gets Acc@1: 80.858, Acc@5: 95.434 w/ 25.6M params
"""
from torchvision.models import resnet50, ResNet50_Weights

# Initialize the Weight Transforms
weights = ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()

# Using pretrained weights:
model = resnet50(weights=weights)

# Print model
print("ResNet Loaded")
