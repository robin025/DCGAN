import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def label_smoothing(labels, smoothing=0.1):
    # Make sure the random tensor is on the correct device
    rand_tensor = torch.rand(labels.size()).to(labels.device)
    return (1.0 - smoothing) * labels + smoothing * rand_tensor
