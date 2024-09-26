from unet import UnetDensenet
from torch import randn, no_grad

unet = UnetDensenet((224, 224, 1))
backbone, a, b = unet.get_backbone()

dummy_input = randn(1, 1, 224, 224)

unet.eval()

with no_grad():
    output = unet(dummy_input)
    
print(output.shape)