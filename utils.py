import torchvision.transforms as transforms
import random
from torchvision.transforms import v2 as T

class MultiRotate(object):
    def __init__(self, angles = [30, 60, 90, 120, 150]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(img, angle)
    
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip())

    transforms.append(T.ToTensor())
    return T.Compose(transforms)