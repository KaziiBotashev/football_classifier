import torchvision.transforms.functional as F
import numpy as np


class SquarePad:
    def __call__(self, image):
        #Make square image with a side equal to the bigger side of input image
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')
