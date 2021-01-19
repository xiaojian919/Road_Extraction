
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os

# h:180-180  s:-255-255 v: -255-255


def randomHueSaturationValue(image, hue_shift_limit=(-15, 15),
                             sat_shift_limit=(-15, 15),
                             val_shift_limit=(-30, 30), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


if __name__ == '__main__':
    img = cv2.imread("../1.jpg")
    img = randomHueSaturationValue(img)
    cv2.imwrite("2.jpg", img)







