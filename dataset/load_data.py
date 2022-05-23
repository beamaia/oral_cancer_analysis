from glob import glob

import cv2

import numpy as np

from dataset.organize_data import *
from const import DATA_WITH_AUG, DATA_NO_AUG

def load_images() -> None:
    """
    Loads oral cancer images. WIP: Depends on dataset structure, not yet defined.
    """
    pass  

def process_images(images, height=224, width=224, interpolation=cv2.INTER_CUBIC):
    print("Processing images...", end="\n\n")
    
    pros_images = np.empty(len(images))
        
    for index, img in enumerate(images):
        full_size_image = cv2.imread(img)
        image = (cv2.resize(full_size_image, (width, height), interpolation = interpolation))
        
        pros_images[index] = image

    return pros_images