from typing import List
from glob import glob
import matplotlib.pyplot as plt

from models.OralImage import OralImage
from dataset.OralDataset import OralDataset
from utils.organize_data import *

if __name__ == "__main__":
    path = "C:/Users/beama/Documents/ufes/labcin/data/oral_cancer"
    print(f"Path: {path}")
    images_v1 = OralDataset(root_path=path, macro_classes=MACRO_CLASS_EN, micro_classes=MICRO_CLASSES_EN)
    images_v1.load_images()
    print(images_v1)

    df = images_v1.dataframe
    classes = df[['macro_class', 'micro_class']]
    classes_distr = classes.value_counts()
    classes_distr.to_csv("classes_distribution.csv")