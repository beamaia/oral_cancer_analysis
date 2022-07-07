import pathlib
import shutil

from typing import Tuple
import unidecode
from difflib import SequenceMatcher

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.dataset_functions.const import CLASSES_EN, CLASSES_PT, CLASSES_DICTIONARY
from src.utils.compare_strings import *

def get_english_class(folder:str) -> tuple[bool, str]:
    # Corrects accents letters
    class_name = unidecode.unidecode(folder.split("\\")[-1]).replace(" ", "_").lower()
    found = False
    
    if class_name not in CLASSES_PT:

        for class_ in CLASSES_PT:
            if similar(class_name, class_, 0.85):
                class_name = class_
                found = True
                break
        
        if not found:
            print(class_name)
            return (False, class_name)
        
    else:
        found = True
    
    class_name = CLASSES_EN[CLASSES_PT.index(class_name)]        
    return (found, class_name)
    
def choose_class_name(value:str) -> str:
    print(f'Class name: {value}')
    print("Values:")
    print(*CLASSES_EN, sep=" | ")
    return str(input("Type in correct class: "))

def get_class(folder):
    classes = get_english_class(folder)
    return classes[1]       
        
def save_image(image_name:str, class_name:str, origin_path, destiny_path:str) -> None:
    if not pathlib.Path(f"{destiny_path}/{class_name}").exists():
        pathlib.Path(f"{destiny_path}/{class_name}").mkdir(parents=True)
        print("Folder created:", class_name)  

    if not pathlib.Path(f"{destiny_path}/{class_name}/{image_name}").exists():
        shutil.copy(origin_path, f"{destiny_path}/{class_name}/{image_name}")
        print("Image copied:", image_name)

def iterate_images(origin_path:pathlib.PosixPath, destiny_path:pathlib.PosixPath) -> None:
    # Iterates over folder
    for folder in origin_path.iterdir():
        if folder.is_dir():
            iterate_images(folder, pathlib.Path(destiny_path))
        elif folder.is_file():
            folder_name = folder.parts[-2]
            folder_name_corrected = get_class(folder_name)
            save_image(folder.name, folder_name_corrected, folder, destiny_path)
            
def calculate_amount_of_images(path:pathlib.PosixPath) -> pd.DataFrame:
    """
    Calculates amount of images in a folder.
    """
    df = pd.DataFrame(columns=["class", "amount"])
    for folder in path.iterdir():
        if folder.is_dir():
            dic_row = {"class": folder.name, "amount": len(glob(f"{folder}/*"))}
            df = df.append(dic_row, ignore_index=True)
    
    return df

def create_X_y(path:pathlib.PosixPath) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates X and y arrays.
    """
    X = []
    y = []

    for folder in path.iterdir():
        if folder.is_dir() and folder.name in CLASSES_DICTIONARY.keys():
            print(folder.name)
            for image in folder.iterdir():
                if image.is_file():
                    X.append(image)
                    y.append(CLASSES_DICTIONARY[folder.name])

    return np.array(X), np.array(y)

def divide_train_test(path:pathlib.PosixPath) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    X_data, y_data = create_X_y(pathlib.Path(path))
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42, stratify=y_data)
    return (X_train, y_train), (X_test, y_test)

def main_count_images() -> None:
    destiny_path = pathlib.Path('/home/bea/oral_cancer_analysis/data_divided').resolve()
    df = calculate_amount_of_images(destiny_path)
    df = df.sort_values(by=["amount"], ascending=False)
    df.to_csv("/home/bea/oral_cancer_analysis/data_divided/amount_of_images.csv", index=False)

def main_divide_images() -> None:
    origin_path = pathlib.Path('/home/bea/oral_cancer_analysis/data_old').resolve()
    print(type(origin_path))
    destiny_path = pathlib.Path('/home/bea/oral_cancer_analysis/data_divided').resolve()

    if not destiny_path.exists():
        destiny_path.mkdir()

    iterate_images(origin_path, destiny_path)

def main_count_train_test():
    # main_divide_images()
    path = '/home/bea/oral_cancer_analysis/data_divided'
    (X_train, y_train), (X_test, y_test) = divide_train_test(path)
    print(X_train.shape, y_train.shape)
    print(np.unique(y_train, return_counts=True))
    print(X_test.shape, y_test.shape)
    print(np.unique(y_test, return_counts=True))

if __name__ == "__main__":
    main_count_train_test()