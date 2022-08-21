from glob import glob
import pathlib
import shutil

import unidecode

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.const import CLASSES_EN, CLASSES_PT, CLASSES_DICTIONARY, REVERSE_CLASSES_DICTIONARY
from src.utils.compare_strings import *

def get_english_class(folder:str) -> tuple[bool, str]:
    '''
    The original data contains folders with names in Portuguese. This function tries to find 
    the correct English name for the folder.
    '''
    # Corrects accents letters and replaces spaces with underscores
    class_name = unidecode.unidecode(folder.split("\\")[-1]).replace(" ", "_").lower()
    found = False
    
    # The class name doesn't exist in the known classes or is mistyped
    if class_name not in CLASSES_PT:
        class_name = class_name
        best_ratio = 0
        for class_ in CLASSES_PT:
            # Checks which class is the most similar
            if similar(class_name, class_, 0.85) > best_ratio:
                best_ratio = similar(class_name, class_, 0.85)
                class_name = class_
                found = True
                
        # If no classes were found, returns the original name
        if not found:
            return (False, class_name)
    else:
        found = True
    
    # If the class name is found, returns the correct name
    class_name = CLASSES_EN[CLASSES_PT.index(class_name)]        
    return (found, class_name)
    
def choose_class_name(value:str) -> str:
    '''
    Function used to choose the class name.
    '''
    print(f'Class name: {value}')
    print("Values:")
    print(*CLASSES_EN, sep=" | ")
    return str(input("Type in correct class: "))

def get_class(folder):
    '''
    Function used to get the class name.
    '''
    classes = get_english_class(folder)

    # currently, using the class_name independent if it was found or not
    return classes[1]       
        
def save_image(image_name:str, class_name:str, origin_path, destiny_path:str) -> None:
    '''
    Copies image from original path to class folder. 
    '''
    class_path = destiny_path / class_name
    if not pathlib.Path(class_path).exists():
        pathlib.Path(class_path).mkdir(parents=True)
        print("Folder created:", class_name)  

    full_destiny_path = class_path / image_name
    if not pathlib.Path(full_destiny_path).exists():
        shutil.copy(origin_path, full_destiny_path)
        # print("Image copied:", image_name)

def iterate_images(origin_path:pathlib.PosixPath, destiny_path:pathlib.PosixPath) -> None:
    '''
    Saves images in the correct class folder. If the directory visited has files, than the
    class_name is the directory name. If the directory visited has subdirectories, than the
    function calls itself recursively and checks it's child files and subdirectories.
    '''
    # Iterates over folder
    for folder in origin_path.iterdir():
        if folder.is_dir():
            iterate_images(folder, pathlib.Path(destiny_path))
        elif folder.is_file():
            folder_name = folder.parts[-2]
            folder_name_corrected = get_class(folder_name)
            # save_image(folder.name, folder_name_corrected, folder, destiny_path)
            save_image(folder.name, folder_name, folder, destiny_path)


def calculate_amount_of_images(path:pathlib.PosixPath) -> pd.DataFrame:
    """
    Calculates amount of images in a folder and returns a DataFrame with the
    columns *class* (for directory name) and *amount* for the amount of images
    in said class.
    """
    df = pd.DataFrame(columns=["class", "amount"])
    for folder in path.iterdir():
        if folder.is_dir():
            dic_row = {"class": folder.name, "amount": len(glob(f"{folder}/*"))}
            df = df.append(dic_row, ignore_index=True)
    
    return df

def create_X_y(path:pathlib.PosixPath) -> tuple[np.ndarray, np.ndarray]:
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

def divide_train_test(X_data:np.array, y_data:np.array) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Divides the data into train and test sets. Returns X_train, y_train), (X_test, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42, stratify=y_data)
    return (X_train, y_train), (X_test, y_test)

def main_count_images() -> None:
    '''
    Counts the amount of images in each class and logs information into a csv.
    '''
    destiny_path = pathlib.Path('/home/bea/oral_cancer_analysis/drop_box_data_divided').resolve() 
    
    for folder in destiny_path.iterdir():
        if folder.is_dir(): 
            
            df = calculate_amount_of_images(folder)
            df = df.sort_values(by=["amount"], ascending=False)
            df.to_csv(f"{destiny_path}/{folder.name}/amount_of_images_{folder.name}.csv", index=False)

def main_divide_images() -> None:
    '''
    Organizes initial data unorganized data strucutre into datset divided by classes.
    Initial data structure includes folders by pacient hash, with a possible unknown 
    amount of nested subdirectories and images inside subdirectories. The output is a
    dataset that has only 1 level of subdirectories and images inside said subdirectories.
    Ex.:
    /root
        /class_name
                  /image_name.png
    '''
    origin_path = pathlib.Path('/home/bea/oral_cancer_analysis/Dropbox').resolve()
    destiny_path = pathlib.Path('/home/bea/oral_cancer_analysis/drop_box_data_divided').resolve()

    if not destiny_path.exists():
        print('hi')
        destiny_path.mkdir()

    for folder in origin_path.iterdir():
        if folder.is_dir():
            destiny_path_aux = pathlib.Path(destiny_path / folder.name)

            if not destiny_path_aux.exists():
                destiny_path_aux.mkdir()
                print("Folder created:", destiny_path_aux)
            else:
                print(destiny_path_aux)
                continue
            iterate_images(folder, destiny_path_aux)

def main_train_test():
    '''
    Saves to disk the train and test sets.
    '''
    path = pathlib.Path('data_divided').resolve()

    X_data = np.array([])
    y_data = np.array([])

    for folder in path.iterdir():
        if folder.is_dir():
            X_data_aux, y_data_aux = create_X_y(folder)
            X_data = np.append(X_data, X_data_aux)
            y_data = np.append(y_data, y_data_aux)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42, stratify=y_data)
    
    output_path = pathlib.Path('/home/bea/oral_cancer_analysis/data').resolve()
    train_path = output_path / "train"
    test_path = output_path / "test"
    
    if not output_path.exists():
        output_path.mkdir()
    if not train_path.exists():
        train_path.mkdir()
    if not test_path.exists():
        test_path.mkdir()

    for (x, y) in zip(X_train, y_train):
        class_path = train_path / REVERSE_CLASSES_DICTIONARY[y]
        if not class_path.exists():
            class_path.mkdir()
        
        if not pathlib.Path(f"{class_path}/{x.name}").exists():
            shutil.copy(x, f"{class_path}/{x.name}")
            print(f"Image copied: {x.name}")

    for (x, y) in zip(X_test, y_test):
        class_path = test_path / REVERSE_CLASSES_DICTIONARY[y]
        if not class_path.exists():
            class_path.mkdir()
        
        if not pathlib.Path(f"{class_path}/{x.name}").exists():
            shutil.copy(x, f"{class_path}/{x.name}")
            print(f"Image copied: {x.name}")
    
    print("Train set:", len(X_train))
    print("Test set:", len(X_test))


def main_count_train_test():
    output_path = pathlib.Path('/home/bea/oral_cancer_analysis/data').resolve()
    train_path = output_path / "train"
    test_path = output_path / "test"

    df = pd.DataFrame(columns=["class", 'set', "amount"])

    count = 0
    for folder in train_path.iterdir():
        if folder.is_dir():
            df_aux = pd.DataFrame({"class": folder.name, 
                                   "set": "train", 
                                   "amount": len(glob(f"{folder}/*"))}, index=[count])
            df = df.append(df_aux)
            count += 1

    for folder in test_path.iterdir():
        if folder.is_dir():
            df_aux = pd.DataFrame({"class": folder.name, 
                                   "set": "test", 
                                   "amount": len(glob(f"{folder}/*"))}, index=[count])
            df = df.append(df_aux)
            count += 1
    
    df = df.sort_values(by=["class", "set"], ascending=True)
    df.to_csv("/home/bea/oral_cancer_analysis/data/amount_of_images.csv", index=False)

if __name__ == "__main__":
    # main_divide_images()
    # main_count_images()
    # main_train_test()
    main_count_train_test()