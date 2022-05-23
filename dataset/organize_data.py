from typing import Tuple
import unidecode
from difflib import SequenceMatcher

from const import CLASSES_EN, CLASSES_PT, ROOT_PATH
from ..utils.compare_strings import *

def get_english_class(folder:str) -> tuple[bool, str]:
    # Corrects accents letters
    class_name = unidecode.unidecode(folder.split("\\")[-1]).replace(" ", "_").lower()
    
    if class_name not in CLASSES_PT:
        found = False

        for class_ in CLASSES_PT:
            if similar(class_name, class_):
                class_name = class_
                found = True
        
        if not found:
            return (False, class_name)
        
    return (True, CLASSES_EN[index])
    
def choose_class_name(value:str) -> str:
    print(f'Class name: {value}')
    print("Values:")
    print(*CLASSES_EN, sep=" | ")
    return str(input("Type in correct class: "))

def get_class(folder):
    classes = get_english_class(folder)

    if not classes[0]:
        return choose_class_name(classes[1])
    else:
        return classes[1]                