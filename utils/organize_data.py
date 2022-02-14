from typing import Tuple
import unidecode
from difflib import SequenceMatcher

#Constants
MICRO_CLASSES_PT = ["com_displasia", "sem_displasia", "outro", "tecido_conjuntivo", "lixo", "carcinoma"]
MICRO_CLASSES_EN = ["with_dysplasia", "no_displasia", "other", "connective_tissue", "trash", "carcinoma"]
MACRO_CLASS_PT = ['carcinoma', 'epitelio_adjacente']
MACRO_CLASS_EN = ['carcinoma', 'adjacent_epithelium']

ROOT_PATH = r"C:\Users\beama\Documents\ufes\labcin\bucal_v1\dataset"

def similar_ratio(original:str, value:str) -> float:
    return SequenceMatcher(None, original, value).ratio()

def similar(original:str, value:str) -> bool:
    return similar_ratio(original, value) >= 0.89

            
def get_english_class(folder:str) -> tuple[bool, str]:
    class_name = unidecode.unidecode(folder.split("\\")[-1]).replace(" ", "_").lower()
    
    if class_name not in MICRO_CLASSES_PT:
        found = False

        for micro_class in MICRO_CLASSES_PT:
            if similar(class_name, micro_class):
                class_name = micro_class
                found = True
        
        if not found:
            if "excluido" or "excluida" in class_name:
                return (True,"other")
            else:
                return (False, class_name)
        
    index = MICRO_CLASSES_PT.index(class_name)
    return (True, MICRO_CLASSES_EN[index])
    
def choose_class_name(value:str) -> str:
    print(f'Class name: {value}')
    print("Values:")
    print(*MICRO_CLASSES_EN, sep=" | ")
    return str(input("Type in correct class: "))

def get_micro_class(folder):
    micro_class = get_english_class(folder)

    if not micro_class[0]:
        return choose_class_name(micro_class[1])
    else:
        return micro_class[1]                