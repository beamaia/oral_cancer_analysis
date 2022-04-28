from pydoc import describe
from typing import List
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

from models.OralImage import OralImage
from utils.organize_data import *

class OralDataset:
    def __init__(self, root_path:str, macro_classes: List[str], micro_classes:List[str]) -> None:
        if root_path[-1] == "\\":
            self.__root_path = root_path.replace("\\", "/")
        else:
            self.__root_path = f"{root_path}/"
        self.__macro_classes = macro_classes
        self.__micro_classes = micro_classes
        self.__size_macro_classes = {key: 0 for key in self.__macro_classes}
        self.__size_micro_classes = {key: 0 for key in self.__micro_classes}
        self.__images = []
    
    def load_images(self) -> None:
        # Create paths to macro classes' folders
        macro_classes_paths = [f'{ROOT_PATH}/{macro_class}' for macro_class in MACRO_CLASS_PT]

        for class_i, macro_path in enumerate(macro_classes_paths):
            # Create paths to pacient's folders from macro class' folder
            pacients_path = glob(f'{macro_path}/*')
            pacient_macro_class = MACRO_CLASS_EN[class_i]
            # Goes through each image/pacient folder
            for pacient in pacients_path:
                pacient_uid = pacient.replace(f'{macro_path}/', "")
                self.__size_macro_classes[pacient_macro_class] += 1
                #get_micro_classes(pacient)

                folders = glob(f'{pacient}/*')
                for folder in folders:
                    micro_class = get_micro_class(folder)

                    images =  glob(f'{folder}/*')

                    for aux in images:
                        self.__size_macro_classes[pacient_macro_class] += 1
                        self.__size_micro_classes[micro_class] += 1
                        self.__images.append(OralImage(aux, pacient_macro_class, micro_class, pacient_uid))           
                                
                
    def show_image(self, index: int) -> None:
        if 0 <= index < len(self.__images):
            plt.imshow(self.__images[index].image)
            plt.show()
        else:
            print("Index out of bound")

    def __len__(self) -> int:
        return len(self.__images)

    def __str__(self) -> str:
        description = f"OralDataset - UFES\nThere are overall {len(self.__images)} images (patches)\n\n"
        description += f"MACRO CLASSES\n{self.__macro_classes[0]}: {self.__size_macro_classes[self.__macro_classes[0]]}\n"
        description += f"{self.__macro_classes[1]}: {self.__size_macro_classes[self.__macro_classes[1]]}\n\n"

        description += "MICRO CLASSES\n"
        for micro_class in self.__micro_classes:
            description += f"{micro_class}: {self.__size_micro_classes[micro_class]}\n"

        return description

    @property
    def micro_classes(self) -> List[str]:
        return self.__micro_classes
    
    @property
    def dataframe(self) -> pd.core.frame.DataFrame:
        columns = ['path', 'macro_class', 'micro_class', 'image_class']
        df = pd.DataFrame(columns=columns)

        rows = []
        for i, image in enumerate(self.__images):
            aux = {'path':image.path, 
                    'macro_class': image.macro_class,
                    'micro_class': image.micro_class,
                    'image_class': image}
            
            row = pd.DataFrame(aux, index=[i])
            df = pd.concat([df, row])#.reset_index()

        return df