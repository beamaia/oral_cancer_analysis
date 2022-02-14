from typing import List
from glob import glob
import matplotlib.pyplot as plt

from ..models.OralImage import OralImage
from ..utils.organize_data import*

class OralDataset:
    def __init__(self, root_path:str, macro_classes: List[str], micro_classes:List[str]) -> None:
        if root_path[-1] == "\\":
            self.__root_path = root_path
        else:
            self.__root_path = f"{root_path}\\"
        self.__macro_classes = macro_classes
        self.__micro_classes = micro_classes
        self.__images = []        

    
    def load_images(self) -> None:
        # Create paths to macro classes' folders
        macro_classes_paths = [f'{ROOT_PATH}\\{macro_class}' for macro_class in MACRO_CLASS_PT]

        for class_i, macro_path in enumerate(macro_classes_paths):

            # Create paths to pacient's folders from macro class' folder
            pacients_path = glob(f'{macro_path}\\*')

            # Goes through each image/pacient folder
            for pacient in pacients_path:
                pacient_uid = pacient.replace(f'{macro_path}\\', "")
                pacient_macro_class = MACRO_CLASS_EN[class_i]
                #get_micro_classes(pacient)

                folders = glob(f'{pacient}\\*')
                for folder in folders:
                    micro_class = get_micro_class(folder)
                                            
                    images =  glob(f'{folder}\\*')
                    for aux in images:
                        # found a folder, not image

                        if aux[-3:] != "png":
                            micro_class = get_micro_class(aux)
                            images_aux = glob(f'{aux}\\*')
                    
                            for other_image in images_aux:
                                self.__images.append(OralImage(other_image, pacient_macro_class, micro_class, pacient_uid))
                        else:
                            print(aux)
                            self.__images.append(OralImage(aux, pacient_macro_class, micro_class, pacient_uid))
            
                                
                
    def show_image(self, index: int) -> None:
        print(len(self.__images))
        if 0 <= index < len(self.__images):
            plt.imshow(self.__images[index].image)
            plt.show()
        else:
            print("Index out of bound")
        
    def __len__(self) -> int:
        return len(self.__images)

    @property
    def micro_classes(self) -> List[str]:
        return self.__micro_classes

if __name__ == "__main__":
    images_v1 = OralDataset(r"C:\Users\\beama\Documents\ufes\labcin\\bucal_v1",MACRO_CLASS_EN, MICRO_CLASSES_EN)
    images_v1.load_images()
    print(f'Size: {len(images_v1)}')
    images_v1.show_image(2)