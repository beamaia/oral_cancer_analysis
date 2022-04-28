import cv2

class OralImage:
    def __init__(self, path: str, macro_class: str, micro_class: str, uuid: str) -> None:
        self.__path = path
        self.__macro_class = macro_class
        self.__micro_class = micro_class
        self.__uuid = uuid
        self.process_image()
        
    def process_image(self) -> None:  
        self.__image = cv2.imread(self.__path)
  
    @property
    def image(self) -> object:
        return self.__image
    
    @property
    def macro_class(self) -> str:
        return self.__macro_class
    
    @property
    def micro_class(self) -> str:
        return self.__micro_class
       
    @property
    def uuid(self) -> str:
        return self.__uuid

    @property
    def path(self) -> str:
        return self.__path

        

    