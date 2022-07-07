from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np            

class OralDataset(Dataset):
    def __init__(self, images: np.array, labels:np.array, classes_names:list[str], transform) -> None:
        self.labels = labels
        self.images = images
        self.transform = transform
        self.classes = classes_names

    def __len__(self) -> int:
        return len(self.__images)

    def __str__(self) -> str:
        description = f"OralDataset - UFES\nThere are overall {len(self.images)} images (patches)\n\n"
        description += "CLASSES\n"
        for micro_class in self.__classes:
            description += f"{micro_class}: {self.__size_classes[micro_class]}\n"

        return description

    @property
    def classes(self) -> list[str]:
        return self.__classes
    
    @property
    def target(self) -> list[int]:
        return self.__target
    
    @property
    def data(self) -> list[list[float]]:
        return self.__data