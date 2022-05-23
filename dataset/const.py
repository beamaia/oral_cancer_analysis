import torchvision.transforms as transforms

############### DATA AUGMENTATION
DATA_WITH_AUG = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ])
}

############### NO DATA AUGMENTATION

DATA_NO_AUG = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ])
}

# Classes strings
CLASSES_PT = ["com_displasia", "sem_displasia", "carcinoma"]
CLASSES_EN = ["with_dysplasia", "no_dysplasia", "carcinoma"]

# Images root path
ROOT_PATH = ""
