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
CLASSES_PT = ["com_displasia", "sem_displasia", "carcinoma", "branco", "epitelio_superficial", "fora_foco", "outro", "tecido_conjuntivo", "coloracao_ruim"]
CLASSES_EN = ["with_dysplasia", "no_dysplasia", "carcinoma", "blank", "superficial_epithelium", "out_of_focus", "other", "connective_tissue", "bad_color"]

# Images root path
ROOT_PATH = "data_old"

# Classes dictionary
CLASSES_DICTIONARY = {'with_dysplasia': 0, 
           'no_dysplasia': 1, 
           'connective_tissue': 2}
