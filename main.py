from typing import List
from glob import glob
import matplotlib.pyplot as plt

from src.models import *
from src.const import *
from src.train import TrainSetup

import torchvision
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold

def main():
    # path to data
    path_data = '/home/bea/oral_cancer_analysis/data/'

    # dataset transforms
    train_transforms=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomApply([transforms.RandomRotation(10)], 0.25),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    train_dataset = torchvision.datasets.ImageFolder(root=path_data + 'train/', transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    skf = StratifiedKFold(n_splits=5, random_state=42,shuffle=True)

    results = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i, (train_index, test_index) in enumerate(skf.split( range(len(train_dataset)), train_dataset.targets)):
        print('Fold: ', i)
        print('Initializing model and parameters...')
        
        model = VGG16()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        print('Separating train and validation data...')
        dataset_train = Subset(train_dataset,train_index)
        dataset_test = Subset(train_dataset,test_index)
        train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=16, shuffle=True)

        print('Creating task pool...')
        
        print(f'Training Fold: {i}')

        train = TrainSetup(model, optimizer, criterion, lr_scheduler, device, name=f'vgg16_{i}')
        train.train(train_loader, num_epoch=200)
        
        print(f'Testing Fold: {i}')
        results_dict = train.test(test_loader)
        
        results.append(results_dict)

if __name__ == "__main__":
    main()