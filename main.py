import copy
import argparse

import matplotlib.pyplot as plt

from src.models import *
from src.const import *
from src.train import TrainSetup

import torchvision
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd
import numpy as np

def main(model_aux, epoch, batch_size, lr, momentum):
    # path to data
    path_data = '/home/bea/oral_cancer_analysis/data/'

    # dataset transforms
    train_transforms=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomApply([transforms.RandomRotation(10)], 0.25),
                # transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    train_dataset = torchvision.datasets.ImageFolder(root=path_data + 'train/', transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=5, random_state=42,shuffle=True)

    for i, (train_index, test_index) in enumerate(skf.split( range(len(train_dataset)), train_dataset.targets)):
        print('Fold: ', i)
        print('Initializing model and parameters...')
        
        model = copy.deepcopy(model_aux)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        weights = compute_class_weight('balanced', np.unique(train_dataset.targets), train_dataset.targets)

        criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        print('Separating train and validation data...')
        dataset_train = Subset(train_dataset,train_index)
        dataset_test = Subset(train_dataset,test_index)
        train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=16, shuffle=True)

        print('Creating task pool...')
        
        print(f'Training Fold: {i}')
        train = TrainSetup(model, optimizer, criterion, lr_scheduler, device, name=f'vgg16_{i}')
        train.train(train_loader, num_epoch=epoch)
        
        print(f'Testing Fold: {i}')
        results_dict = train.test(test_loader)

        # Saving y values (predicted and true)
        df = pd.DataFrame(results_dict)
        df.to_csv(f'{path_data}results/{model}_{i}_yvalue.csv', columns=['y_pred', 'y_true'], index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validation for Oral Cancer Analysis",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", help="especify model that will be used")
    parser.add_argument("-e", "--epoch", default=200, help="especify number of epochs (default 200)")
    parser.add_argument("-p", "--path", help="especify dataset path")
    parser.add_argument("-b", "--batch", default=8, help="especify batch size (default 8)")
    parser.add_argument("-l", "--lr", default=0.001, help="especify learning rate (default 0.001)")
    parser.add_argument("-m", "--momentum", default=0.9, help="especify momentum (default 0.9)")
    parser.add_argument("-s", "--step", default=20, help="especify step size (default 20)")

    args = parser.parse_args()
    config = vars(args)

    if config['model'] == 'vgg16':
        model = VGG16()
    elif config['model'] == 'resnet50':
        model = ResNet50()
    elif config['model'] == 'densenet121':
        model = DenseNet121()
    elif config['model'] == 'mobilenetv2':
        model = MobileNetv2()
    elif config['model'] == 'efficientnetb4':
        model = EfficientNetB4()
    
    main(model_aux=model, epoch=config['epoch'], batch_size=config['batch'], lr=config['lr'], momentum=config['momentum'])