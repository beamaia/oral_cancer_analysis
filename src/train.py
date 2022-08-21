import time
import torch
import numpy as np

from logger.log import Log

class TrainSetup:
    def __init__(self, model, optimizer, criterion, scheduler, device, name='model', path='models'):  
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device       
        self.name = name
        self.path = f'{path}/{name}.pt'

    def _train_epoch(self):
        torch.cuda.empty_cache()

        # Model in training mode
        self.model.train()
        running_loss = 0
        total_train = 0
        accuracies_train = 0
        
        for _, data in enumerate(self.train_loader):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            #Forward
            outputs = self.model(images)

            loss = self.criterion(outputs, labels).to(self.device) 
            loss.backward()

            #Optimize
            self.optimizer.step()

            running_loss += loss.item()

            #Train accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            accuracies_train += (predicted == labels).sum().item()

        accuracy = accuracies_train / total_train * 100
        running_loss = running_loss / total_train
        self.scheduler.step()
        
        return running_loss, accuracy

    def train(self, train_loader, num_epoch=100):
        self.log = Log()
        
        self.train_loader = train_loader
        
        self.model.train()

        for epoch in range(num_epoch):   
           # Train epoch   
            running_loss, accuracy, y_preds = self.train_epoch()

            # Logs information
            self.log.epoch(epoch, running_loss, accuracy)

            if epoch % 10 == 0:
                self.log.save_model(epoch, self.path)
                torch.save(self.model.state_dict(), self.path)

        # Validation 
        self.log.finish_train(running_loss, accuracy)

        self.log.save_model(epoch, self.path)
        torch.save(self.model.state_dict(), self.path)

        results = self.test()
        
        return results

    def test(self, test_loader):
        self.test_loader = test_loader

        results_dict = {'name': self.name, 
                        'predicted': np.array([], dtype=int), 
                        'real': np.array([], dtype=int)}

        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                
                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)

                results_dict['real'] = np.append(results_dict['real'], labels.flatten().cpu().numpy())
                results_dict['predicted'] = np.append(results_dict['predicted'], predicted.flatten().cpu().numpy())

        return results_dict