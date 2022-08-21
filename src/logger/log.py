import time
from datetime import datetime

class Log:
    def __init__(self):
        self.init = datetime.now()
        print(f"Logger initialized at {self.init}")

    def epoch(self, epoch, loss, accuracy):
        print('-' * 10)
        print(f"Epoch: {epoch} - Loss: {loss} - Accuracy: {accuracy}", end='\n\n\n')
    
    def finish_train(self, loss, accuracy):
        end_time = datetime.now()
        print('-' * 10)
        print(f"Finished training at {end_time}")
        print(f"Total time: {end_time - self.init}")
        print(f"Average loss: {loss} - Average accuracy: {accuracy}")

    def test(self, loss, accuracy):
        print('-' * 10)
        print(f"Test loss: {loss} - Test accuracy: {accuracy}")

    def save_model(self, epoch, path):
        print('-' * 10)
        print(f"Saving model at epoch {epoch}")
        print(f"Model saved at {datetime.now()}")
        print(f"Model saved at {path}")
        print('-' * 10)
