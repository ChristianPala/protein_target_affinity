# Auxiliary file to define the baseline protein target interaction model and train it
# Libraries:
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error


# Classes:
class DrugTargetNET(nn.Module):
    """
    Class to define the DrugTargetNET baseline model
    """
    def __init__(self, input_size: int) -> None:
        """
        Constructor for the DrugTargetNET class
        """
        super(DrugTargetNET, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass of the model
        :param x: the input tensor
        :return: the output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x).squeeze(1)


class ModelTrainer:
    """
    Class to train and evaluate the model
    """
    def __init__(self, model, train_loader, val_loader, lr=0.001) -> None:
        """
        Constructor for the ModelTrainer class
        :param model: the model to train
        :param train_loader: the training data loader
        :param val_loader: the validation data loader
        :param lr: the learning rate
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self, epochs: int) -> None:
        """
        Function to train the model
        :param epochs: the number of epochs to train for
        :return: None
        """
        for epoch in range(epochs):
            for data in self.train_loader:
                self.optimizer.zero_grad()
                inputs, labels = data['combined_features'].float(), data['affinity'].float()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                validation_loss = sum(self.criterion(self.model(data['combined_features'].float()),
                                                     data['affinity'].float()).item() for data in self.val_loader)
            print(f'Epoch: {epoch + 1}, Loss: {validation_loss / len(self.val_loader)}')

    def test(self, test_data) -> None:
        """
        Function to test the model on the test data and print the results
        :param test_data: the test data
        :return: None
        """
        with torch.no_grad():
            X_test = torch.Tensor(list(test_data['combined_features']))
            y_test = torch.Tensor(list(test_data['affinity']))
            outputs = self.model(X_test)
            # save the metrics to a file
            with open('test_metrics.txt', 'w') as f:
                f.write("Baseline\n:")
                f.write(f'RMSE: {np.sqrt(self.criterion(outputs, y_test))}\n')
                f.write(f'MAE: {mean_absolute_error(y_test, outputs)}\n')
                f.write(f'R2: {r2_score(y_test, outputs)}\n')
                f.write(f'Pearson: {pearsonr(y_test, outputs)}\n')

    def save(self, filename: str) -> None:
        """
        Function to save the model
        :param filename: the filename to save the model to
        :return: None
        """
        torch.save(self.model.state_dict(), filename)

    def load(self, filename: str) -> None:
        """
        Function to load the model
        """
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
