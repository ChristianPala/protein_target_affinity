# Auxiliary file to define the DrugTargetNET model, train and test it on concordance index.
# Libraries:
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

# Constants:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DrugTargetNET(nn.Module):
    """
    DrugTargetNET model, which takes as input the concatenation of the Morgan fingerprint and the protein encoding
    and outputs the affinity, this part of the architecture was copied by the DeepDTA model.
    """
    def __init__(self, smiles_encoding: int, protein_encoding: int, dropout_p: float = 0.1) -> None:
        super(DrugTargetNET, self).__init__()
        self.fc1 = nn.Linear(smiles_encoding + protein_encoding, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout_p)


    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.train_losses = []
        self.val_losses = []

    def train(self, epochs: int) -> None:
        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0

            for data in self.train_loader:
                self.optimizer.zero_grad()
                inputs, labels = data['combined_features'].float().to(device), data['affinity'].float().to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            with torch.no_grad():
                for data in self.val_loader:
                    val_loss += self.criterion(self.model(data['combined_features'].float().to(device)),
                                               data['affinity'].float().to(device)).item()

            self.train_losses.append(train_loss / len(self.train_loader))
            self.val_losses.append(val_loss / len(self.val_loader))

            print(
                f'Epoch: {epoch + 1}, Train Loss: {self.train_losses[-1]:.3f}, '
                f'Validation Loss: {self.val_losses[-1]:.3f}')

    def test(self, test_loader) -> None:
        """
        Test the model on the test set and write the results to a file on the metrics we selected.
        :param test_loader: the test set loader
        :return: None. Writes the results to a file.
        """
        self.model.eval()
        total_ci = 0
        total_mse = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in test_loader:
                X_test = batch['combined_features'].float().to(device)
                y_test = batch['affinity'].float().to(device)

                outputs = self.model(X_test)

                outputs_np = outputs.cpu().numpy()
                y_test_np = y_test.cpu().numpy()

                total_ci += concordance_index(y_test_np, outputs_np.flatten())  # compute CI

                mse = self.criterion(outputs, y_test).item()
                total_mse += mse
                predictions.append(outputs_np.flatten())
                targets.append(y_test_np)

        # Compute averages
        avg_ci = total_ci / len(test_loader)
        avg_mse = total_mse / len(test_loader)

        # Compute Pearson Correlation
        pearson_corr, _ = pearsonr(np.concatenate(predictions), np.concatenate(targets))

        # Write to file
        with open('test_results.txt', 'w') as f:
            f.write("Test results:\n")
            f.write(f'CI: {avg_ci:.3f}\n')
            f.write(f"MSE: {avg_mse:.3f}\n")
            f.write(f"Pearson: {pearson_corr:.3f}\n")

        # Print to screen
        print("Test results:")
        print(f'CI: {avg_ci:.3f}')
        print(f"MSE: {avg_mse:.3f}")
        print(f"Pearson: {pearson_corr:.3f}")

        self.model.train()


    def save(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename)

    def load(self, filename: str) -> None:
        self.model.load_state_dict(torch.load(filename))
        self.model.to(device)

    def plot_losses(self) -> None:
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Train and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        sns.set_style("whitegrid")
        plt.legend()
        plt.show()

    def train_cnn(self, epochs: int) -> None:
        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0

            for data in self.train_loader:
                self.optimizer.zero_grad()
                x_smiles, x_protein, labels = data['smiles_fp'].float().to(device),  \
                    data['protein_encoded'].float().to(device), data['affinity'].float().to(device)
                outputs = self.model(x_smiles, x_protein)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            with torch.no_grad():
                for data in self.val_loader:
                    val_x_smiles, val_x_protein, val_labels = data['smiles_fp'].float().to(device), \
                        data['protein_encoded'].float().to(device), data['affinity'].float().to(device)

                    val_loss += self.criterion(self.model(val_x_smiles, val_x_protein), val_labels).item()

            self.train_losses.append(train_loss / len(self.train_loader))
            self.val_losses.append(val_loss / len(self.val_loader))

            print(
                f'Epoch: {epoch + 1}, Train Loss: {self.train_losses[-1]:.3f}, '
                f'Validation Loss: {self.val_losses[-1]:.3f}')

    def test_cnn(self, test_loader) -> None:
        self.model.eval()
        total_ci = 0
        total_mse = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in test_loader:
                x_smiles, x_protein, y_test = batch['smiles_fp'].float().to(device), \
                    batch['protein_encoded'].float().to(device), batch['affinity'].float().to(device)
                outputs = self.model(x_smiles, x_protein)

                outputs_np = outputs.cpu().numpy()
                y_test_np = y_test.cpu().numpy()

                total_ci += concordance_index(y_test_np, outputs_np.flatten())  # compute CI

                mse = self.criterion(outputs, y_test).item()
                total_mse += mse
                predictions.append(outputs_np.flatten())
                targets.append(y_test_np)

                # Compute averages
            avg_ci = total_ci / len(test_loader)
            avg_mse = total_mse / len(test_loader)

            # Compute Pearson Correlation
            pearson_corr, _ = pearsonr(np.concatenate(predictions), np.concatenate(targets))

            # Write to file
            with open('cnn_test_results.txt', 'w') as f:
                f.write("Test results with CNN encoders:\n")
                f.write(f'CI: {avg_ci:.3f}\n')
                f.write(f"MSE: {avg_mse:.3f}\n")
                f.write(f"Pearson: {pearson_corr:.3f}\n")

            # Print to screen
            print("Test results:")
            print(f'CI: {avg_ci:.3f}')
            print(f"MSE: {avg_mse:.3f}")
            print(f"Pearson: {pearson_corr:.3f}")

            self.model.train()
