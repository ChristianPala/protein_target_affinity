# Auxiliary file to define the DrugTargetNET model, train and test it on concordance index.
# Libraries:
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index

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

    def forward(self, x):
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

    def train(self, epochs: int) -> None:
        for epoch in range(epochs):
            for data in self.train_loader:
                self.optimizer.zero_grad()
                inputs, labels = data['combined_features'].float().to(device), data['affinity'].float().to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                validation_loss = sum(self.criterion(self.model(data['combined_features'].float().to(device)),
                                                     data['affinity'].float().to(device)).item()
                                      for data in self.val_loader)
            print(f'Epoch: {epoch + 1}, Loss: {validation_loss / len(self.val_loader):.3f}')

    def test(self, test_loader) -> None:
        self.model.eval()
        with torch.no_grad():
            total_ci = 0
            n_batches = 0

            for batch in test_loader:
                X_test = batch['combined_features'].float().to(device)
                y_test = batch['affinity'].float().to(device)

                outputs = self.model(X_test)

                outputs_np = outputs.cpu().numpy()
                y_test_np = y_test.cpu().numpy()

                total_ci += concordance_index(y_test_np, outputs_np.flatten())  # compute CI
                n_batches += 1

            avg_ci = total_ci / n_batches

            with open('test_set_concordance_index.txt', 'w') as f:
                f.write("Baseline results:\n")
                f.write(f'CI: {avg_ci:.3f}\n')

        self.model.train()

    def save(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename)

    def load(self, filename: str) -> None:
        self.model.load_state_dict(torch.load(filename)).to(device)
        self.model.eval()
