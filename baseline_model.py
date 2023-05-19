# Libraries
import numpy as np
import pandas as pd
from datasets import load_dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from typing import List


# Functions:
def smiles_to_fp(smiles, radius=2, n_bits=1024) -> np.array:
    """
    Function to convert a SMILES string into a Morgan fingerprint
    @param smiles: str:  The SMILES string
    @param radius: int: The radius of the fingerprint, how many atoms away from the central atom are considered
    @param n_bits: int: The length of the fingerprint
    :return: np.array: The fingerprint as a numpy array
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    np_fp = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, np_fp)
    return np_fp


def protein_to_encoding(protein_seq) -> List[int]:
    """
    Function to convert a protein sequence into a vector of integers
    @param protein_seq: str: The protein sequence
    :return: list: The protein sequence encoded as a list of integers
    """
    amino_acid_dict = {amino_acid: i for i, amino_acid in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    protein_encoded = [amino_acid_dict[aa] for aa in protein_seq]
    return protein_encoded


def normalize_affinity(x) -> pd.Series:
    """
    Function to normalize the affinity values
    @param x: pd.Series: The dictionary containing the affinity value
    :return: pd.Series: The dictionary with the normalized affinity value
    """
    mean = train_data['affinity'].mean()
    std = train_data['affinity'].std()
    x['affinity'] = (x['affinity'] - mean) / std
    return x


def combine_features(x) -> pd.DataFrame:
    """
    Function to combine the encoded protein sequence and the fingerprint into a single feature vector
    :param x: pd.DataFrame: The dataframe containing the fingerprint and the encoded protein sequence
    :return: pd.DataFrame: The dataframe containing the combined feature vector
    """
    x['combined_features'] = np.concatenate((x['smiles_fp'], x['protein_encoded']))
    return x


class DrugTargetDataset(Dataset):
    """
    Class to create a PyTorch Dataset from the dataset created with the combine_features function above,
    which contains the combined feature vectors and the normalized affinity values
    """
    def __init__(self, data) -> None:
        """
        Constructor
        @param data: pd.DataFrame: The dataframe containing the combined feature vectors and the normalized affinity values
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles_fp = smiles_to_fp(self.data[idx]['smiles'])
        protein_encoded = protein_to_encoding(self.data[idx]['seq'])
        affinity = normalize_affinity(self.data[idx])
        combined_features = combine_features({'smiles_fp': smiles_fp, 'protein_encoded': protein_encoded})
        return combined_features, affinity


class DrugTargetNET(nn.Module):
    def __init__(self):
        super(DrugTargetNET, self).__init__()
        self.fc1 = nn.Linear(1124, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


if __name__ == '__main__':

    ###################################################################
    # Data loading
    train_data = load_dataset("jglaser/binding_affinity", split='train[:90%]')
    test_data = load_dataset("jglaser/binding_affinity", split='train[90%:]')
    ###################################################################
    # Preprocessing
    # Morgan Fingerprint
    train_data = train_data.map(lambda x: {'smiles_fp': smiles_to_fp(x['smiles'])})
    validation_data = test_data.map(lambda x: {'smiles_fp': smiles_to_fp(x['smiles'])})
    # Normalize the affinity
    train_data = train_data.map(normalize_affinity)
    validation_data = validation_data.map(normalize_affinity)
    # Encode the protein sequence
    train_data = train_data.map(lambda x: {'protein_encoded': protein_to_encoding(x['seq'])})
    validation_data = validation_data.map(lambda x: {'protein_encoded': protein_to_encoding(x['seq'])})
    # Apply the function to create combined feature vectors
    train_data = train_data.map(combine_features)
    validation_data = validation_data.map(combine_features)
    ###################################################################
    # Modeling
    model = DrugTargetNET()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Convert dataset to PyTorch DataLoader
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=False)
    ###################################################################
    # Training loop
    epochs = 10

    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['smiles_fp'], data['affinity']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
        running_loss = 0.0
        for i, data in enumerate(validation_dataloader, 0):
            inputs, labels = data['smiles_fp'], data['affinity']
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        print(f'Epoch: {epoch + 1}, Loss: {running_loss / len(validation_dataloader)}')

    print('Finished Training')
    ###################################################################
    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    ###################################################################
    # Load the model
    model = DrugTargetNET()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    ###################################################################
    # Test the model
    # Convert dataframe to tensor
    X_test = torch.Tensor(list(test_data['smiles_fp']))
    y_test = torch.Tensor(list(test_data['affinity']))

    with torch.no_grad():
        test_loss = 0.0
        for i in range(len(X_test)):
            outputs = model(X_test[i])
            test_loss += criterion(outputs, y_test[i]).item()
        print(f'Test loss: {test_loss / len(X_test)}')

    print('Finished Testing')
    ###################################################################
