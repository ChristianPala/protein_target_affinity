# Library to encode SMILES and protein sequences using CNN models
import torch
from torch import nn
from dti_model import DrugTargetNET


class CNNEncoderModel(nn.Module):
    def __init__(self, smiles_encoding: int, protein_encoding: int, dropout_p: float = 0.1):
        super(CNNEncoderModel, self).__init__()
        self.cnn_smiles = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # we reduce the size of the input.
            # further layers...
        )
        self.cnn_protein = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # further layers...
        )
        # Initialize your DrugTargetNET with the desired final number of dimensions.
        self.drug_target_net = DrugTargetNET(smiles_encoding, protein_encoding, dropout_p)

    def forward(self, x_smiles, x_protein):
        x_smiles = x_smiles
        x_protein = x_protein
        out_smiles = self.cnn_smiles(x_smiles)
        out_smiles = out_smiles.view(out_smiles.size(0), -1)
        out_protein = self.cnn_protein(x_protein)
        out_protein = out_protein.view(out_protein.size(0), -1)
        out = torch.cat((out_smiles, out_protein), dim=1)
        out = self.drug_target_net(out)
        return out