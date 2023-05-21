# Library to encode SMILES and protein sequences using CNN models
import torch
from torch import nn
from dti_model import DrugTargetNET


class ConvBlock(nn.Module):
    """
    Class to create a convolutional block for the CNN encoders for SMILES and protein sequences.
    """
    def __init__(self, in_channels: int):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=3, stride=1, padding=1)
        self.adaptive_maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.adaptive_maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.adaptive_maxpool(x)
        return x


class CNNEncoderModel(nn.Module):
    """
    Class to create a CNN encoder model for SMILES and protein sequences, combining the two encodings
    and passing them through the DrugTargetNET model.
    """
    def __init__(self, smiles_encoding: int, protein_encoding: int, dropout_p: float = 0.1):
        super(CNNEncoderModel, self).__init__()
        self.smiles_features = smiles_encoding
        self.protein_features = protein_encoding
        self.cnn_smiles = ConvBlock(smiles_encoding)
        self.cnn_protein = ConvBlock(protein_encoding)
        self.fcn = DrugTargetNET(self.smiles_features, self.protein_features, dropout_p)

    def forward(self, x_smiles, x_protein):
        out_smiles = self.cnn_smiles(x_smiles)
        out_smiles = out_smiles.view(out_smiles.size(0), -1)

        out_protein = self.cnn_protein(x_protein)
        out_protein = out_protein.view(out_protein.size(0), -1)

        out = torch.cat((out_smiles, out_protein), dim=1)
        out = self.fcn(out)
        return out