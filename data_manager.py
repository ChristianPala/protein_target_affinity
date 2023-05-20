# Auxiliary class to load and preprocess the protein affinity dataset from the JGLaser HuggingFace repository
# Libraries:
import numpy as np
import torch
from datasets import load_dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Classes:
class ProteinAffinityData:
    """
    Class to load and preprocess the protein affinity dataset from the JGLaser HuggingFace repository
    """
    def __init__(self, split: str) -> None:
        """
        Constructor for the ProteinAffinityData class
        :param split:
        """
        self.data = load_dataset("jglaser/binding_affinity", split=split)
        self.global_max_seq_len = None

    def compute_global_max_seq_len(self):
        """
        Compute the maximum sequence length across the entire dataset (train, test)
        """
        if self.global_max_seq_len is None:
            self.global_max_seq_len = max(len(x['protein_encoded']) for x in self.data)
        return self.global_max_seq_len

    def preprocess(self):
        """
        Function to preprocess the data, including computing the fingerprints and encoding the protein sequences,
        and combining the features into a single tensor.
        :return:
        """
        self.data = self.data.map(self._safe_smiles_to_fp).filter(lambda x: x is not None)
        self.data = self.data.map(self._protein_encoding)
        self.data = self.data.map(self._combine_features)

    def normalize_affinity(self, mean: float, std: float) -> None:
        """
        Function to normalize the affinity values in the dataset
        :param mean: float: The mean value of affinity in the training set
        :param std: float: The standard deviation of affinity in the training set
        :return: None. The data is modified in place
        """
        self.data = self.data.map(lambda x: self._normalize_affinity(x, mean, std))

    def get_dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        """
        Function to get a PyTorch DataLoader for the dataset
        :param batch_size: int: The batch size
        :param shuffle: bool: Whether to shuffle the data
        :return: DataLoader: The PyTorch DataLoader
        """
        return DataLoader(self.data, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)

    @staticmethod
    def _safe_smiles_to_fp(x):
        """
        Function to convert a SMILES string into a Morgan fingerprint and handle exceptions
        :param x: str: The SMILES string
        :return: dict: The input dictionary with the fingerprint added
        """
        try:
            mol = Chem.MolFromSmiles(x['smiles'])
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            np_fp = np.zeros((0,))
            DataStructs.ConvertToNumpyArray(fp, np_fp)
            x['smiles_fp'] = np_fp
        except Exception as e:
            print(f"Could not compute fingerprint for SMILES {x['smiles']}. Error: {e}")
        return x

    @staticmethod
    def _protein_encoding(x):
        """
        Function to encode a protein sequence as a list of integers, X is the unknown amino acid.
        :param x: dict: The input dictionary
        :return: dict: The input dictionary with the protein sequence encoded
        """
        amino_acid_dict = {amino_acid: i for i, amino_acid in enumerate('ACDEFGHIKLMNPQRSTVXWY')}
        x['protein_encoded'] = [amino_acid_dict[aa.capitalize()] for aa in x['seq']]
        return x

    def _calculate_max_seq_len(self):
        """
        Auxiliary function to calculate the maximum sequence length of the proteins in the dataset
        :return: None. The maximum sequence length is stored as an attribute of the class
        """
        self.max_seq_len = max([len(item['protein_encoded']) for item in self.data])

    @staticmethod
    def _normalize_affinity(x, mean, std):
        """
        Function to normalize the affinity values in the dataset
        :param x: the input dictionary
        :param mean: the mean value of affinity in the training set
        :param std: the standard deviation of affinity in the training set
        :return: the input dictionary with the affinity normalized
        """
        x['affinity'] = (x['affinity'] - mean) / std
        return x

    @staticmethod
    def _combine_features(x):
        """
        Function to combine the protein encoding and the SMILES fingerprint into a single tensor
        :param x: the input dictionary
        :return: the input dictionary with the combined features
        """
        x['combined_features'] = torch.Tensor(np.concatenate((x['smiles_fp'], x['protein_encoded'])))
        return x

    def calculate_max_seq_len(self, dataset):
        """
        Function to compute the maximum protein sequence length across an entire dataset.
        :param dataset: the input dataset
        :return: the maximum protein sequence length
        """
        self.max_seq_len = max(len(x['seq']) for x in dataset)

    def _collate_fn(self, batch):
        """
        Function to collate the data into a single tensor for the DataLoader
        :param batch: the batch of data
        :return: the collated data
        """
        smiles_fp = torch.stack([torch.from_numpy(np.array(item['smiles_fp'])) for item in batch])
        protein_encoded = pad_sequence([torch.tensor(item['protein_encoded']) for item in batch], batch_first=True,
                                       padding_value=0)
        protein_encoded_padded = F.pad(protein_encoded, (0, self.compute_global_max_seq_len() - protein_encoded.shape[1]))
        combined_features = torch.cat((smiles_fp, protein_encoded_padded), dim=1)
        affinity = torch.tensor([item['affinity'] for item in batch])
        return {'combined_features': combined_features, 'affinity': affinity}

    def set_global_max_seq_len(self, max_seq_len):
        """
        Function to set the maximum sequence length
        :param max_seq_len: the maximum sequence length
        :return: None. The maximum sequence length is stored as an attribute of the class
        """
        self.global_max_seq_len = max_seq_len
