# Auxiliary class to load and preprocess the protein affinity dataset from the JGLaser HuggingFace repository
# for drug target interaction prediction
# Libraries:
import itertools
import numpy as np
import torch
from datasets import load_dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer  #  Imports from ChemBERTa, function removed due to poor
# performance
from collections import Counter



# Constants:
# We use prot_bert to encode protein sequences.
# See: https://huggingface.co/Rostlab/prot_bert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pretrained prot_bert transformer and tokenizer for protein sequences
model_name = 'Rostlab/prot_bert'
model = BertModel.from_pretrained(model_name).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
# avoid some weights of the model checkpoint... warning
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# Classes:
class DataPreprocessor:
    """
    Class to load and preprocess the protein affinity dataset from the JGLaser HuggingFace repository
    """
    # Constants:
    amino_acid_properties = { # From https://doi.org/10.1186/s12859-015-0828-1 by Wang, H., and Hu, X.
        'A': 'A', 'G': 'A', 'V': 'A',  # AGV
        'I': 'B', 'L': 'B', 'F': 'B', 'P': 'B',  # ILFP
        'Y': 'C', 'M': 'C', 'T': 'C', 'S': 'C',  # YMTS
        'H': 'D', 'N': 'D', 'Q': 'D', 'W': 'D',  # HNQW
        'R': 'E', 'K': 'E',  # RK
        'D': 'F', 'E': 'F',  # DE
        'C': 'G',  # C
        'X': 'H'  # Unknown amino acid, added by us to account for the fact that some proteins have unknown amino acids
        # and to ensure the last amino acid is always encoded as a triplet.
    }

    def __init__(self, split: str) -> None:
        """
        Constructor for the ProteinAffinityData class
        :param split:
        """
        self.data = load_dataset("jglaser/binding_affinity", split=split)
        self.all_possible_triads = [''.join(t) for t in
                                   itertools.product(set(DataPreprocessor.amino_acid_properties.values()),
                                                     repeat=3)]

    def preprocess(self) -> None:
        """
        Function to preprocess the data, including computing the fingerprints and encoding the protein sequences,
        and combining the features into a single tensor.
        :return:
        """
        self.data = self.data.map(self._safe_morgan_smiles_to_fp)
        self.data = self.data.map(self._protein_encoding_transformer)

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
    def _safe_morgan_smiles_to_fp(x):
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
    def _protein_encoding_transformer(x):
        """
        Function to encode a protein sequence using the ProtBert Transformer model
        :return: dict: The input dictionary with the encoded protein sequence added
        """
        sequence = x['seq']
        tokens = tokenizer.encode(sequence, add_special_tokens=True)
        input_ids = torch.tensor(tokens).unsqueeze(0)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            encoded_sequence = outputs.last_hidden_state.squeeze(0)
        x['protein_encoded'] = encoded_sequence

        return x

    def _protein_encoding_conjoint(self, x) -> None:
        """
        Function to encode a protein sequence using Conjoint Triad properties
        :return:
        """
        assert len(self.all_possible_triads) == 512, "There should be 8^3 possible triads"
        sequence = x['seq']
        property_sequence = [self.amino_acid_properties.get(aa.capitalize(), 'X') for aa in sequence]
        property_sequence.extend(['X', 'X'])  # Append 'X' for last two amino acids to make triads
        triads = ["".join(triad) for triad in zip(property_sequence, property_sequence[1:],
                                                  property_sequence[2:])]
        triad_frequencies = Counter(triads)
        triad_vector = [triad_frequencies[triad] for triad in self.all_possible_triads]
        x['protein_encoded'] = np.array(triad_vector)
        return x
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
    def _collate_fn(batch):
        """
        Function to collate the data into a single tensor for the DataLoader, flattening the features
        and returning the affinity as a tensor.
        :param batch: the batch of data
        :return: the collated data
        """
        smiles_fp = torch.stack(
            [torch.from_numpy(np.array(item['smiles_fp'])).to(device) for item in batch])
        protein_encoded = torch.stack(
            [torch.from_numpy(np.array(item['protein_encoded'])).to(device) for item in batch])

        # Flatten the features
        flattened_smiles_fp = smiles_fp.view(smiles_fp.size(0), -1)
        flattened_protein_encoded = protein_encoded.view(protein_encoded.size(0), -1)

        # Concatenate the flattened features
        combined_features = torch.cat((flattened_smiles_fp, flattened_protein_encoded), dim=1).to(device)

        affinity = torch.tensor([item['affinity'] for item in batch]).float().to(device)
        return {'combined_features': combined_features, 'affinity': affinity}


class DataPreprocessorCNN(DataPreprocessor):

    def __init__(self, split: str) -> None:
        """
        Constructor for the ProteinAffinityData class
        :param split:
        """
        super().__init__(split)
        self.data = self.data.map(self._safe_morgan_smiles_to_fp)
        self.data = self.data.map(self._protein_encoding_transformer)

    def preprocess(self):
        """
        Function to preprocess the data, including computing the fingerprints and encoding the protein sequences,
        and combining the features into a single tensor.
        :return:
        """
        self.data = self.data.map(self._safe_morgan_smiles_to_fp)
        self.data = self.data.map(self._encode_protein_sequence)


    @staticmethod
    def _encode_protein_sequence(x, amino_acids: str = 'ACDEFGHIKLMNPQRSTVWXY', max_length: int = 1200):
        """
        One-hot encode a protein sequence.

        Args:
            x: The input dictionary.
            amino_acids: A string of possible amino acids.
            max_length: The maximum length of a protein sequence. Longer sequences are truncated,
            and shorter ones are padded with zeros.

        Returns:
            A one-hot encoding for the protein sequence.
        """
        protein = x['seq']
        encoding = []
        for i in protein:
            if i in amino_acids:
                encoding.append([int(i == aa.upper()) for aa in amino_acids])
            else:
                return None

        # If the protein sequence is shorter than max_length, pad the encoding with zeros
        if len(encoding) < max_length:
            encoding += [[0 for _ in amino_acids]] * (max_length - len(encoding))
        # If the protein sequence is longer than max_length, truncate the encoding, we selected 1200 based on
        # the EDA and our applicability domain analysis
        elif len(encoding) > max_length:
            encoding = encoding[:max_length]

        x['protein_encoded'] = np.array(encoding)

        return x

    @staticmethod
    def _collate_fn(batch):
        """
        Function to collate the data into a single tensor for the DataLoader, reshaping the features
        for compatibility with a Conv1D layer, and returning the affinity as a tensor.
        :param batch: the batch of data
        :return: the collated data
        """
        smiles_fp = torch.stack(
            [torch.from_numpy(np.array(item['smiles_fp'])).unsqueeze(1).to(device) for item in batch])
        protein_encoded = torch.stack(
            [torch.from_numpy(np.array(item['protein_encoded'])).unsqueeze(0).to(device) for item in
             batch])
        affinity = torch.tensor([item['affinity'] for item in batch]).float().to(device)
        return {'smiles_fp': smiles_fp, 'protein_encoded': protein_encoded, 'affinity': affinity}

    def get_dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        """
        Function to get a PyTorch DataLoader for the dataset
        :param batch_size: int: The batch size
        :param shuffle: bool: Whether to shuffle the data
        :return: DataLoader: The PyTorch DataLoader
        """
        return DataLoader(self.data, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)