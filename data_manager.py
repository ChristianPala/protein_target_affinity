# Auxiliary class to load and preprocess the protein affinity dataset from the JGLaser HuggingFace repository
# for drug target interaction prediction
# Libraries:
import numpy as np
import torch
from datasets import load_dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer


# Constants:
# We use prot_bert to encode protein sequences.
# See: https://huggingface.co/Rostlab/prot_bert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pretrained prot_bert transformer and tokenizer for protein sequences
model_name = 'Rostlab/prot_bert'
model = BertModel.from_pretrained(model_name).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
# Load the pretrained Roberta transformer and tokenizer for protein sequences from
# ChemBERTa: A Pre-trained Language Model for Chemical Text Mining
s_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
s_model = SentenceTransformer("seyonec/PubChem10M_SMILES_BPE_450k")


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
        Function to convert a SMILES string into an embedding using a pretrained model and handle exceptions
        :param x: str: The SMILES string
        :return: dict: The input dictionary with the embedding added
        """
        try:
            # Encode the SMILES string
            encoding = s_tokenizer(x['smiles'], return_tensors='pt')

            # Pass the encoding through the model
            embeddings = s_model.encode(encoding)

            # Use the mean embedding if there are multiple
            mean_embedding = embeddings.mean(axis=0)

            x['smiles_fp'] = mean_embedding
        except Exception as e:
            print(f"Could not compute embedding for SMILES {x['smiles']}. Error: {e}")
        return x

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
    def _protein_encoding(x):
        """
        Function to encode a protein sequence using the ProtBert Transformer model
        :return:
        """
        # Tokenize the protein sequence
        sequence = x['seq']
        tokens = tokenizer.encode(sequence, add_special_tokens=True)
        # Convert the tokens to PyTorch tensors
        input_ids = torch.tensor(tokens).unsqueeze(0)

        # Move the input tensors to the appropriate device
        input_ids = input_ids.to(device)

        # Forward pass through the model to obtain the encoded representations
        with torch.no_grad():
            outputs = model(input_ids)
            encoded_sequence = outputs.last_hidden_state.squeeze(0)

        # Add the encoded sequence to the input dictionary
        x['protein_encoded'] = encoded_sequence

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

    @staticmethod
    def _combine_features(x):
        """
        Function to combine the features into a single tensor and flatten it to a 1D array
        :param x: the input dictionary
        :return: the input dictionary with the features combined and flattened
        """
        smiles_fp = torch.from_numpy(np.array(x['smiles_fp'])).unsqueeze(0).unsqueeze(0).to(device)
        protein_encoded = torch.from_numpy(np.array(x['protein_encoded'])).unsqueeze(0).to(device)
        combined_features = torch.cat((smiles_fp, protein_encoded), dim=1).to(device)
        flattened_features = combined_features.view(combined_features.size(0), -1)
        x['combined_features'] = flattened_features
        return x
