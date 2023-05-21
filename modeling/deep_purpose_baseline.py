# Library to create a baseline model using DeepPurpose on the JGLaser HuggingFace repository
# Libraries:
from DeepPurpose import CompoundPred as models
from DeepPurpose.utils import data_process, generate_config
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from typing import Tuple
from itertools import product


# Functions:
def load_binding_db_data(proportion: str) -> Dataset:
    """
    Function to load the binding database dataset from the JGLaser HuggingFace repository
    :param proportion: str: The split for the training set
    :return: Tuple[Dataset, Dataset, Dataset]: The training, validation, and test datasets
    """
    return  load_dataset("jglaser/binding_affinity", split=proportion)

def preprocess_data(dataset: Dataset, drug_encoding: str, protein_encoding: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Function to preprocess the data, including computing the fingerprints and encoding the protein sequences,
    and combining the features into a single tensor.
    @param dataset: Dataset: The dataset to preprocess
    @param protein_encoding:  str: The encoding method for the protein sequences
    @param drug_encoding: str: The encoding method for the drug sequences
    :return: Dataset: The preprocessed dataset for training with DeepPurpose
    """
    # Get the drug sequences
    drug_sequences = dataset['smiles']

    # Get the protein sequences
    protein_sequences = dataset['seq']

    # Get the affinity values
    affinity = dataset['affinity']

    # Create a dictionary for the data
    train_d, val_d, test_d = data_process(drug_sequences, protein_sequences, affinity,
                                    drug_encoding=drug_encoding,
                                    target_encoding=protein_encoding,
                                    split_method='cold_protein', frac=[0.7, 0.1, 0.2])
    return train_d, val_d, test_d

def config_model(drug_encoding: str, target_encoding: str, train_epoch: int) -> dict:
    """
    Function to generate the configuration for the model
    :param drug_encoding: str: The encoding method for the drug sequences
    :param target_encoding: str: The encoding method for the protein sequences
    :param train_epoch: int: The number of epochs to train the model
    :return: dict: The configuration for the model
    """
    config = generate_config(drug_encoding=drug_encoding,
                             target_encoding=target_encoding,
                             cls_hidden_dims=[1024, 1024, 512],
                             train_epoch=train_epoch)

    return config

def deep_purpose_baseline() -> None:
    """
    Function to create a baseline models using DeepPurpose
    :return: None. Prints the results of the model
    """
    down_sample = 'train[:10%]'

    # Load the data
    dataset = load_binding_db_data(down_sample)

    # drug_encodings = ['Morgan', 'Pubchem', 'Daylight', 'rdkit_2d_normalized', 'CNN',
                      # 'CNN_RNN', 'Transformer', 'MPNN']

    # protein_encodings = ['Conjoint_triad','CNN', 'CNN_RNN', 'Transformer']
    # From our testing, the best encodings are: CNN and CNN
    drug_encodings = ['CNN']
    protein_encodings = ['CNN']

    best_mse = float('inf')
    best_model = None
    best_drug_encoding = None
    best_protein_encoding = None

    for d, p in product(drug_encodings, protein_encodings):
        # Preprocess the data
        train_d, val_d, test_d = preprocess_data(dataset, d, p)
        config = config_model(d, p, 10)

        # Create the model
        model = models.model_initialize(**config)

        # Train the model
        model.train(train_d, val_d, test_d)

        # Evaluate the model on validation set
        y_val_pred = model.predict(val_d)
        mse = mean_squared_error(y_val_pred, val_d.Label)

        # If this model performs better, save it
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_drug_encoding = d
            best_protein_encoding = p

    print("Best model drug encoding: ", best_drug_encoding)
    print("Best model protein encoding: ", best_protein_encoding)
    print("Best model MSE: ", best_mse)
    # Print the results
    print("DeepPurpose Best Baseline Results")
    print("Best model: ", best_model)
    print("Best model drug encoding: ", best_drug_encoding)
    print("Best model protein encoding: ", best_protein_encoding)
    print("Best model MSE: ", best_mse)

if __name__ == '__main__':
    deep_purpose_baseline()