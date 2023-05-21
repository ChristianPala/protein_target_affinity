# Libraries:
import numpy as np
import logging

from torch.utils.data import Dataset

from cnn_encoders import CNNEncoderModel
from data_manager import DataPreprocessor, DataPreprocessorCNN
from dti_model import DrugTargetNET, ModelTrainer
import torch
logging.getLogger("datasets").setLevel(logging.ERROR)

# Constants
# Dataset downloading and down-sampling:
train_proportion = 'train[:7000]'
validation_proportion = 'train[7000:8000]'
test_proportion = 'train[8000:10000]'
# Model parameters:
morgan_fingerprint_encoding = 1024
conjoint_triad_encoding = 512  # Normally 7^3 = 343, but we have an unkown token X, so 8^3 = 512
protein_bert_encoding = 3072 # 768 (standard BERT embeddings) * 4 flattened layers
epochs = 10 # Best epoch was 10 with DeepPurpose
batch_size = 256 # Literature uses 256 often.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Functions:
def preprocess_dataset(dataset) -> torch.utils.data.DataLoader:
    """
    Preprocesses the dataset, normalizes the affinity and returns a dataloader.
    :param dataset: The downsampled dataset downloaded from the HuggingFace library by Jglaser.
    :return: A dataloader with the preprocessed data.
    """
    dataset.preprocess()
    mean_affinity = float(np.mean([x['affinity'] for x in dataset.data]))
    std_affinity = float(np.std([x['affinity'] for x in dataset.data]))
    dataset.normalize_affinity(mean_affinity, std_affinity)
    return dataset.get_dataloader(batch_size=batch_size, shuffle=False)

def train_model(model, train_loader, validation_loader, test_loader) -> None:
    """
    Trains the model and saves it.
    :param model: DrugTargetNET model or CNNEncoderModel to train.
    :param train_loader: the train dataloader.
    :param validation_loader: the validation dataloader.
    :param test_loader: the test dataloader.
    :return: None. Saves the model and plots the losses.
    """
    trainer = ModelTrainer(model, train_loader, validation_loader)
    trainer.train(epochs=epochs)
    trainer.save('dti_model.pth')
    trainer.test(test_loader)
    trainer.plot_losses()

def train_cnn_model(model, train_loader, validation_loader, test_loader) -> None:
    trainer = ModelTrainer(model, train_loader, validation_loader)
    trainer.train_cnn(epochs=epochs, smiles_encoding=morgan_fingerprint_encoding)
    trainer.save('cnn_dti_model.pth')
    trainer.test(test_loader)
    trainer.plot_losses()


def model_evaluation():
    train_dataset = DataPreprocessor(train_proportion)
    validation_dataset = DataPreprocessor(validation_proportion)
    test_dataset = DataPreprocessor(test_proportion)

    train_loader = preprocess_dataset(train_dataset)
    validation_loader = preprocess_dataset(validation_dataset)
    test_loader = preprocess_dataset(test_dataset)

    model = DrugTargetNET(smiles_encoding=morgan_fingerprint_encoding,
                          protein_encoding=protein_bert_encoding, dropout_p=0.1).to(device)

    train_model(model, train_loader, validation_loader ,test_loader)


def cnn_model_evaluation():
    train_dataset = DataPreprocessorCNN(train_proportion)
    validation_dataset = DataPreprocessorCNN(validation_proportion)
    test_dataset = DataPreprocessorCNN(test_proportion)

    train_loader = preprocess_dataset(train_dataset)
    validation_loader = preprocess_dataset(validation_dataset)
    test_loader = preprocess_dataset(test_dataset)

    model = CNNEncoderModel(smiles_encoding=1024, protein_encoding=64512, dropout_p=0.1).to(device)
    train_cnn_model(model, train_loader, validation_loader, test_loader)


if __name__ == '__main__':
    cnn_model_evaluation()