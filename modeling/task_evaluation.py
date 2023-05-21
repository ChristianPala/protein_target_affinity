# Library to evaluate the DrugTargetNET model on the Drug Target Interaction (DTI) task
# Libraries:
import numpy as np
import logging
from cnn_encoders import CNNEncoderModel
from data_manager import ProteinAffinityData
from dti_model import DrugTargetNET, ModelTrainer
import torch
logging.getLogger("datasets").setLevel(logging.ERROR)  # Suppress warnings from the datasets library


# Constants:
# We trained the model on an NVIDIA A200 GPU machine from Google cloud,
# so we down-sample the dataset accordingly, fit to your hardware.
train_proportion = 'train[:7%]'
validation_proportion = 'train[7%:8%]'
test_proportion = 'train[8%:10%]'
morgan_fingerprint_encoding = 1024
conjoint_triad_encoding = 512 # 7^3 usually, but we have 1 unknown amino acid placeholder X, so 8^3.
protein_bert_encoding = 3072
epochs = 10
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_evaluation() -> None:
    """
    Main function to train and test the baseline model
    :return: None
    """
    # Load the data
    train_dataset = ProteinAffinityData(train_proportion)
    validation_dataset = ProteinAffinityData(validation_proportion)
    test_dataset = ProteinAffinityData(test_proportion)

    # Preprocess the data
    train_dataset.preprocess()
    validation_dataset.preprocess()
    test_dataset.preprocess()

    # Normalize the labels: papers often do not report this.
    mean_affinity = float(np.mean([x['affinity'] for x in train_dataset.data]))
    std_affinity = float(np.std([x['affinity'] for x in train_dataset.data]))
    train_dataset.normalize_affinity(mean_affinity, std_affinity)
    validation_dataset.normalize_affinity(mean_affinity, std_affinity)
    test_dataset.normalize_affinity(mean_affinity, std_affinity)

    # Get the dataloaders
    train_loader = train_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    validation_loader = validation_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    test_loader = test_dataset.get_dataloader(batch_size=batch_size, shuffle=False)

    # Get the input length for the model
    # Train the model
    model = DrugTargetNET(morgan_fingerprint_encoding, protein_bert_encoding).to(device)
    trainer = ModelTrainer(model, train_loader, validation_loader)
    trainer.train(epochs=epochs)

    # Save the model.
    trainer.save('dti_model.pth')

    # Test the model on the test set
    trainer.test(test_loader)

    # Plot the training and validation loss
    trainer.plot_losses()


def cnn_model_evaluation() -> None:
    """
    Main function to train and test the CNN model
    :return: None
    """
    # Load the data
    train_dataset = ProteinAffinityData(train_proportion)
    validation_dataset = ProteinAffinityData(validation_proportion)
    test_dataset = ProteinAffinityData(test_proportion)

    # Preprocess the data
    train_dataset.preprocess()
    validation_dataset.preprocess()
    test_dataset.preprocess()

    # Normalize the labels
    mean_affinity = float(np.mean([x['affinity'] for x in train_dataset.data]))
    std_affinity = float(np.std([x['affinity'] for x in train_dataset.data]))
    train_dataset.normalize_affinity(mean_affinity, std_affinity)
    validation_dataset.normalize_affinity(mean_affinity, std_affinity)
    test_dataset.normalize_affinity(mean_affinity, std_affinity)

    # Get the dataloaders
    train_loader = train_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    validation_loader = validation_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    test_loader = test_dataset.get_dataloader(batch_size=batch_size, shuffle=False)

    # Initialize the model. You need to provide appropriate dimensions based on your data.
    model = CNNEncoderModel(smiles_encoding=2048, protein_encoding=1024, dropout_p=0.1).to(device)
    trainer = ModelTrainer(model, train_loader, validation_loader)

    # Train the model
    trainer.train_cnn(epochs=epochs)

    # Save the model
    trainer.save('cnn_dti_model.pth')

    # Test the model on the test set
    trainer.test_cnn(test_loader)

    # Plot the training and validation loss
    trainer.plot_losses()


if __name__ == '__main__':
    cnn_model_evaluation()

