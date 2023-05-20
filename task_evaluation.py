# Library to evaluate the DrugTargetNET model on the Drug Target Interaction (DTI) task
# Libraries:
import numpy as np
import logging
from data_manager import ProteinAffinityData
from dti_model import DrugTargetNET, ModelTrainer
import torch
logging.getLogger("datasets").setLevel(logging.ERROR)  # Suppress warnings from the datasets library

# Constants:
# We trained the model on an NVIDIA A200 GPU machine from Google cloud,
# so we down-sample the dataset accordingly, fit to your hardware.
train_proportion = 'train[:2%]'
validation_proportion = 'train[2%:3%]'
test_proportion = 'train[3%:4%]'
pubchem_roberta_encoding_length = 1024
prot_bert_encoding_length = 3072
epochs = 30
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

    # Get the input length for the model
    # Train the model
    model = DrugTargetNET(pubchem_roberta_encoding_length, prot_bert_encoding_length).to(device)  # Move model to device
    trainer = ModelTrainer(model, train_loader, validation_loader)
    trainer.train(epochs=epochs)

    # Save the model.
    trainer.save('dti_model.pth')

    # Test the model on the test set
    trainer.test(test_loader)


if __name__ == '__main__':
    model_evaluation()