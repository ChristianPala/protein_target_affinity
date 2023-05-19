# Library to evaluate the baseline model
# Libraries:
import numpy as np
import logging
from data_manager import ProteinAffinityData
from baseline_model import DrugTargetNET, ModelTrainer
logging.getLogger("datasets").setLevel(logging.ERROR)  # Suppress warnings from the datasets library

# Constants:
# as per JGLaser HuggingFace repository documentation. Adapt to CPU/GPU memory availability and performance.
train_proportion = 'train[:2%]'
validation_proportion = 'train[2%:3%]'
test_proportion = 'train[3%:4%]'


def baseline_evaluation() -> None:
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
    mean_affinity = float(np.mean([x['affinity'] for x in train_dataset.data]))
    std_affinity = float(np.std([x['affinity'] for x in train_dataset.data]))
    train_dataset.normalize_affinity(mean_affinity, std_affinity)
    validation_dataset.normalize_affinity(mean_affinity, std_affinity)
    test_dataset.normalize_affinity(mean_affinity, std_affinity)
    train_loader = train_dataset.get_dataloader(batch_size=64, shuffle=True)
    validation_loader = validation_dataset.get_dataloader(batch_size=64, shuffle=False)

    # Get the input length for the model
    input_length = train_dataset.max_seq_len + 1024  # Morgan fingerprint length

    # Train the model
    model = DrugTargetNET(input_length)
    trainer = ModelTrainer(model, train_loader, validation_loader)
    trainer.train(epochs=10)

    # Save and load the model to test the save/load functions
    trainer.save('baseline_model.pth')
    trainer.load('baseline_model.pth')

    # Test the model on the test set
    trainer.test(test_dataset.data)


if __name__ == '__main__':
    baseline_evaluation()
