# Library to evaluate the baseline model
# Libraries:
import numpy as np
import logging
from data_manager import ProteinAffinityData
from baseline_model import DrugTargetNET, ModelTrainer
logging.getLogger("datasets").setLevel(logging.ERROR)  # Suppress warnings from the datasets library

# Constants:
full_proportion = 'train[:4%]'
train_proportion = 'train[:2%]'
validation_proportion = 'train[2%:3%]'
test_proportion = 'train[3%:4%]'
morgan_fp_length = 1024


def baseline_evaluation() -> None:
    """
    Main function to train and test the baseline model
    :return: None
    """
    # Load the data
    full_dataset = ProteinAffinityData(full_proportion)
    train_dataset = ProteinAffinityData(train_proportion)
    validation_dataset = ProteinAffinityData(validation_proportion)
    test_dataset = ProteinAffinityData(test_proportion)

    # Preprocess the data
    full_dataset.preprocess()
    global_max_seq_len = full_dataset.compute_global_max_seq_len()
    train_dataset.preprocess()
    validation_dataset.preprocess()
    test_dataset.preprocess()
    train_dataset.set_global_max_seq_len(global_max_seq_len)
    validation_dataset.set_global_max_seq_len(global_max_seq_len)
    test_dataset.set_global_max_seq_len(global_max_seq_len)
    mean_affinity = float(np.mean([x['affinity'] for x in train_dataset.data]))
    std_affinity = float(np.std([x['affinity'] for x in train_dataset.data]))
    train_dataset.normalize_affinity(mean_affinity, std_affinity)
    validation_dataset.normalize_affinity(mean_affinity, std_affinity)
    test_dataset.normalize_affinity(mean_affinity, std_affinity)
    train_loader = train_dataset.get_dataloader(batch_size=64, shuffle=True)
    validation_loader = validation_dataset.get_dataloader(batch_size=64, shuffle=False)
    test_loader = test_dataset.get_dataloader(batch_size=64, shuffle=False)

    # Get the input length for the model
    input_length = global_max_seq_len + morgan_fp_length

    # Train the model
    model = DrugTargetNET(input_length)
    trainer = ModelTrainer(model, train_loader, validation_loader)
    trainer.train(epochs=10)

    # Save and load the model to test the save/load functions
    trainer.save('baseline_model.pth')
    trainer.load('baseline_model.pth')

    # Test the model on the test set
    trainer.test(test_loader)


if __name__ == '__main__':
    baseline_evaluation()
