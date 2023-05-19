# Library to evaluate the baseline model
# Libraries:
import numpy as np
import logging
from protein_affinity_data_manager import ProteinAffinityData
from baseline_model import DrugTargetNET, ModelTrainer
logging.getLogger("datasets").setLevel(logging.ERROR)  # Suppress warnings from the datasets library


def main() -> None:
    """
    Main function to train and test the baseline model
    :return: None
    """
    train_dataset = ProteinAffinityData('train[:3%]')
    validation_dataset = ProteinAffinityData('train[3%:4%]')
    test_dataset = ProteinAffinityData('train[4%:5%]')

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

    model = DrugTargetNET()
    trainer = ModelTrainer(model, train_loader, validation_loader)

    trainer.train(epochs=10)
    trainer.save('baseline_model.pth')

    trainer.load('baseline_model.pth')
    trainer.test(test_dataset.data)


if __name__ == '__main__':
    main()
