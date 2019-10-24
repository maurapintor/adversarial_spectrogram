from audio_torch import ModelTrainer
import numpy as np

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    accuracies = []
    model = 'trained-penalty-0.000000.pt'
    model_trainer.load_model(model)
    eps_values = np.arange(start=0, stop=0.2, step=0.1)
    eps_values /= model_trainer.train_dataset.max_value
    for eps_value in eps_values:
        model_trainer.create_audio_examples(eps=eps_value, n_samples=1)
