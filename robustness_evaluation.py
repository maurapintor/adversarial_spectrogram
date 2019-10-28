from decimal import Decimal

from src.audio_torch import ModelTrainer
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    models = [
        'trained-penalty-0.000000.pt',
        'trained-penalty-0.000100.pt']
    plt.figure()
    label = ['DNN', 'Robust DNN']
    color = ['r', 'g']
    for j, model in enumerate(models):
        if model.endswith('.pt'):
            print("Running sec eval for model: {}".format(model))
            model_trainer.load_model(model)
            eps_values = np.arange(start=0, stop=1, step=0.05)
            eps_values /= model_trainer.train_dataset.max_value
            accs_adv = model_trainer.security_evaluation(eps_values, noise=False)
            plt.plot(eps_values*1e4, accs_adv, label=label[j], c=color[j])
            accs_noise = model_trainer.security_evaluation(eps_values, noise=True)
            plt.plot(eps_values*1e4, accs_noise, label=label[j], c=color[j], linestyle='-.')
            plt.title("Robustness evaluation (random noise)")
            plt.xlabel("Perturbation strength")
            plt.ylabel("Test accuracy")
            plt.ylim([0, 1])
    plt.legend()
    plt.xlabel("Perturbation (mels) x 1E4")
    plt.savefig(os.path.join(model_trainer.plot_dir, "Robustness evaluation.pdf"), format='pdf')
