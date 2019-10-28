from decimal import Decimal

from src.audio_torch import ModelTrainer
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    accuracies = []
    models = ['trained-penalty-0.000100.pt']
    plt.figure()
    label = ['DNN', 'Robust DNN']
    color = ['g']
    for j, model in enumerate(models):
        if model.endswith('.pt'):
            print("Running sec eval for model: {}".format(model))
            model_trainer.load_model(model)
            eps_values = np.arange(start=0, stop=1, step=0.05)
            eps_values /= model_trainer.train_dataset.max_value
            accs = model_trainer.security_evaluation(eps_values)
            wd = float(model.split('-')[-1][:-3])
            plt.plot(eps_values*1e4, accs, label=label[j], c=color[j])
            plt.title("Security evaluation")
            plt.xlabel("Perturbation strength")
            plt.ylabel("Test accuracy")
            plt.ylim([0, 1])
    plt.xlabel("Perturbation (mels) x 1E4")
    plt.savefig(os.path.join(model_trainer.plot_dir, "Security evaluation.pdf"), format='pdf')
    np.save("accuracies", np.array(accuracies))
