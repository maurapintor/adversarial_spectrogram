from decimal import Decimal

from audio_torch import ModelTrainer
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    accuracies = []
    models = ['trained-wd-0.000000.pt', 'trained-wd-0.000100.pt']
    plt.figure()
    for model in models:
        if model.endswith('.pt'):
            model_trainer.load_model(model)
            eps_values = np.arange(start=0, stop=0.1, step=0.01)
            eps_values /= model_trainer.train_dataset.max_value
            accs = model_trainer.security_evaluation(eps_values)
            accuracies.append(accs)
            wd = Decimal(float(model.split('-')[-1][:-3]))
            plt.plot(eps_values, accs, label="{:.2E}".format(wd))
            plt.title("Security evaluation")
            plt.xlabel("Perturbation strength")
            plt.ylabel("Test accuracy")
    plt.legend()
    plt.savefig(os.path.join(model_trainer.plot_dir, "Security evaluation.pdf"), format='pdf')
    np.save("accuracies", np.array(accuracies))
