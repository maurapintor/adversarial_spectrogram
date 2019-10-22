from audio_torch import ModelTrainer

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    accuracies = []
    model = 'trained-wd-0.000100.pt'
    model_trainer.load_model(model)
    eps = 0.01 / model_trainer.train_dataset.max_value
    model_trainer.create_audio_examples(eps=eps)
