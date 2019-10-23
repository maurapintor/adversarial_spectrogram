from audio_torch import ModelTrainer

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    accuracies = []
    model = 'trained-penalty-0.000000.pt'
    model_trainer.load_model(model)
    eps = 0.01 / model_trainer.train_dataset.max_value
    model_trainer.create_audio_examples(eps=eps)
