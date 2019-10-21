from audio_torch import ModelTrainer

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    accuracies = []
    model = 'trained-wd-0.000000.pt'
    model_trainer.load_model(model)
    eps = 0.001
    model_trainer.create_audio_examples(eps=eps)
