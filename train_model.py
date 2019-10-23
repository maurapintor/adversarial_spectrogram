from audio_torch import ModelTrainer

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    model_trainer.fit(penalty=0)
    model_trainer.fit(penalty=0.01)
    model_trainer.fit(penalty=0.1)
