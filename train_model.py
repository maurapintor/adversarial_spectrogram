from src.audio_torch import ModelTrainer

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    model_trainer.fit(penalty=0)
    model_trainer.fit(penalty=1e-6)
    model_trainer.fit(penalty=1e-5)
    model_trainer.fit(penalty=1e-4)
    model_trainer.fit(penalty=1e-3)
    model_trainer.fit(penalty=1e-2)
    model_trainer.fit(penalty=1e-1)
