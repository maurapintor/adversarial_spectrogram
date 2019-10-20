from audio_torch import ModelTrainer

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    model_trainer.load_model("trained-wd-0.000000.pt")
    model_trainer.evaluate(model_trainer.test_loader)
