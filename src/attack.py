import logging

import librosa
import sounddevice
import torchvision
from PIL import Image
from scipy.io import wavfile
from torch.autograd import Variable

logging.basicConfig()
logging.root.setLevel(logging.INFO)

from audio_parser.audio_dataset import AudioDataFolders, SAMPLING_RATE, HOP_LENGTH
import torch
from torchvision import transforms
from torch.utils import data as data
from torch import optim
import os

from audio_parser.net import Net

base_dir = '/home/maurapintor/data/SPEECH/'

include_dirs = ['yes', 'no', 'down', 'up']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = AudioDataFolders(root=os.path.join(base_dir, 'test'),
                                extensions=('.wav',),
                                transform=transform,
                                include_folders=include_dirs)

test_loader = data.DataLoader(test_dataset, batch_size=10, shuffle=True)

model = Net().to(device)
model.load_state_dict(torch.load("models/trained-wd-001.pt"))


def i_fgsm(model, x, y=None, targeted=False,
           eps=0.001, alpha=0.001, iteration=1,
           x_val_min=0, x_val_max=1):
    x.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(iteration):
        outputs = model(x)
        if targeted:
            loss = criterion(outputs, y)
        else:
            loss = -criterion(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = x.grad.data
        grad = torch.sign(grad)
        x = x - alpha * grad
        x = torch.where(x > x + eps, x + eps, x)
        x = torch.where(x < x - eps, x - eps, x)
        x = torch.clamp(x, x_val_min, x_val_max)
        x = Variable(x.data, requires_grad=True)
    return x


def to_audio(x, x_adv, y):
    for i, (sp, sp_adv, label) in enumerate(zip(x, x_adv, y)):
        sp = sp.squeeze().detach().numpy()
        sp_adv = sp_adv.squeeze().detach().numpy()
        label = label.item()
        if not os.path.exists("adv_audio/{}".format(label)):
            os.mkdir("adv_audio/{}".format(label))
        audio = librosa.feature.inverse.mel_to_audio(sp, sr=SAMPLING_RATE, hop_length=HOP_LENGTH)
        wavfile.write("adv_audio/{}/{}_orig.wav".format(label, i), SAMPLING_RATE, audio)
        audio_adv = librosa.feature.inverse.mel_to_audio(sp_adv, sr=SAMPLING_RATE, hop_length=HOP_LENGTH)
        wavfile.write("adv_audio/{}/{}_adv.wav".format(label, i), SAMPLING_RATE, audio_adv)


def test(model, loader):
    model.eval()
    correct = 0
    adv_correct = 0
    total = 0
    i = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        adv_data = i_fgsm(model, data, target)
        # to_audio(data, adv_data, target)
        # if i==0: quit()
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        adv_output = model(adv_data)
        _, adv_predicted = torch.max(adv_output.data, 1)
        adv_correct += (adv_predicted == target).sum().item()

    logging.info('Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, total,
        100. * correct / total))
    logging.info('Adv accuracy: {}/{} ({:.0f}%)\n'.format(
        adv_correct, total,
        100. * adv_correct / total))


logging.info("Computing test accuracy...")
test(model, test_loader)
