import logging

logging.basicConfig()
logging.root.setLevel(logging.INFO)

from audio_parser.audio_dataset import AudioDataFolders
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

train_dataset = AudioDataFolders(root=os.path.join(base_dir, 'train'),
                                 extensions=('.wav',),
                                 transform=transform,
                                 include_folders=include_dirs)
validation_dataset = AudioDataFolders(root=os.path.join(base_dir, 'validation'),
                                      extensions=('.wav',),
                                      transform=transform,
                                      include_folders=include_dirs)
test_dataset = AudioDataFolders(root=os.path.join(base_dir, 'test'),
                                extensions=('.wav',),
                                transform=transform,
                                include_folders=include_dirs)

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6)
validation_loader = data.DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=6)
test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()


def train(model, epoch):
    losses = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:  # print training stats
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            losses.append(loss.item())
    return losses


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        logging.info('Accuracy: {}/{} ({:.0f}%)'.format(
            correct, total,
            100. * correct / total))

    return correct / total


log_interval = 10
best_acc = 0
losses = []
lrs = []
steps = [15,]
epochs = 30
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)

logging.info("Init training. LR: {}\tepochs: {}\tlr_scheduling: {}".format(scheduler.get_lr(), epochs, steps))

for epoch in range(1, epochs+1):
    losses.extend(train(model, epoch))
    logging.info("Epoch {} completed.".format(epoch))
    logging.info("Computing validation accuracy...")
    val_acc = test(model, validation_loader)
    if val_acc > best_acc:
        best_acc = val_acc
        logging.info("Validation acc: {:.3f}\tBest acc so far: {:.3f}."
                     "\tSaving model...".format(val_acc, best_acc))
        torch.save(model.state_dict(), "models/trained-wd-0.pt")
    scheduler.step(epoch)
    lrs.append(scheduler.get_lr())

logging.info("Training completed. Computing test accuracy...")

test(model, test_loader)

import matplotlib.pyplot as plt

plt.plot(losses)
plt.ylim([0, 3])
plt.show()

plt.figure()
plt.plot(lrs)
plt.show()
