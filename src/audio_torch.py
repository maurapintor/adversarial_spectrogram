import logging

logging.basicConfig()
logging.root.setLevel(logging.INFO)

from audio_dataset import AudioDataFolders
import torch
from torchvision import transforms
from torch.utils import data as data
from torch import optim
import os
import configparser
from torch import nn
import matplotlib.pyplot as plt

from net import Net


class ModelTrainer:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.log_interval = self.config.getint('TRAINING', 'log_interval', fallback=10)
        self.scheduler_steps = self.config.get('TRAINING', 'scheduler_steps', fallback='15,20').split(',')
        self.scheduler_steps = list(map(int, self.scheduler_steps))
        self.epochs = self.config.getint('TRAINING', 'epochs', fallback=30)
        self.data_dir = self.config.get('DATA', 'data_dir', fallback='data/speech_commands_prepared')
        self.include_dirs = self.config.get('DATA', 'include_dirs', fallback='yes,no,down,up').split(',')
        self.lr = self.config.getfloat('TRAINING', 'learning_rate', fallback=0.001)
        self.weight_decay = self.config.getfloat('TRAINING', 'weight_decay', fallback=0.001)
        self.lr_decay = self.config.getfloat('TRAINING', 'lr_decay', fallback=0.1)
        self.batch_size = self.config.getint('TRAINING', 'batch_size', fallback=64)
        self.n_workers = self.config.getint('TRAINING', 'n_workers', fallback=1)
        self.model_dir = self.config.get('RESULTS', 'model_dir', fallback="data/models")
        self.plot_dir = self.config.get('RESULTS', 'plot_dir', fallback="data/plots")
        for d in (self.model_dir, self.plot_dir):
            if not os.path.exists(d):
                os.mkdir(d)
        self.best_acc = 0
        self.losses = []
        self.lrs = []
        self.device = torch.device("cuda" if torch.cuda.is_available() and
                                             self.config.getboolean('TRAINING', 'use_cuda', fallback=True) else "cpu")
        logging.info("Using device : {}".format(self.device))
        self.transform = transforms.ToTensor()

        self.train_dataset = AudioDataFolders(root=os.path.join(self.data_dir, 'train'),
                                              extensions=('.wav',),
                                              transform=self.transform,
                                              include_folders=self.include_dirs)
        self.validation_dataset = AudioDataFolders(root=os.path.join(self.data_dir, 'validation'),
                                                   extensions=('.wav',),
                                                   transform=self.transform,
                                                   include_folders=self.include_dirs)
        self.test_dataset = AudioDataFolders(root=os.path.join(self.data_dir, 'test'),
                                             extensions=('.wav',),
                                             transform=self.transform,
                                             include_folders=self.include_dirs)
        self.train_loader = data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=self.n_workers)
        self.validation_loader = data.DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.n_workers)
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                           shuffle=False, num_workers=self.n_workers)
        self.model = Net().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.scheduler_steps,
                                                        gamma=self.lr_decay)

    def train_epoch(self, epoch):
        losses = []
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:  # print training stats
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))
                losses.append(loss.item())
        return losses

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        logging.info('Accuracy: {}/{} ({:.0f}%)'.format(
            correct, total,
            100. * correct / total))

        return correct / total

    def plot_info(self):
        plt.figure()
        plt.plot(self.losses)
        plt.savefig(os.path.join(self.plot_dir, "losses_{}.pdf".format(self.weight_decay)), format='pdf')
        plt.figure()
        plt.plot(self.lrs)
        plt.savefig(os.path.join(self.plot_dir, "learning_rates_{}.pdf".format(self.weight_decay)), format='pdf')

    def fit(self):
        logging.info("Init training. LR: {}\tepochs: {}\tlr_scheduling: {}"
                     "".format(self.scheduler.get_lr(), self.epochs, self.scheduler_steps))

        for epoch in range(1, self.epochs + 1):
            self.losses.extend(self.train_epoch(epoch))
            logging.info("Epoch {} completed.".format(epoch))
            logging.info("Computing validation accuracy...")
            val_acc = self.evaluate(self.validation_loader)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                logging.info("Validation acc: {:.3f}\tBest acc so far: {:.3f}."
                             "\tSaving model...".format(val_acc, self.best_acc))
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_dir, "trained-wd-{:.6f}.pt"
                                                        "".format(self.weight_decay)))
            self.scheduler.step(epoch)
            self.lrs.append(self.scheduler.get_lr())
        logging.info("Training completed. Computing test accuracy...")
        self.plot_info()
        self.evaluate(self.test_loader)
