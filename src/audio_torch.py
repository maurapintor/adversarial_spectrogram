import logging
from decimal import Decimal

import torchattacks

logging.basicConfig()
logging.root.setLevel(logging.INFO)

from src.audio_dataset import AudioDataFolders
import torch
from torchvision import transforms
from torch.utils import data as data
from torch import optim
import os
import configparser
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.net import Net

class ModelTrainer:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.log_interval = self.config.getint('TRAINING', 'log_interval', fallback=100)
        self.scheduler_steps = self.config.get('TRAINING', 'scheduler_steps', fallback='20,40').split(',')
        self.scheduler_steps = list(map(int, self.scheduler_steps))
        self.epochs = self.config.getint('TRAINING', 'epochs', fallback=30)
        self.data_dir = self.config.get('DATA', 'data_dir', fallback='data/speech_commands_prepared')
        self.include_dirs = self.config.get('DATA', 'include_dirs',
                                            fallback='down,go,left,no,off,'
                                                     'on,right,stop,up,yes').split(',')
        self.n_test = self.config.getint('DATA', 'n_test', fallback=1000)
        self.lr = self.config.getfloat('TRAINING', 'learning_rate', fallback=0.001)
        self.weight_decay = self.config.getfloat('TRAINING', 'weight_decay', fallback=0)
        self.gradient_penalty = self.config.getfloat('TRAINING', 'gradient_penalty', fallback=0)
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

        self.subset_idxs = torch.randperm(len(self.test_dataset))
        self.subset_idxs = self.subset_idxs[:self.n_test]
        self.test_subset = data.Subset(self.test_dataset, self.subset_idxs)

        self.train_loader = data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=self.n_workers)
        self.validation_loader = data.DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.n_workers)
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                           shuffle=False, num_workers=self.n_workers)
        self.subset_loader = data.DataLoader(self.test_subset, batch_size=self.batch_size,
                                             shuffle=False, num_workers=self.n_workers)

        self.model = Net(n_output=len(self.include_dirs)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.scheduler_steps,
                                                        gamma=self.lr_decay)

        self.scale_factor = self.train_dataset.max_value
        logging.info("Scaling the images, scale factor = {}".format(self.scale_factor))
        # self.rescale = transforms.Lambda(lambda x: x / self.scale_factor)
        self.inverse_rescale = transforms.Lambda(lambda x: x * self.scale_factor)

    def train_epoch(self, epoch):
        losses = []
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = self.model(data)
            loss = self.criterion(output, target)
            if self.gradient_penalty != 0:
                loss += self.gradient_penalty * \
                        torch.norm(torch.autograd.grad(self.criterion(output, target),
                                                       data,
                                                       retain_graph=True,
                                                       create_graph=True)[0], p=1)
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
        plt.savefig(os.path.join(self.plot_dir, "losses_{}.pdf".format(self.gradient_penalty)), format='pdf')
        plt.figure()
        plt.plot(self.lrs)
        plt.savefig(os.path.join(self.plot_dir, "learning_rates_{}.pdf".format(self.gradient_penalty)), format='pdf')

    def fit(self, penalty=None):
        self.best_acc = 0
        self.losses = []
        self.lrs = []
        if penalty is not None:
            self.gradient_penalty = penalty
        logging.info("Init training. LR: {}\tepochs: {}\tlr_scheduling: {}\tpenalty: {}"
                     "\tkeywords : {}".format(self.scheduler.get_lr(),
                               self.epochs,
                               self.scheduler_steps,
                               self.gradient_penalty,
                               self.include_dirs))

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
                           os.path.join(self.model_dir, "trained-penalty-{:.6f}.pt"
                                                        "".format(self.gradient_penalty)))
            self.scheduler.step(epoch)
            self.lrs.append(self.scheduler.get_lr())
        logging.info("Training completed. Computing test accuracy...")
        self.plot_info()
        self.evaluate(self.test_loader)

    def load_model(self, model_name):
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, model_name),
                       map_location=self.device))

    def create_adv_ds(self, loader, eps, mask=False):
        if eps == 0:
            for data, target in loader:
                yield data, target
        else:
            attack = torchattacks.PGD(self.model, eps=eps, alpha=eps / 20, iters=20)
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                adversarial_images = attack(data, target)
                if mask is True:
                    adversarial_images[data < 1e-6] = data[data < 1e-6]
                    adversarial_images[adversarial_images < 1e-6] = data[adversarial_images < 1e-6]
                yield adversarial_images, target

    def create_noisy_ds(self, loader, eps, mask=False):
        if eps == 0:
            for data, target in loader:
                yield data, target
        else:
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                noisy = data + (torch.randn(data.shape, device=self.device).sign() * eps)
                if mask is True:
                    noisy[data < 1e-6] = data[data < 1e-6]
                    noisy[noisy < 1e-6] = data[noisy < 1e-6]
                yield noisy, target

    def security_evaluation(self, values, noise=False):
        """

        :param values:
        :param noise: if True, uses random noise instead of the
            worst-case adversarial perturbation
        :return:
        """
        accuracies = []
        for i, eps_value in enumerate(values):
            if noise is False:
                adv_ds = self.create_adv_ds(self.subset_loader, eps=eps_value)
            else:
                adv_ds = self.create_noisy_ds(self.subset_loader, eps=eps_value)
            accuracies.append(self.evaluate(adv_ds))
        return accuracies

    def create_audio_examples(self, eps, n_samples=3):
        i = 0
        adv_loader = self.create_adv_ds(self.subset_loader, eps=eps)
        cls_to_idx = self.train_dataset.class_to_idx
        cls_to_idx = {v: k for k, v in cls_to_idx.items()}
        for (batch_images, batch_labels), (batch_images_adv, _) in zip(self.subset_loader, adv_loader):
            eps = Decimal(eps)
            for image, adv_image, label in zip(batch_images, batch_images_adv, batch_labels):
                image, adv_image = image.to(self.device), adv_image.to(self.device)
                predicted = self.model(image.unsqueeze(0)).topk(1)[1]
                adv_pred = self.model(adv_image.unsqueeze(0)).topk(1)[1]
                if label.item() == predicted.item() \
                        and predicted.item() != adv_pred.item():
                    i += 1
                    folder = os.path.join(self.plot_dir,
                                          "eps{:.2E}_to_{}_{}".format(eps, label.item(), adv_pred.item()))
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                        image, adv_image = self.inverse_rescale(image), self.inverse_rescale(adv_image)
                        image = image.squeeze().cpu().detach().numpy()
                        adv_image = adv_image.squeeze().cpu().detach().numpy()
                        diff_img = (image - adv_image)
                        diff_img -= diff_img.min()
                        # store spectrogram
                        plt.figure(figsize=(5, 10))
                        plt.subplot(3, 1, 1)
                        plt.tick_params(color='black')
                        plt.imshow(image, cmap='gray_r')
                        plt.xticks([])
                        plt.yticks([])
                        plt.title("ORIGINAL IMAGE\n{} ({})"
                                  "".format(cls_to_idx[label.item()], cls_to_idx[predicted.item()]),
                                  color=("green" if label.item() == predicted.item() else "red"))
                        plt.subplot(3, 1, 2)
                        plt.tick_params(color='black')
                        plt.imshow(adv_image, cmap='gray_r')
                        plt.xticks([])
                        plt.yticks([])
                        plt.title("ADV IMAGE\n{} ({})"
                                  "".format(cls_to_idx[label.item()], cls_to_idx[adv_pred.item()]),
                                  color=("green" if label.item() == adv_pred.item() else "red"))
                        plt.subplot(3, 1, 3)
                        plt.title("Perturbation (dmax = {:.2E})".format(Decimal(eps)))
                        plt.tick_params(color='black')
                        plt.imshow(diff_img, cmap='gray_r')
                        plt.xticks([])
                        plt.yticks([])
                        plt.savefig(os.path.join(folder,
                                                 "perturbation_{}_to_{}_eps_{:.2E}.pdf"
                                                 "".format(label.item(), adv_pred.item(),
                                                           eps)),
                                    format='pdf')
                        plt.close()
                        print("Saving audios")
                        audio1 = self.test_dataset.invert_spectrogram(image)
                        audio2 = self.test_dataset.invert_spectrogram(adv_image)
                        AudioDataFolders.save_audio(
                            audio1,
                            os.path.join(folder,
                                         'original_{}_eps_{:.2E}.wav'
                                         ''.format(label.item(), eps)))
                        adv_audio_path = os.path.join(folder,
                                                      'perturbed_{}_eps_{:.2E}.wav'
                                                      ''.format(adv_pred.item(), eps))
                        AudioDataFolders.save_audio(
                            audio2,
                            adv_audio_path)
                        print("Converting back to audio")
                        retransformed = transforms.ToTensor()(self.test_dataset._load_spectrogram(adv_audio_path))
                        print(retransformed.max())
                        print(adv_image.max())
                        print(label.item(), adv_pred.item(), self.model(retransformed.unsqueeze(0)).topk(1)[1].item())
                        if i == n_samples: break
            break
