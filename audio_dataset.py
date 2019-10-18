import os
import sys

import librosa
import matplotlib.pyplot as plt
import sounddevice
import pyaudio
import torch
from torchvision import datasets, transforms
from torch.utils import data as data
import numpy as np
from PIL import Image

SAMPLING_RATE = 16000
DURATION = 2
SAMPLES = SAMPLING_RATE * DURATION
HOP_LENGTH = 200


class AudioDataFolders(datasets.DatasetFolder):
    def __init__(self, include_folders, **kwargs):
        self._include_folders = include_folders
        kwargs['loader'] = kwargs.get('loader', self._load_spectrogram)
        super(AudioDataFolders, self).__init__(**kwargs)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if self._include_folders is None:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            # get only included folders
            classes = self._include_folders
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _load_spectrogram(self, fname):
        audio, sr = librosa.load(fname, sr=SAMPLING_RATE, mono=False)
        y = np.zeros([SAMPLES])
        duration = audio.shape[0]
        if duration < SAMPLES:
            y[:duration] = audio[:]
        S = librosa.feature.melspectrogram(y, sr=sr, hop_length=HOP_LENGTH)
        return Image.fromarray(S)

    def __len__(self):
        return len(self.samples)
