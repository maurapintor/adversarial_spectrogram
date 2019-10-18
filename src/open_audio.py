import os
import librosa
import matplotlib.pyplot as plt
import sounddevice
import numpy as np
import pyaudio
base_dir = '/home/maurapintor/data/speech/'

kw_list = ['yes', 'no', 'down', 'up']
n_imgs = 5
SAMPLES = 16000
plt.figure(figsize=(n_imgs*5, len(kw_list)*5))
for kw_idx, kw in enumerate(kw_list):
    d = os.path.join(base_dir, kw)
    for i in range(n_imgs):
        dd = os.listdir(d)[i]
        plt.subplot(len(kw_list), n_imgs, i+1+n_imgs*kw_idx)
        plt.title(kw)
        dd = os.path.join(d, dd)
        audio, sr = librosa.load(dd, sr=8000, mono=True)
        y = np.zeros([SAMPLES])
        duration = audio.shape[0]
        if duration < SAMPLES:
            y[:duration] = audio[:]
        S = librosa.feature.melspectrogram(y, sr=sr, hop_length=100)

        plt.imshow(S)
plt.show()

#
# audio = librosa.feature.inverse.mel_to_audio(S, sr=sr, hop_length=100)
# sounddevice.play(y, blocking=True, samplerate=sr)
# sounddevice.play(audio, blocking=True, samplerate=sr)
#
# S2 = librosa.feature.melspectrogram(audio, sr=sr, hop_length=100)
# plt.figure()
# plt.imshow(S2 - S)
# print(np.linalg.norm(S2 - S))
#
# print(y.shape)
# print(audio.shape)
# plt.figure(figsize=(10, 5))
# plt.plot(y[4000:7000])
# plt.plot(audio[4000:7000])
# plt.show()
