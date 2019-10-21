import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice

base_dir = '/home/maurapintor/data/speech/'

kw_list = ['yes', 'no', 'down', 'up']
n_imgs = 3
DURATION = 2
SAMPLING_RATE = 16000
SAMPLES = SAMPLING_RATE*DURATION
plt.figure(figsize=(n_imgs*5, len(kw_list)*5))
for kw_idx, kw in enumerate(kw_list):
    d = os.path.join(base_dir, kw)
    for i in range(n_imgs):
        dd = os.listdir(d)[i]
        plt.subplot(len(kw_list), n_imgs, i+1+n_imgs*kw_idx)
        plt.title(kw)
        dd = os.path.join(d, dd)
        audio, sr = librosa.load(dd, sr=SAMPLING_RATE)
        y = np.zeros([SAMPLES])
        duration = audio.shape[0]
        if duration < SAMPLES:
            y[:duration] = audio[:]
        S = librosa.feature.mfcc(y, sr=sr, hop_length=100, n_mfcc=200, power=1.0)
        print(S.max())
        # audio = librosa.feature.inverse.mfcc_to_audio(S, sr=sr, hop_length=100, power=1.0)
        # sounddevice.play(y, blocking=True, samplerate=sr)
        # sounddevice.play(audio, blocking=True, samplerate=sr)
        plt.imshow(S/255, cmap='gray_r')
plt.show()
#
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
