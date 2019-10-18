import h5py
import numpy as np
filename = "ds.hdf5"

with h5py.File(filename, 'r') as f:
    # List all groups
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])

for i in range(4):
    d = data[i]
    # import matplotlib.pyplot as plt
    # plt.imshow(d)
    # plt.show()

    import librosa
    import sounddevice as sd
    hl = 16000 // 1000 * 10
    audio = librosa.feature.inverse.mfcc_to_audio(d, n_mels=40, sr=16000, hop_length=hl,
                                                  fmin=20, fmax=4000, n_fft=480)
    sd.play(audio, 16000, blocking=True)