import librosa
import numpy as np
import torch

from src.constants import SAMPLE_RATE, WINDOW_LENGTH

HOP_LENGTH = 20
HOP_LENGTH = int(HOP_LENGTH / 1000 * SAMPLE_RATE)
SAMPLE_RATE = 16000

audio, _ = librosa.load("input/j3k.wav", sr=SAMPLE_RATE)
audio_l = len(audio)
audio = np.pad(audio, WINDOW_LENGTH // 2, mode="reflect")

audio = torch.from_numpy(audio).float()
audio_steps = audio_l // HOP_LENGTH + 1

seq_len = 2.55
seq_len = int(seq_len * SAMPLE_RATE)

data = []

n_steps = seq_len // HOP_LENGTH + 1
for i in range(audio_l // seq_len):
    begin_t = i * seq_len
    end_t = begin_t + seq_len + WINDOW_LENGTH
    begin_step = begin_t // HOP_LENGTH
    end_step = begin_step + n_steps
    data.append(
        audio[begin_t:end_t],
    )
data.append(
    audio[-seq_len - WINDOW_LENGTH :],
)

