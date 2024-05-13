import pandas as pd
import numpy as np
from scipy.io.wavfile import write

f0 = pd.read_csv("f0.csv")


def nco(fcw, sr):
    phase = 0
    phase_result = []
    for fcw_samp in fcw:
        ph_step = 2 * np.pi * fcw_samp * 1 / sr
        phase += ph_step
        phase_result.append(phase)
    return np.cos(phase_result)


step_size = 20
SAMPLE_RATE = 16000
quantum = (SAMPLE_RATE * step_size) // 1000
# sample_rate = quantum * (1000 // step_size)

fcw = np.zeros(quantum * f0.shape[0])

for i in range(f0.shape[0]):
    fcw[quantum * (i) : quantum * (i + 1)] = f0["frequency"][i]

output = nco(fcw, SAMPLE_RATE)



write("j3k.wav", SAMPLE_RATE, output)