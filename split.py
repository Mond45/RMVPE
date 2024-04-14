import glob
from os import makedirs
from shutil import copy

paths = list(
    zip(glob.glob("MIR-1K\\Wavfile\\*.wav"), glob.glob("MIR-1K\\PitchLabel\\*.pv"))
)

train_paths = paths[: int(len(paths) * 0.8)]
valid_paths = paths[int(len(paths) * 0.8) :]


makedirs("dataset\\train", exist_ok=True)
makedirs("dataset\\test", exist_ok=True)

for file in train_paths:
    copy(file[0], "dataset\\train\\")
    copy(file[1], "dataset\\train\\")

for file in valid_paths:
    copy(file[0], "dataset\\test\\")
    copy(file[1], "dataset\\test\\")
