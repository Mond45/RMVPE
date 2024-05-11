# RMVPE

This repo is the Pytorch implementation of ["RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music"](https://arxiv.org/abs/2306.15412v2). 

## Dependencies

```
conda create -n rmvpe2 python=3.10 librosa=0.9.2 pytorch torchvision torchaudio scipy tqdm pandas tensorboard pytorch-cuda=11.8 -c pytorch -c nvidia
```
```
pip install mir_eval
```

## Dataset

[MIR-1K](https://www.dropbox.com/s/0jil7nsxrjkpr48/dataset.zip?dl=0)

Extract the archive into the same directory as `train.py`.

## Training

Simply run `python train.py`.
