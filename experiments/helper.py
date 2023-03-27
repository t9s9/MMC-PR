import pickle
from pathlib import Path
import librosa
from pympi.Elan import Eaf

root = Path('/home/t9s9/Documents/Uni/MMC-PR/')
exp_names = list(map(lambda x: x.stem[:-2], (root / 'AudioOut').iterdir()))
exp_names.remove('V8')
sr = 441000


def load(exp):
    eaf = Eaf(root / f'SaGADateienIS/Grob{exp}.eaf')
    audio_file = root / f'AudioOut/{exp}K2.mp3'

    with open(root / f'annot/{exp}K2.pkl', 'rb') as f:
        transcription = pickle.load(f)

    with open(root / f'f0/{exp}K2.pkl', 'rb') as f:
        f0_raw = pickle.load(f)

    f0 = f0_raw[..., 1]
    times = f0_raw[..., 0] * 1000
    return eaf, transcription, f0, times
