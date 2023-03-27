from .outputs import TranscriptionResult, Segment
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import librosa
import matplotlib.patches as mpatches
from librosa.display import specshow, waveshow


def plot_wave(wave: np.ndarray, sample_rate: int, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax = waveshow(wave, sr=sample_rate, ax=ax, **kwargs)
    return ax


def plot_spectrogram(spectrogram: np.ndarray, sample_rate: int, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax = specshow(spectrogram, sr=sample_rate, ax=ax, **kwargs)
    return ax


def plot_gender(segments: list[Segment], ax: plt.Axes = None, **kwargs) -> plt.Axes:
    gender_c = {'male': 'blue', 'female': 'pink', 'unknown': 'grey'}

    if ax is None:
        ax = plt.gca()
    f0 = np.concatenate([segment.fundamental_frequency for segment in segments])
    f0[f0 == 0] = np.nan
    times = librosa.times_like(f0, sr=16000)

    ax.plot(times, f0, label='fundamental frequencies', color='black')
    for i, seg in enumerate(segments):
        ax.axvspan(seg.start, seg.end, color=gender_c[seg.gender], alpha=0.2)
        ax.axvline(seg.end, color='black')

    ax.axhspan(85, 155, color='red', alpha=0.3, label='male')
    ax.axhspan(165, 255, color='green', alpha=0.3, label='female')

    ax.set_xlim(left=0, right=max(times))

    ax.set_ylabel('frequency in Hertz')
    ax.set_xlabel('Time (s)')

    patches = [mpatches.Patch(color=c, label=f'predicted {gender}', alpha=0.2) for gender, c in gender_c.items()]

    ax.set_title('Fundamental frequencies + Gender')
    return ax, patches


def plot_speaker(segments: list[Segment], ax: plt.Axes = None, **kwargs) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    f0 = np.concatenate([segment.fundamental_frequency for segment in segments])
    f0[f0 == 0] = np.nan
    times = librosa.times_like(f0, sr=16000)

    ax.plot(times, f0, label='fundamental frequencies')
    for i, seg in enumerate(segments):
        color = 'blue' if seg.speaker == 'A' else 'pink'
        ax.axvspan(seg.start, seg.end, color=color, alpha=0.2)
        ax.axvline(seg.end, color='black')

    ax.axhline(85, linestyle='--', color='blue', alpha=1, label='A')
    ax.axhline(155, linestyle='--', color='blue', alpha=1, label='A')
    ax.axhline(165, linestyle='--', color='pink', alpha=1, label='B')
    ax.axhline(255, linestyle='--', color='pink', alpha=1, label='B')

    ax.set_xlim(left=0, right=max(times))

    ax.set_ylabel('frequency in Hertz')
    ax.set_xlabel('Time (s)')
    return ax
