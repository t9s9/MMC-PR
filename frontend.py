import pickle
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from annotation.outputs import TranscriptionResult, emotion_dict

st.set_page_config(layout="wide", page_title='Visualize Annotation')
root = Path("")
sr = 16000


@st.cache(allow_output_mutation=True)
def load_annot(filename: str) -> TranscriptionResult:
    with open((root / 'annot' / filename).with_suffix('.pkl'), 'rb') as f:
        return pickle.load(f)


@st.cache
def load_audio(filename: str):
    path = (root / 'AudioOut' / filename).with_suffix('.mp3')
    return librosa.load(path, sr=sr)[0]


@st.cache
def list_annot_files(root: Path) -> list[Path]:
    return list((root / 'annot').glob('*.pkl'))


st.sidebar.header('Settings')
file = st.sidebar.selectbox('Select file', list_annot_files(root), format_func=lambda x: x.name)

transcription = load_annot(file.name)
# audio = load_audio(file.name)

min_seg, max_seg = st.sidebar.slider('Select segment', 0, len(transcription.segments),
                                     (0, 10))
attr = st.sidebar.selectbox('Attributes', options=['gender', 'speaker', 'emotion'])

st.header('MMC-PR')
st.text(f'File: {file.name}')
st.text(f'Language: {transcription.language}')
st.text(f'Number of segments: {len(transcription.segments)}')
st.text(f'Duration: {transcription.segments[-1].end:.2f}s')

st.subheader('Transcription')
with st.expander('Show transcription'):
    st.write(transcription.text)

with st.expander('Show Annotation'):
    show_details = st.checkbox('Details')
    if show_details:
        st.text(transcription.details())
    else:
        st.text(transcription)

st.subheader('Visualization')
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
segments = transcription.segments[min_seg:max_seg]

f0 = np.concatenate([segment.fundamental_frequency for segment in segments])
f0[f0 == 0] = np.nan
times = librosa.times_like(f0, sr=16000) + segments[0].start
patches = []
line = ax.plot(times, f0, label='fundamental frequencies', color='black')

viridis = mpl.colormaps['rainbow'].resampled(len(emotion_dict))

for i, seg in enumerate(segments):
    ax.axvline(seg.end, color='tab:blue')
    ax.text(x=seg.start + 0.1, y=np.nanmax(f0), s=str(i + min_seg))
    if 'gender' in attr:
        ax.text(x=seg.start + 0.1, y=np.nanmin(f0), s=str(seg.gender))
    if 'speaker' in attr:
        print(seg.speaker)
        color = 'green' if seg.speaker == 1 else 'red'
        ax.axvspan(seg.start, seg.end, alpha=0.2, color=color)
    if attr == 'emotion':
        ax.axvspan(seg.start, seg.end, alpha=0.2, color=viridis(seg.emotion))

if attr == 'gender':
    ax.axhspan(85, 155, color='red', alpha=0.3, label='male')
    ax.axhspan(165, 255, color='green', alpha=0.3, label='female')
if attr == 'speaker':
    patches = [mpatches.Patch(color=color, label=f'Speaker {i}', alpha=0.2) for i, color in
               enumerate(['green', 'red'])]
    patches.append(line[0])

if attr == 'emotion':
    patches = [mpatches.Patch(color=viridis(i), label=f'{e}', alpha=0.2) for i, e in
               emotion_dict.items()]
    patches.append(line[0])

ax.set_xlim(left=segments[0].start, right=max(times))
ax.set_ylabel('frequency in Hertz')
ax.set_xlabel('Time (s)')

if patches:
    ax.legend(handles=patches)
else:
    plt.legend()
st.pyplot(fig)

st.text('\n'.join([f"{i + min_seg: <4} " + str(seg.text) for i, seg in enumerate(segments)]))
