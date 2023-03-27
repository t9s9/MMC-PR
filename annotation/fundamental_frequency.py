import librosa
import numpy as np
import torch
from tqdm import tqdm

import whisper
from .outputs import TranscriptionResult


class FundamentalFrequencyAnnotator:
    """
    Annotates the fundamental frequency of each segment in a transcription.
    """

    def __init__(self,
                 sample_rate: int = whisper.audio.SAMPLE_RATE,
                 verbose: bool = False,
                 **kwargs
                 ):
        """
        Initialize the fundamental frequency annotator.
        :param sample_rate: Sample rate of the audio.
        :param verbose: Verbosity.
        :param kwargs: Options for librosa.pyin. See: https://librosa.org/doc/main/generated/librosa.pyin.html
        """
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.pyin_kwargs = kwargs

    def get_fundamental_frequency(self, audio: np.ndarray, no_speech_value=0) -> np.ndarray:
        """
        Annotates the fundamental frequency of an audio segment.
        :param audio: Audio segment as a numpy array.
        :param no_speech_value: Value to use for no speech.
        :return: Array of fundamental frequencies.
        """
        f0, voiced_flag, voiced_prob = librosa.pyin(audio, sr=self.sample_rate,
                                                    fmin=librosa.note_to_hz('C2'),
                                                    fmax=librosa.note_to_hz('C7'),
                                                    fill_na=no_speech_value,
                                                    **self.pyin_kwargs)
        return f0

    def __call__(self,
                 audio: str | np.ndarray | torch.Tensor,
                 transcription: TranscriptionResult
                 ) -> TranscriptionResult:

        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=self.sample_rate, mono=True)

        for segment in tqdm(transcription.segments, desc='Computing fundamental frequency', disable=not self.verbose):
            audio_segment = audio[int(segment.start * self.sample_rate):int(segment.end * self.sample_rate)]
            segment.fundamental_frequency = self.get_fundamental_frequency(audio_segment)
        return transcription


class GenderClassification:
    """
    Simple gender classification based on f0.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def get_gender(self, fundamental_frequency: np.ndarray) -> str:
        # average the fundamental frequency over the segment but only consider voiced frames
        average_f0 = fundamental_frequency[fundamental_frequency.nonzero()].mean()
        if 85 <= average_f0 <= 155:
            return 'male'
        elif 165 <= average_f0 <= 255:
            return 'female'
        else:
            return 'unknown'

    def __call__(self, transcription: TranscriptionResult, **kwargs) -> TranscriptionResult:
        for segment in tqdm(transcription.segments, desc='Classify gender', disable=not self.verbose):
            segment.gender = self.get_gender(segment.fundamental_frequency)
        return transcription
