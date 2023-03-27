import pickle
from dataclasses import dataclass, field

import numpy as np

from whisper.utils import format_timestamp

emotion_dict = {0: 'neutral',
                1: 'calm',
                2: 'happy',
                3: 'sad',
                4: 'angry',
                5: 'fearful',
                6: 'disgust',
                7: 'surprised'}


@dataclass(frozen=False)
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str = ""
    tokens: list[int] = field(default_factory=list)
    audio_features: np.ndarray = field(default_factory=np.ndarray)
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan
    gender: str = None
    speaker: int = -1
    emotion: int = -1
    emotion_prob: np.ndarray = None
    fundamental_frequency: np.ndarray = None

    def __str__(self) -> str:
        return f"[{format_timestamp(self.start)} --> {format_timestamp(self.end)}] {self.text}"

    def details(self) -> str:
        speaker_str = f" [Speaker: {self.speaker}]" if self.speaker >= 0 else ""
        emotion_str = f" [{emotion_dict[self.emotion]: <9} ({self.emotion_prob.max():.2f})]" if self.emotion >= 0 else ""
        gender_str = f" [{self.gender: <7}]" if self.gender is not None else ""
        info_string = f"[{format_timestamp(self.start)} --> {format_timestamp(self.end)}]{speaker_str}{gender_str}{emotion_str}:"
        return f"{info_string} {self.text}"


@dataclass(frozen=False)
class TranscriptionResult:
    language: str
    segments: list[Segment]
    text: str = ""

    def scatter_segments(self, values: list | tuple | np.ndarray, field_name: str) -> None:
        assert len(values) == len(self.segments)
        for segment, value in zip(self.segments, values):
            setattr(segment, field_name, value)

    def save(self, path: str) -> bool:
        """
        Saves annotation to disk using pickle.

        :param path: Path to save annotation to.
        :return: True if successful, False otherwise.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return True

    def __str__(self) -> str:
        return '\n'.join([str(segment) for segment in self.segments])

    def details(self) -> str:
        return '\n'.join([segment.details() for segment in self.segments])
