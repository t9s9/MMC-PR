from pathlib import Path

import librosa
from torch.utils.data import Dataset


class RAVDESS(Dataset):
    """
    RAVDESS dataset.

    :param root: Path to the dataset. It must contain the following files:
        - unzipped Audio_Song_Actors_01-24.zip
        - unzipped Audio_Speech_Actors_01-24.zip if include_songs is True
    :param include_songs: If True, include songs in the dataset.
    """

    def __init__(self, root: str, include_songs: bool = False):
        self.emotion_dict = {0: 'neutral',
                             1: 'calm',
                             2: 'happy',
                             3: 'sad',
                             4: 'angry',
                             5: 'fearful',
                             6: 'disgust',
                             7: 'surprised'}
        self.root = Path(root)
        self.speech_path = self.root / "Audio_Speech_Actors_01-24"
        self.song_path = self.root / "Audio_Song_Actors_01-24"
        self.include_songs = include_songs

        if not self.speech_path.exists():
            raise FileNotFoundError(f"Speech path {self.speech_path} does not exist.")
        if self.include_songs and not self.song_path.exists():
            raise FileNotFoundError(f"Song path {self.song_path} does not exist.")

        self.files = []
        self._load_data()

    def _load_data(self):
        self.files.extend(self.speech_path.rglob('*.wav'))
        if self.include_songs:
            self.files.extend(self.song_path.rglob('*.wav'))

    def extract_emotion(self, file: Path) -> int:
        return int(file.stem.split("-")[2]) - 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file = self.files[idx]
        audio, _ = librosa.load(file, sr=None, mono=True)

        emotion = self.extract_emotion(file)
        return audio, emotion
