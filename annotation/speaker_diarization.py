import librosa
import numpy as np
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

import whisper.audio
from .outputs import TranscriptionResult


class SpeakerDiarization:
    """
    Speaker diarization using the pyannote.audio pipeline to:
        1. Extract speaker embeddings from audio segments.
        2. Cluster embeddings using agglomerative clustering.
    """

    def __init__(self,
                 model_name: str = 'speechbrain/spkrec-ecapa-voxceleb',
                 num_speakers: int = 2,
                 device: str | torch.device | None = None,
                 verbose: bool = False):
        """
        Initializes the speaker diarization pipeline.
        :param model_name: Name of the speaker embedding model.
        :param num_speakers: Number of speakers to cluster.
        :param device: Device to use for inference.
        :param verbose: Verbose output.
        """
        self.model_name = model_name
        self.verbose = verbose
        self.num_speakers = num_speakers

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model = PretrainedSpeakerEmbedding(model_name, device=device)

    @torch.no_grad()
    def _embed_audio_segements(self,
                               audio: np.ndarray | torch.Tensor,
                               timestamps: list[tuple[float, float]]) -> torch.Tensor:
        """
        Embeds audio segments using the speaker embedding model.
        :param audio: Audio data as a numpy array or torch tensor.
        :param timestamps: (start, end) timestamps for each audio segment.
        :return: Embeddings for each audio segment.
        """
        embeddings = np.zeros(shape=(len(timestamps), self.model.dimension))

        for i, (start, end) in enumerate(tqdm(timestamps, desc='Embedding audio segments for speaker diarization',
                                              disable=not self.verbose)):
            audio_segment = audio[int(start * self.model.sample_rate):int(end * self.model.sample_rate)]
            if not isinstance(audio_segment, torch.Tensor):
                audio_segment = torch.from_numpy(audio_segment)

            audio_segment = audio_segment.reshape(1, 1, -1).to(self.device)
            embeddings[i] = self.model(audio_segment)

        return embeddings

    def get_speakers(self,
                     audio: str | np.ndarray | torch.Tensor,
                     timestamps: list[tuple[float, float]],
                     sample_rate: int = 16000
                     ) -> list[int]:
        """
        Returns a list of speaker labels for each audio segment.
        :param audio: Audio data as a numpy array or torch tensor.
        :param timestamps: (start, end) timestamps for each audio segment.
        :param sample_rate: Sample rate of the audio data.
        :return: List of speaker labels for each audio segment.
        """
        if isinstance(audio, str):
            audio, sample_rate = librosa.load(audio, sr=self.model.sample_rate, mono=True)

        if sample_rate != self.model.sample_rate:
            audio = librosa.resample(audio, sample_rate, self.model.sample_rate)

        embeddings = self._embed_audio_segements(audio, timestamps)

        clustering = AgglomerativeClustering(n_clusters=self.num_speakers).fit(embeddings)
        return list(clustering.labels_)

    def __call__(self,
                 audio: str | np.ndarray | torch.Tensor,
                 transcription: TranscriptionResult) -> TranscriptionResult:
        timestamps = [(segment.start, segment.end) for segment in transcription.segments]
        speakers = self.get_speakers(audio, timestamps, sample_rate=whisper.audio.SAMPLE_RATE)
        transcription.scatter_segments(speakers, 'speaker')
        return transcription
