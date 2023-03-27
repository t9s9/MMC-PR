import pickle

import numpy as np
import torch
from sklearn.base import is_classifier

from .outputs import TranscriptionResult


class SimpleEmotionAnnotator:
    """
    Simple emotion annotator using a pre-trained classifier from sklearn.
    """

    def __init__(self, weights: str, verbose: bool = False):
        """
        Initialize the annotator.
        :param weights: Path to the pickled model. See ´emotion_classification´ for details.
        :param verbose: Verbosity.
        """
        self.verbose = verbose
        with open(weights, 'rb') as f:
            self.model = pickle.load(f)

        if not is_classifier(self.model):
            raise ValueError('The provided model is not a classifier from sklean.')

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Predict the emotion of a given feature vector.
        :param x: Feature vector of shape (n_samples, n_features).
        :return: Probability of each emotion.
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.model.predict_proba(x)

    def __call__(self, transcription: TranscriptionResult, **kwargs) -> TranscriptionResult:
        # extract features and mean over time
        features = np.array([segment.audio_features.mean(0).numpy() for segment in transcription.segments])
        emotions = self.predict(features)
        transcription.scatter_segments(emotions, 'emotion_prob')
        transcription.scatter_segments(emotions.argmax(1), 'emotion')
        return transcription
