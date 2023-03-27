import numpy as np
import torch

from .outputs import TranscriptionResult


class Pipeline:
    def __init__(self, *args, verbose: bool = False):
        self.verbose = verbose
        self.steps = args
        for step in self.steps:
            step.verbose = verbose

    def __call__(self, audio: str | np.ndarray | torch.Tensor) -> TranscriptionResult:
        transcription = self.steps[0](audio)
        for step in self.steps[1:]:
            transcription = step(transcription=transcription, audio=audio)
        return transcription
