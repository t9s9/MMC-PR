import numpy as np
import torch

import whisper
from whisper import DecodingOptions
from .outputs import TranscriptionResult, Segment


class WhisperAnnotator:
    """
    Annotate audio with speech-to-text transcription.
    """

    def __init__(self,
                 model_name: str = 'small',
                 device: str | torch.device | None = None,
                 download_root: str = None,
                 verbose: bool = False,
                 temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                 compression_ratio_threshold: float | None = 2.4,
                 logprob_threshold: float | None = -1.0,
                 no_speech_threshold: float | None = 0.6,
                 condition_on_previous_text: bool = True,
                 decode_options: DecodingOptions | None = None
                 ):
        """
        Initialize the annotator.

        :param model_name: Name of the model to use. See whisper.available_models() for a list of available models.

        :param device: Device to run the model on. If None, uses the default device, which is the GPU if available.

        :param download_root: Location to download the model to. If None, uses the default location.

        :param verbose: Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

        :param temperature: Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

        :param compression_ratio_threshold: If the gzip compression ratio is above this value, treat as failed

        :param logprob_threshold: If the average log probability over sampled tokens is below this value, treat as
        failed

        :param no_speech_threshold: If the no_speech probability is higher than this value AND the average log
        probability over sampled tokens is below `logprob_threshold`, consider the segment as silent

        :param condition_on_previous_text: if True, the previous output of the model is provided as a prompt for the
        next window; disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

        :param decode_options: Keyword arguments to construct `DecodingOptions` instances

        """
        self.model_name = model_name
        self.download_root = download_root
        self.verbose = verbose

        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.decode_options = decode_options or {}

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model = whisper.load_model(model_name, device=device, download_root=download_root)

    def __call__(self, audio: str | np.ndarray | torch.Tensor) -> TranscriptionResult:
        transciption = self.model.transcribe(audio,
                                             temperature=self.temperature,
                                             compression_ratio_threshold=self.compression_ratio_threshold,
                                             logprob_threshold=self.logprob_threshold,
                                             no_speech_threshold=self.no_speech_threshold,
                                             condition_on_previous_text=self.condition_on_previous_text,
                                             verbose=self.verbose,
                                             **self.decode_options)

        segments = [Segment(**segment) for segment in transciption['segments']]
        return TranscriptionResult(text=transciption['text'], language=transciption['language'], segments=segments)
