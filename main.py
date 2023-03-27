from pathlib import Path

from annotation import *
from annotation.outputs import TranscriptionResult

if __name__ == '__main__':
    emotion_clf_path = "emotion_classification/emotion_clf.pkl"

    whisper_decoding_opt = DecodingOptions(
        fp16=True,
        language='de',
    )

    pipe = Pipeline(
        WhisperAnnotator(model_name='medium', decode_options=whisper_decoding_opt),
        SpeakerDiarization(num_speakers=2, device='cpu'),
        SimpleEmotionAnnotator(emotion_clf_path),
        FundamentalFrequencyAnnotator(),
        GenderClassification(),
        verbose=True
    )

    result: TranscriptionResult = pipe('audio.mp3')
    result.save('out.pkl')
