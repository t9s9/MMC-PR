import pickle

import torch
from sklearn import preprocessing, pipeline, neural_network
from torch.utils.data import DataLoader
from tqdm import tqdm

import whisper
from dataset import RAVDESS


def embed_audio(model, dataset_root, out_file):
    dataset = RAVDESS(root=dataset_root,
                      include_songs=True)
    dataloader = DataLoader(dataset, num_workers=8, batch_size=None)

    X, y = [], []
    for audio, label in tqdm(dataloader):
        outputs = model.transcribe(audio, verbose=False, language='en', fp16=True)
        X.append(outputs['segments'][0]['audio_features'].mean(0).cpu())
        y.append(label)

    X = torch.stack(X)
    y = torch.stack(y)

    torch.save(X, out_file + "_X.pt")
    torch.save(y, out_file + "_y.pt")


def train_clf(in_file, out_file):
    X = torch.load(in_file.format('X')).numpy()
    y = torch.load(in_file.format('y')).numpy()

    pipe = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        neural_network.MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), learning_rate='adaptive')
    ).fit(X, y)

    with open(out_file, 'wb') as f:
        pickle.dump(pipe, f)


if __name__ == '__main__':
    device = 'cuda'

    model = whisper.load_model("medium").to(device)

    dataset_root = ""  # todo
    in_file = "RAVDESS_whisper_medium"
    out_file = 'emotion_clf.pkl'

    embed_audio(model, dataset_root="/media/t9s9/SSD_ubuntu/datasets/RAVDESS/", out_file=in_file)
    train_clf(in_file, out_file)
