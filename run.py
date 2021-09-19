import argparse
import torch
import pickle
import onnx
import onnxruntime

import numpy as np
import soundfile as sf

from sklearn.metrics import classification_report
from models.vad_models import LSTMModel
from utils.processing import *
from tqdm import tqdm


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_audio', help='Path to test audio file')
    parser.add_argument('--checkpoint', default='data/vad.pt', help='Path to saved checkpoint')
    parser.add_argument('--device', default='cuda', help='device for pytorch model')
    parser.add_argument('--type', default='torch', help='type of model onnx or torch')
    parser.add_argument('--threshold', default='0.7', help='threshold for model')
    namespace = parser.parse_args()
    argv = vars(namespace)

    model_type = argv['type']
    if model_type == 'torch':
        model = LSTMModel(inpud_dim=40, hidden_dim=64, n_layers=2, dropout=0.5).float()
        model.load_state_dict(torch.load(argv['checkpoint']))
        model.eval()

    elif model_type == 'onnx':
        onnx_model = onnx.load(argv['checkpoint'])
        onnx.checker.check_model(onnx_model)
        model = argv['checkpoint']
    else:
        print('model type must be onnx or torch')

    audio_file = argv['test_audio']
    device = argv['device']

    if device == 'cuda':
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    thr = float(argv['threshold'])

    data, samplerate = sf.read(argv['test_audio'])
    features = process_audio(data)
    if model_type == 'torch':
        model.to(device)
        with torch.no_grad():
            output = to_numpy(model(torch.unsqueeze(features, 0).float().to(device))).reshape(features.shape[0])
    elif model_type == 'onnx':
        ort_session = onnxruntime.InferenceSession(model, providers=providers)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch.unsqueeze(features, 0))}
        ort_outs = ort_session.run(None, ort_inputs)
        output = ort_outs[0].reshape(features.shape[0])


    output = (output > thr).astype(int)
    print(output)
