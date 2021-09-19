import argparse
import torch
import pickle
import onnx
import onnxruntime

import numpy as np
import torch.utils.data as data_utils

from sklearn.metrics import classification_report
from models.vad_models import LSTMModel
from utils.processing import *
from tqdm import tqdm


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self,sample, target):
        self.samples = sample
        self.targets = target
        

    def __getitem__(self, n):
        return self.samples[n].float(), torch.from_numpy(self.targets[n]).float()

    def __len__(self):
        return len(self.samples)

    
def testModel(model, test_X, test_y, device='cuda', model_type='torch'):
    
    
    test_dataset = CustomDataset(test_X, test_y)

    print('DATASET SIZE: {}'.format(len(test_dataset)))
    preds = []
    if model_type == 'torch':
        model.to(device)
        model.eval()
        for x, _ in tqdm(test_dataset):

            with torch.no_grad():
                output = model(torch.unsqueeze(x, 0).float().to(device))
                pred = output.cpu().detach().numpy()
                preds.extend(pred)
    else:
        ort_session = onnxruntime.InferenceSession("vad.onnx")
        for x, _ in tqdm(test_dataset):
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch.unsqueeze(x[0], 0))}
            ort_outs = ort_session.run(None, ort_inputs)
            preds.append(ort_outs[0].squeeze(0))
    return test_y, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='test', help='Path to test data')
    parser.add_argument('--checkpoint', default='data/vad.pt', help='Path to saved checkpoint')
    parser.add_argument('--number', default='1000', help='Number of test samples')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--type', default='torch', help='type of model onnx or torch')
    namespace = parser.parse_args()
    argv = vars(namespace)
    
    model_type = argv['type']
    if model_type == 'torch':
        model = LSTMModel(inpud_dim=40, hidden_dim=64, n_layers=2, dropout=0.5).float()
        model.load_state_dict(torch.load(argv['checkpoint']))
    elif model_type == 'onnx':
        onnx_model = onnx.load(argv['checkpoint'])
        onnx.checker.check_model(onnx_model)
        model = argv['checkpoint']
    else:
        print('model type must be onnx or torch')
        return
    
    max_samples = int(argv['number'])
    test_path = argv['test_path']
    device = argv['device']
    
    with open('dev.pkl', 'rb') as f:
        dev_samples = pickle.load(f) 
    
    test_X, test_y = getTestSamples(max_samples, test_path, dev_samples)
    target, pred = testModel(model, test_X, test_y, device, model_type=model_type)
    
    tmp_pred = np.vstack(pred).squeeze(1) > 0.5
    print(classification_report(np.hstack(target), tmp_pred.astype(int)))