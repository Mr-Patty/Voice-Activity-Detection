import os
import random
import argparse
import pickle

import torch.utils.data as data_utils

from models.vad_models import LSTMModel
from datetime import datetime
from utils.processing import *


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
    use_cuda = device == 'cuda'

    print('DATASET SIZE: {}'.format(len(test_dataset)))
    
    if model_type == 'torch':
        model.to(device)
        model.eval()
        preds = []
        for x, _ in tqdm(test_dataset):

            with torch.no_grad():
                output = model(torch.unsqueeze(x, 0).float().to(device))
                pred = output.cpu().detach().numpy()
                preds.extend(pred)
    
    return test_y, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='test', help='Path to test data')
    parser.add_argument('--checkpoint', default='pretrainModel', help='Path to saved checkpoint')
    parser.add_argument('--number', default='1000', help='Number of test samples')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--type', default='onnx', help='type of model onnx or torch')
    namespace = parser.parse_args()
    argv = vars(namespace)
    
    model_type = argv['type']
    if model_type == 'torch':
        model = LSTMModel(inpud_dim=40, hidden_dim=64, n_layers=2, dropout=0.5).float()
        model.load_state_dict(torch.load(argv['checkpoint']))
    elif model_type == 'onnx':
        # todo
        return
    else:
        print('model type must be onnx or torch')
        return
    
    max_samples = int(argv['number'])
    test_path = argv['test_path']
    device = argv['device']
    
    with open('dev.pkl', 'rb') as f:
        dev_samples = pickle.load(f) 
    
    test_X, test_y = getTestSamples(max_samples, test_path, dev_samples)
    target, pred = testModel(model, test_X, test_y, device)
    
    tmp_pred = np.vstack(pred).squeeze(1) > 0.5
    print(classification_report(np.hstack(target), tmp_pred.astype(int)))