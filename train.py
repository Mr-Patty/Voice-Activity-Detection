import os
import argparse
import torch
import json
import pickle

import torch.utils.data as data_utils
import soundfile as sf

from models.vad_models import LSTMModel
from datetime import datetime
from utils.processing import *
from os import listdir
from tqdm import tqdm


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self,sample, target):
        self.samples = sample
        self.targets = target

    def __getitem__(self, n):
        return self.samples[n].float(), torch.from_numpy(self.targets[n]).float()

    def __len__(self):
        return len(self.samples)


def trainModel(model, X, y, checkpoints_path, lr=1e-3, EPOCHS=10, batch_size=64, device='cuda', each=20, step_size=5,
               class_weight=[0.65, 0.35]):
    
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.BCELoss()
    
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=0.1)
    last_epoch = -1
    
    train_dataset = CustomDataset(X, y)
    use_cuda = device == 'cuda'
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    print('DATASET SIZE: {}'.format(len(train_dataset)))
    
    model.train()
    iterator = tqdm(range(EPOCHS), desc='epochs')
    print('START TRAINING')
    
    for epoch in iterator:
        try:
            if epoch <= last_epoch:
                continue

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=lambda x: data_processing(x),
                                    **kwargs)

            mean_loss = 0
            for batch_x, batch_y in train_loader:
                optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.reshape(batch_y.shape[0], -1, 1).to(device)

                output = model(batch_x)
                loss = loss_function(output, batch_y)

                loss.backward()
                optim.step()

                mean_loss += loss.detach().cpu().item()*len(batch_x)
            mean_loss /= len(train_dataset)
            scheduler.step()


            if epoch != 0 and epoch % each == 0:
                check_path = os.path.join(checkpoints_path, 'model_checpoint{}'.format(datetime.now().strftime("_%Y%m%d_%H%M%S")) + '.pt')
                torch.save({odel.state_dict(), check_path)
            iterator.set_postfix({'train': mean_loss})
        
        except KeyboardInterrupt:
            PATH = os.path.join(checkpoints_path, 'model_checpoint{}'.format(datetime.now().strftime("_%Y%m%d_%H%M%S")) + '_{}'.format(epoch) + '.pt')
            torch.save(model.state_dict(), PATH)
            return
   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_libri', default='LibriSpeech', help='Path to librispeech dataset')
    parser.add_argument('--path_nonspeech', default='Nonspeech', help='Path to nonspeech data')
    parser.add_argument('--checkpoints', default='checkpoints', help='Path to saved checkpoints')
    parser.add_argument('--number', default='50000', help='Number of train samples')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--batch_size', default='128', help='batch size') # занимает где-то 6гб видео памяти
    parser.add_argument('--epochs', default='30', help='Number of train epochs')
    namespace = parser.parse_args()
    argv = vars(namespace)
    
    
    with open('train.pkl', 'rb') as f:
        train_samples = pickle.load(f)   
        
    
    nonspeech_path = argv['path_nonspeech']
    nonspeech_dict = {}
    for i in listdir(nonspeech_path):
        if 'enc' in i:
            path = os.path.join(nonspeech_path, i)
            signal, samplerate = sf.read(path)
            nonspeech_dict[i] = signal
    
    with opent('waves.json', 'w') as f:
        waves = json.load(f)
    
    samples_number = int(argv['number'])
    train_X, train_y = get_train_samples(samples_number, train_samples, waves, nonspeech_dict)
                            
    checkpoints_path = argv['checkpoints']
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
        
    device = argv['device']
    batch_size = int(argv['batch_size'])
    epochs = int(argv['epochs'])
    model = LSTMModel(inpud_dim=40, hidden_dim=64, n_layers=2, dropout=0.5).float()
    
    # train model
    trainModel(model, train_X, train_y, checkpoints_path, lr=1e-3, EPOCHS=epochs, batch_size=batch_size, device=device, each=10, step_size=5)