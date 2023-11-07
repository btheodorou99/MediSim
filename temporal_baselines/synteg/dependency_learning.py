import os
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from config import MediSimConfig
from temporal_baselines.synteg.synteg import DependencyModel

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
config.batch_size = config.batch_size//8
MAX_VISIT_LENGTH = 280
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('./data/valDataset.pkl', 'rb'))

def get_batch(loc, batch_size, mode):
    if mode == 'train':
        ehr = train_ehr_dataset[loc:loc+batch_size]
    elif mode == 'valid':
        ehr = val_ehr_dataset[loc:loc+batch_size]
    else:
        ehr = test_ehr_dataset[loc:loc+batch_size]
    
    batch_ehr = np.ones((len(ehr), config.n_ctx, MAX_VISIT_LENGTH)) * (config.code_vocab_size + 2) # Initialize each code to the padding code
    batch_lens = np.ones((len(ehr), config.n_ctx, 1))
    batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
    batch_num_visits = np.zeros(len(ehr))
    for i, p in enumerate(ehr):
        visits = p['visits']
        for j, v in enumerate(visits):
            batch_mask[i,j+1] = 1
            batch_lens[i,j+1] = len(v) + 1
            for k, c in enumerate(v):
                batch_ehr[i,j+1,k+1] = c
        batch_ehr[i,j+1,len(v)+1] = config.code_vocab_size + 1 # Set the last code in the last visit to be the end record code
        batch_lens[i,j+1] = len(v) + 2
        batch_num_visits[i] = len(visits)
    
    batch_mask[:,1] = 1  # Set the mask to cover the labels
    batch_ehr[:,:,0] = config.code_vocab_size # Set the first code in each visit to be the start/class token
    batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
    return batch_ehr, batch_lens, batch_mask, batch_num_visits

def shuffle_training_data(train_ehr_dataset):
    np.random.shuffle(train_ehr_dataset)

EPOCHS = 50
LR = 1e-4
model = DependencyModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
if os.path.exists("./save/temporal/synteg_dependency_model"):
    print("Loading previous model")
    checkpoint = torch.load("./save/temporal/synteg_dependency_model", map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Train
global_loss = 1e10
for e in tqdm(range(EPOCHS)):
    shuffle_training_data(train_ehr_dataset)
    for i in range(0, len(train_ehr_dataset), config.batch_size):
        model.train()
    
        batch_ehr, batch_lens, batch_mask, _ = get_batch(i, config.batch_size, 'train')
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.int).to(device)
        batch_lens = torch.tensor(batch_lens, dtype=torch.int).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.int).to(device)
        
        optimizer.zero_grad()
        diagnosis_output = model(batch_ehr, batch_lens)
        labels = torch.sum(nn.functional.one_hot(batch_ehr.long(), num_classes=config.total_vocab_size), dim=2).float()
        labels = labels[..., 1:, :-1].contiguous()
        diagnosis_output = diagnosis_output * batch_mask
        labels = labels * batch_mask
        bce = nn.BCELoss()
        loss = bce(diagnosis_output[:,:,:-1], labels)
        loss.backward()
        optimizer.step()
        
        if i % (2500*config.batch_size) == 0:
            print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss))
        if i % (5000*config.batch_size) == 0:
            if i == 0:
                continue
    
            model.eval()
            with torch.no_grad():
                val_l = []
                for v_i in range(0, len(val_ehr_dataset), config.batch_size):
                    batch_ehr, batch_lens, batch_mask, _ = get_batch(v_i, config.batch_size, 'valid')
                    batch_ehr = torch.tensor(batch_ehr, dtype=torch.int).to(device)
                    batch_lens = torch.tensor(batch_lens, dtype=torch.int).to(device)
                    batch_mask = torch.tensor(batch_mask, dtype=torch.int).to(device)
                    
                    diagnosis_output = model(batch_ehr, batch_lens)
                    labels = torch.sum(nn.functional.one_hot(batch_ehr.long(), num_classes=config.total_vocab_size), dim=2).float()
                    labels = labels[..., 1:, :-1].contiguous()
                    diagnosis_output = diagnosis_output * batch_mask
                    labels = labels * batch_mask
                    bce = nn.BCELoss()
                    val_loss = bce(diagnosis_output[:,:,:-1], labels)
                    val_l.append(val_loss.item())
          
            cur_val_loss = np.mean(val_l)
            print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
            if cur_val_loss < global_loss:
                global_loss = cur_val_loss
                state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iteration': i
                    }
                torch.save(state, './save/temporal/synteg_dependency_model')
                print('\n------------ Save best model ------------\n')