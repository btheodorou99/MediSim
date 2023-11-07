import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model import MediSimModel
from config import MediSimConfig
from torch.nn import DataParallel

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

train_ehr_dataset = pickle.load(open('data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('data/valDataset.pkl', 'rb'))
  
def get_batch(loc, batch_size, mode):
  if mode == 'train':
    ehr = train_ehr_dataset[loc:loc+batch_size]
  elif mode == 'valid':
    ehr = val_ehr_dataset[loc:loc+batch_size]
  else:
    ehr = test_ehr_dataset[loc:loc+batch_size]
    
  batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
  batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
  for i, p in enumerate(ehr):
    visits = p['visits']
    for j, v in enumerate(visits):
      batch_ehr[i,j+1][v] = 1
      batch_mask[i,j+1] = 1
    batch_ehr[i,len(visits),config.code_vocab_size+1] = 1 # Set the final visit to have the end token
    batch_ehr[i,len(visits)+1:,config.code_vocab_size+2] = 1 # Set the rest to the padded visit token
  
  batch_ehr[:,0,config.code_vocab_size] = 1 # Set the first visits to be the start token
  batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
  return batch_ehr, batch_mask

def shuffle_training_data(train_ehr_dataset):
  np.random.shuffle(train_ehr_dataset)

model = MediSimModel(config)
model = DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists("./save/medisim_model"):
  print("Loading previous model")
  checkpoint = torch.load('./save/medisim_model', map_location=torch.device(device))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])

# Train Model
global_loss = 1e10
bce = nn.BCELoss()
for e in tqdm(range(config.epoch)):
  shuffle_training_data(train_ehr_dataset)
  for i in tqdm(range(0, len(train_ehr_dataset), config.batch_size*2)):
    model.train()
    
    batch_ehr, batch_mask = get_batch(i, config.batch_size*2, 'train')
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
    
    optimizer.zero_grad()
    _, pred, labels = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
    loss = bce(pred, labels)
    loss.backward()
    optimizer.step()
    
    if i % (250*config.batch_size*2) == 0:
      print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss))
    if i % (500*config.batch_size*2) == 0:
      if i == 0:
        continue
    
      model.eval()
      with torch.no_grad():
        val_l = []
        for v_i in range(0, len(val_ehr_dataset), config.batch_size*2):
          batch_ehr, batch_mask = get_batch(v_i, config.batch_size*2, 'valid')
          batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
          batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
  
          val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
          val_l.append((val_loss).cpu().detach().numpy())
          
        cur_val_loss = np.mean(val_l)
        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
        if cur_val_loss < global_loss:
          global_loss = cur_val_loss
          state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': i
            }
          torch.save(state, './save/medisim_model')
          print('\n------------ Save best model ------------\n')