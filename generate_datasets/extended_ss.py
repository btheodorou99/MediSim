import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from model import MediSimModel
from config import MediSimConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

NUM_TRAINING = 7000
full = pickle.load(open('./data/testDataset.pkl', 'rb'))
dataset = full[:NUM_TRAINING]
model = MediSimModel(config).to(device)
checkpoint = torch.load('./save/medisim_model_ss', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])

def get_batch(loc, batch_size, dataset):
  ehr = dataset[loc:loc+batch_size]
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

def convert_ehr(ehrs, index_to_code=None):
  ehr_outputs = []
  for i in range(len(ehrs)):
    ehr = ehrs[i]
    ehr_output = []
    for j in range(1, len(ehr)):
      visit = ehr[j]
      visit_output = []
      indices = np.nonzero(visit)[0]
      end = False
      for idx in indices:
        if idx < config.code_vocab_size: 
          visit_output.append(index_to_code[idx] if index_to_code is not None else idx)
        elif idx == config.code_vocab_size+1:
          end = True
      if visit_output != []:
        ehr_output.append(visit_output)
      if end:
        break
    ehr_outputs.append({'visits': ehr_output})
    
  ehr = None
  ehr_output = None
  visit = None
  visit_output = None
  indices = None
  
  return ehr_outputs

def finish_sequence(model, length, context, batch_size, device='cuda', sample=True):
  empty = torch.zeros((batch_size,1,config.total_vocab_size), device=device, dtype=torch.float32)
  prev = context
  with torch.no_grad():
    for _ in range(length-prev.size(1)):
      prev = model.sample(torch.cat((prev,empty), dim=1), sample)
      
  return prev

ehr_dataset = []
with torch.no_grad():
  for i in tqdm(range(0, len(dataset), config.sample_batch_size)):
    batch_ehr, _ = get_batch(i, config.sample_batch_size, dataset)
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    predictions = finish_sequence(model, config.n_ctx, batch_ehr[:,:2,:], batch_size=batch_ehr.size(0), device=device, sample=True)
    batch_converted_ehr = convert_ehr(predictions.detach().cpu().numpy())
    ehr_dataset += batch_converted_ehr
    
pickle.dump(ehr_dataset, open('./results/datasets/extended_ss.pkl', 'wb'))    