import json
import torch
import pickle
import random
import numpy as np
from sys import argv
from tqdm import tqdm
from model import MediSimModel
from config import MediSimConfig

process = 4
SEED = process
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

NUM_TRAINING = 7000
model = MediSimModel(config).to(device)
checkpoint = torch.load('./save/medisim_model_ss', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])

def sample_sequence(model, length, context, batch_size, device='cuda', sample=True):
  empty = torch.zeros((1,1,config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
  context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
  prev = context.unsqueeze(1)
  context = None
  with torch.no_grad():
    for _ in range(length-1):
      prev = model.sample(torch.cat((prev,empty), dim=1), sample)
      if torch.sum(torch.sum(prev[:,:,config.code_vocab_size+1], dim=1).bool().int(), dim=0).item() == batch_size:
        break
  ehr = prev.cpu().detach().numpy()
  prev = None
  empty = None
  return ehr

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

# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
stoken = np.zeros(config.total_vocab_size)
stoken[config.code_vocab_size] = 1
for i in tqdm(range(0, NUM_TRAINING, config.sample_batch_size)):
  bs = min([NUM_TRAINING-i, config.sample_batch_size])
  batch_synthetic_ehrs = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
  batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
  synthetic_ehr_dataset += batch_synthetic_ehrs

pickle.dump(synthetic_ehr_dataset, open(f'./results/datasets/fully_synthetic_ss.pkl', 'wb'))