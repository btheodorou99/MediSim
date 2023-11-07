import torch
import pickle
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
from model import MediSimModel
from config import MediSimConfig

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()

HIDDEN_DIM = 128
EMBEDDING_DIM = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

NUM_TRAINING = 7000
full = pickle.load(open('./data/testDataset.pkl', 'rb'))
dataset = full[:NUM_TRAINING]

class ImputationModel(nn.Module):
    def __init__(self, config):
        super(ImputationModel, self).__init__()
        self.embedding = nn.Linear(config.diagnosis_vocab_size, EMBEDDING_DIM)
        self.fc = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, config.procedure_vocab_size+config.medication_vocab_size)

    def forward(self, inputs):
        logits = self.output(torch.relu(self.fc(torch.relu(self.embedding(inputs)))))
        prob = torch.sigmoid(logits)
        return prob

model = ImputationModel(config).to(device)
state = torch.load(f'./save/modality/neural_network')
model.load_state_dict(state['model'])

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
      indices = np.nonzero(visit)
      end = False
      for idx in indices:
        idx = idx.item()
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
  labels_output = None
  visit = None
  visit_output = None
  indices = None
  
  return ehr_outputs

def finish_sequence(model, length, record, sample=True):
  with torch.no_grad():
    for i in range(1, length):
      record[:,i,config.diagnosis_vocab_size:config.diagnosis_vocab_size+config.procedure_vocab_size+config.medication_vocab_size] = torch.bernoulli(model(record[:,i,:config.diagnosis_vocab_size]))
      
  return record

ehr_dataset = []
with torch.no_grad():
  for i in tqdm(range(0, len(dataset), config.sample_batch_size)):
    # Get batch inputs
    batch_ehr, _ = get_batch(i, config.sample_batch_size, dataset)
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    predictions = finish_sequence(model, config.n_ctx, batch_ehr, sample=True)
    batch_converted_ehr = convert_ehr(predictions)
    ehr_dataset += batch_converted_ehr
    
pickle.dump(ehr_dataset, open('./results/datasets_modality_baselines/modality_added_neural_network.pkl', 'wb'))
    