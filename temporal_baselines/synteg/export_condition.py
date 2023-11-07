import torch
import random
import pickle
import numpy as np
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
    
LR = 1e-4
model = DependencyModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
checkpoint = torch.load("./save/temporal/synteg_dependency_model", map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

condition_dataset = []
for i in tqdm(range(0, len(train_ehr_dataset), config.batch_size)):
    model.train()
    
    batch_ehr, batch_lens, _, batch_num_visits = get_batch(i, config.batch_size, 'train')
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.int).to(device) # bs * visit * code
    batch_lens = torch.tensor(batch_lens, dtype=torch.int).to(device) # bs * visit
    condition_vector = model(batch_ehr, batch_lens, export=True) # bs * visit * 256
    batch_ehr = batch_ehr.detach().cpu().numpy()
    condition_vector = condition_vector.detach().cpu().numpy()

    for b, num_visits in enumerate(batch_num_visits-1):
        for v in range(int(num_visits+1)):
            ehr_tmp = batch_ehr[b, v+1, :]
            condition_vector_tmp = condition_vector[b, v, :]
            datum = {"ehr": ehr_tmp, "condition": condition_vector_tmp}
            condition_dataset.append(datum)

pickle.dump(condition_dataset, open("./temporal_baselines/synteg/data/conditionDataset.pkl", "wb"))