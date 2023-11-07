import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from config import MediSimConfig
from temporal_baselines.synteg.synteg import Generator, DependencyModel

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
config.batch_size = config.batch_size//8
MAX_VISIT_LENGTH = 280
Z_DIM = 128

local_rank = -1
fp16 = False
if local_rank == -1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
if torch.cuda.is_available():
   torch.cuda.manual_seed_all(SEED)

NUM_TRAINING = 90000
full = pickle.load(open('./data/testDataset.pkl', 'rb'))
dataset = full[:NUM_TRAINING]

model = DependencyModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
generator = Generator(config).to(device)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=4e-6, weight_decay=1e-5)
checkpoint1 = torch.load("./save/temporal/synteg_dependency_model", map_location=torch.device(device))
checkpoint2 = torch.load("./save/temporal/synteg_condition_model", map_location=torch.device(device))
model.load_state_dict(checkpoint1['model'])
optimizer.load_state_dict(checkpoint1['optimizer'])
generator.load_state_dict(checkpoint2['generator'])
generator_optimizer.load_state_dict(checkpoint2['generator_optimizer'])

def get_batch(loc, batch_size, dataset):
    ehr = dataset[loc:loc+batch_size]
    
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

def finish_sequence(model, generator, length, context, context_lengths, batch_size, device='cuda'):
    ehr = context
    batch_ehr = torch.tensor(np.ones((batch_size, config.n_ctx, MAX_VISIT_LENGTH)) * (config.code_vocab_size + 2), dtype=torch.long).to(device)
    batch_ehr[:,0,0] = config.code_vocab_size
    batch_lens = torch.zeros((batch_size, config.n_ctx, 1), dtype=torch.int).to(device)
    batch_lens[:,0,0] = 1
    with torch.no_grad():
        for _ in range(length - context.size(1)):
            condition_vector = model(context, context_lengths, export=True)
            condition = condition_vector[:,-1]
            z = torch.randn((batch_size, Z_DIM)).to(device)
            visit = generator(z, condition)
            visit = torch.bernoulli(visit)
            toAdd = torch.tensor(np.ones((batch_size, 1, MAX_VISIT_LENGTH)) * (config.code_vocab_size + 2), dtype=torch.long).to(device)
            toAdd[:,0,0] = config.code_vocab_size
            toAddLengths = torch.ones(batch_size, 1, 1, dtype=torch.int).to(device)
            for i in range(batch_size):
                codes = torch.nonzero(visit[i], as_tuple=True)[0].tolist()
                toAdd[i,0,1:min(len(codes)+1,MAX_VISIT_LENGTH)] = torch.LongTensor(codes[:MAX_VISIT_LENGTH-1])
                toAddLengths[i] = min(len(codes)+1, MAX_VISIT_LENGTH)
            context = torch.cat((context, toAdd), dim=1)
            context_lengths = torch.cat((context_lengths, toAddLengths), dim=1)
        
    ehr = torch.sum(F.one_hot(context, num_classes=config.total_vocab_size), dim=2).float()
    return ehr

ehr_dataset = []
with torch.no_grad():
  for i in tqdm(range(0, len(dataset), config.batch_size)):
    batch_ehr, batch_lens, _, _ = get_batch(i, config.batch_size, dataset)
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.long).to(device)
    batch_lens = torch.tensor(batch_lens, dtype=torch.int).to(device)
    predictions = finish_sequence(model, generator, config.n_ctx, batch_ehr[:,:11,:], batch_lens[:,:11], batch_size=batch_ehr.size(0), device=device)
    batch_converted_ehr = convert_ehr(predictions.detach().cpu().numpy())
    ehr_dataset += batch_converted_ehr
    
pickle.dump(ehr_dataset, open('./results/datasets_temporal_baselines/extended_synteg.pkl', 'wb'))