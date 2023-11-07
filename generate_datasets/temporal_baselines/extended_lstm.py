import torch
import pickle
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
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

class LSTMBaseline(nn.Module):
    def __init__(self, config):
        super(LSTMBaseline, self).__init__()
        self.embedding_matrix = nn.Linear(config.total_vocab_size, config.n_embd, bias=False)
        self.lstm = nn.LSTM(input_size=config.n_embd,
                            hidden_size=config.n_embd,
                            num_layers=6,
                            batch_first=True, 
                            bidirectional=False)
        self.ehr_head = nn.Linear(config.n_embd, config.total_vocab_size)

    def forward(self, input_visits, ehr_labels=None, ehr_masks=None):
        embeddings = self.embedding_matrix(input_visits)
        hidden_states, _ = self.lstm(embeddings)
        code_logits = self.ehr_head(hidden_states)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        if ehr_labels is not None:    
            shift_probs = code_probs[..., :-1, :].contiguous()
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            if ehr_masks is not None:
                shift_probs = shift_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks

            bce = nn.BCELoss()
            loss = bce(shift_probs, shift_labels)
            return loss, shift_probs, shift_labels
        
        return code_probs
    
    def sample(self, input_visits, random=True):
        embeddings = self.embedding_matrix(input_visits)
        hidden_states, _ = self.lstm(embeddings)
        next_logits = self.ehr_head(hidden_states[:,-1:,:])
        sig = nn.Sigmoid()
        next_probs = sig(next_logits)
        if random:
            visit = torch.bernoulli(next_probs)
        else:
            visit = torch.round(next_probs)

        return visit

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

model = LSTMBaseline(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
checkpoint = torch.load('./save/temporal/lstm', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])


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
    with torch.no_grad():
        for _ in range(length-context.size(1)):
            context = torch.cat((context, model.sample(context, sample)), dim=1)
      
    return context

ehr_dataset = []
with torch.no_grad():
  for i in tqdm(range(0, len(dataset), config.sample_batch_size)):
    batch_ehr, _ = get_batch(i, config.sample_batch_size, dataset)
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    predictions = finish_sequence(model, config.n_ctx, batch_ehr[:,:2,:], batch_size=batch_ehr.size(0), device=device, sample=True)
    batch_converted_ehr = convert_ehr(predictions.detach().cpu().numpy())
    ehr_dataset += batch_converted_ehr
    
pickle.dump(ehr_dataset, open('./results/datasets_temporal_baselines/extended_lstm.pkl', 'wb'))