import torch
import pickle
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from config import MediSimConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

class RETAIN(nn.Module):
    def __init__(self, config):
        super(RETAIN, self).__init__()
        self.embedding = nn.Linear(config.total_vocab_size, config.n_embd, bias=False)
        self.a_rnn = nn.GRU(input_size=config.n_embd,
                            hidden_size=config.n_embd,
                            batch_first=True)
        self.a_att = nn.Linear(config.n_embd, 1)
        self.b_rnn = nn.GRU(input_size=config.n_embd,
                            hidden_size=config.n_embd,
                            batch_first=True)
        self.b_att = nn.Linear(config.n_embd, config.n_embd)
        self.fc = nn.Linear(config.n_embd, config.total_vocab_size)
        self.n_embd = config.n_embd

    def forward(self, input_visits, lengths, ehr_labels=None, ehr_masks=None):
        mask = torch.zeros(input_visits.size(0), input_visits.size(1), 1, device=input_visits.device)
        att_mask = torch.zeros(input_visits.size(0), input_visits.size(1), 1, device=input_visits.device)
        for i in range(len(lengths)):
            mask[i,:lengths[i]] = 1
            att_mask[i,lengths[i]:] = -99999

        visit_emb = self.embedding(input_visits)
        # for i in range(len(visit_emb)):
        #     visit_emb[i,:lengths[i]] = torch.flip(visit_emb[i,:lengths[i]], [0]) # Reverse input visit embeddings
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output_a, _ = self.a_rnn(packed_input)
        output_a, _ = pad_packed_sequence(packed_output_a, batch_first=True)
        packed_output_b, _ = self.b_rnn(packed_input)
        output_b, _ = pad_packed_sequence(packed_output_b, batch_first=True)
        true_len = output_b.size(1)
        output_a = torch.cat((output_a, torch.zeros(input_visits.size(0), input_visits.size(1) - output_a.size(1), self.n_embd).to(output_a.device)), 1)
        output_b = torch.cat((output_b, torch.zeros(input_visits.size(0), input_visits.size(1) - output_b.size(1), self.n_embd).to(output_b.device)), 1)
        
        progressive_c = torch.zeros(input_visits.size(0), input_visits.size(1), self.n_embd).to(input_visits.device)
        for i in range(1,true_len):
            alpha = F.softmax(self.a_att(output_a[:,:i,:]) + att_mask[:,:i,:], dim=1)
            beta = torch.tanh(self.b_att(output_b[:,:i,:]))

            c = alpha * beta * visit_emb[:,:i,:] * mask[:,:i,:]
            c = torch.sum(c * mask[:,:i,:], dim=1)
            progressive_c[:,i-1,:] = c
        
        patient_embeddings = self.fc(progressive_c)
        code_probs = torch.sigmoid(patient_embeddings)
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
        mask = torch.ones(input_visits.size(0), input_visits.size(1), 1, device=input_visits.device)
        att_mask = torch.zeros(input_visits.size(0), input_visits.size(1), 1, device=input_visits.device)
        lengths = torch.ones(input_visits.size(0)) * input_visits.size(1)

        visit_emb = self.embedding(input_visits)
        # for i in range(len(visit_emb)):
        #     visit_emb[i,:lengths[i]] = torch.flip(visit_emb[i,:lengths[i]], [0]) # Reverse input visit embeddings
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output_a, _ = self.a_rnn(packed_input)
        output_a, _ = pad_packed_sequence(packed_output_a, batch_first=True)
        packed_output_b, _ = self.b_rnn(packed_input)
        output_b, _ = pad_packed_sequence(packed_output_b, batch_first=True)
        output_a = torch.cat((output_a, torch.zeros(input_visits.size(0), input_visits.size(1) - output_a.size(1), self.n_embd).to(output_a.device)), 1)
        output_b = torch.cat((output_b, torch.zeros(input_visits.size(0), input_visits.size(1) - output_b.size(1), self.n_embd).to(output_b.device)), 1)
        
        alpha = F.softmax(self.a_att(output_a) + att_mask, dim=1)
        beta = torch.tanh(self.b_att(output_b))

        c = alpha * beta * visit_emb * mask
        c = torch.sum(c * mask, dim=1, keepdim=True)

        next_logits = self.fc(c)
        next_probs = torch.sigmoid(next_logits)
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
    batch_lens = batch_mask.sum(1).squeeze(-1).astype(np.int8)
    return batch_ehr, batch_mask, batch_lens

model = RETAIN(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
checkpoint = torch.load('./save/temporal/retain', map_location=torch.device(device))
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
        batch_ehr, _, _ = get_batch(i, config.batch_size, dataset)
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        
        predictions = finish_sequence(model, config.n_ctx, batch_ehr[:,:11,:], batch_size=batch_ehr.size(0), device=device, sample=True)
        batch_converted_ehr = convert_ehr(predictions.detach().cpu().numpy())
        ehr_dataset += batch_converted_ehr
    
pickle.dump(ehr_dataset, open('./results/datasets_temporal_baselines/extended_retain.pkl', 'wb'))