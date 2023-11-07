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
device = 'cpu'
NUM_TRAINING = 7000
MAX_VISIT_LENGTH = 280
full = pickle.load(open('./data/testDataset.pkl', 'rb'))
dataset = full[:NUM_TRAINING]

class SingleVisitTransformer(nn.Module):
    def __init__(self):
        super(SingleVisitTransformer, self).__init__()
        encoderLayer = nn.TransformerEncoderLayer(config.n_embd, config.n_head, 
                        dim_feedforward=config.n_embd, dropout=0.1, activation="relu", 
                        layer_norm_eps=1e-08, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoderLayer, 2)

    def forward(self, code_embeddings, visit_lengths):
        bs, vs, cs, ed = code_embeddings.shape
        mask = torch.ones((bs, vs, cs)).to(code_embeddings.device)
        for i in range(bs):
            for j in range(vs):
                mask[i,j,:visit_lengths[i,j]] = 0
        visits = torch.reshape(code_embeddings, (bs*vs,cs,ed))
        mask = torch.reshape(mask, (bs*vs,cs))
        encodings = self.transformer(visits, src_key_padding_mask=mask)
        encodings = torch.reshape(encodings, (bs,vs,cs,ed))
        visit_representations = encodings[:,:,0,:]
        return visit_representations

class CONAN(nn.Module):
    def __init__(self, config):
        super(CONAN, self).__init__()
        self.embedding = nn.Embedding(config.code_vocab_size+2, config.n_embd)
        self.visit_att = SingleVisitTransformer()
        self.lstm = nn.LSTM(input_size=config.n_embd,
                            hidden_size=config.n_embd,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.att_weights = nn.Linear(config.n_embd, config.n_embd)
        self.att_weights2 = nn.Linear(config.n_embd, 1, bias=False)
        self.fc = nn.Linear(config.n_embd, config.n_embd)
        self.output = nn.Linear(config.n_embd, config.total_vocab_size)

    def forward(self, input_visits, lengths, visit_lengths, ehr_labels=None, ehr_masks=None):
        mask = torch.zeros(input_visits.size(0), input_visits.size(1), 1, device=input_visits.device)
        for i in range(len(lengths)):
            mask[i,lengths[i]:] = -99999
        inputs = self.embedding(input_visits) # bs * visits * codes * embedding_dim
        visit_emb = self.visit_att(inputs, visit_lengths) # bs * visits * embedding_dim
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        true_len = output.size(1)
        output = torch.cat((output, torch.zeros(input_visits.size(0), input_visits.size(1) - output.size(1), config.n_embd).to(output.device)), 1)

        u_vector = self.att_weights2(torch.tanh(self.att_weights(output)))
        u_vector = u_vector + mask

        progressive_output = torch.zeros(input_visits.size(0), input_visits.size(1), config.n_embd).to(input_visits.device)
        for i in range(1, true_len):
            alpha = F.softmax(u_vector[:,:i,:], 1)
            curr_output = (output[:,:i,:] * alpha).sum(1)
            progressive_output[:,i-1,:] = curr_output

        patient_embeddings = self.output(torch.relu(self.fc(progressive_output)))
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
        
    def sample(self, input_visits, visit_lengths, random=True):
        mask = torch.zeros(input_visits.size(0), input_visits.size(1), 1, device=input_visits.device)
        lengths = torch.ones(input_visits.size(0)).to('cpu') * input_visits.size(1)
        inputs = self.embedding(input_visits) # bs * visits * codes * embedding_dim
        visit_emb = self.visit_att(inputs, visit_lengths) # bs * visits * embedding_dim
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        true_len = output.size(1)
        output = torch.cat((output, torch.zeros(input_visits.size(0), input_visits.size(1) - output.size(1), config.n_embd).to(output.device)), 1)

        u_vector = self.att_weights2(torch.tanh(self.att_weights(output)))
        u_vector = u_vector + mask

        alpha = F.softmax(u_vector, 1)
        output = (output * alpha).sum(1)

        next_logits = self.output(torch.relu(self.fc(output)))
        next_probs = torch.sigmoid(next_logits)

        if random:
            visit = torch.bernoulli(next_probs)
        else:
            visit = torch.round(next_probs)

        return visit

def get_batch(loc, batch_size, dataset):
    ehr = dataset[loc:loc+batch_size]
    
    batch_ehr = np.ones((len(ehr), config.n_ctx, MAX_VISIT_LENGTH)) * (config.code_vocab_size + 1) # Initialize everything to the pad code
    batch_lens = np.zeros(len(ehr), np.int8)
    batch_visit_lens = np.zeros((len(ehr), config.n_ctx), np.int8)
    batch_ehr_label = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
    batch_ehr_mask = np.zeros((len(ehr), config.n_ctx, 1))
    for i, p in enumerate(ehr):
        visits = p['visits']
        batch_lens[i] = len(visits)
        for j, v in enumerate(visits):
            batch_visit_lens[i,j+1] = len(v) + 1
            for k, c in enumerate(v):
                batch_ehr[i,j+1,k+1] = c
            batch_ehr_label[i,j+1][v] = 1
            batch_ehr_mask[i,j+1] = 1
        batch_ehr_label[i,len(visits),config.code_vocab_size+1] = 1 # Set the final visit to have the end token

    batch_ehr[:,:,0] = config.code_vocab_size # Set the first code in each visit to be the start/class token
    batch_ehr_mask = batch_ehr_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return

    return batch_ehr, batch_lens, batch_visit_lens, batch_ehr_label, batch_ehr_mask



model = CONAN(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
checkpoint = torch.load('./save/temporal/conan', map_location=torch.device(device))
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

def finish_sequence(model, length, model_context, model_visit_lengths, record_context, batch_size, device='cuda', sample=True):
    with torch.no_grad():
        for _ in range(length-record_context.size(1)):
            record_context = torch.cat((record_context, model.sample(model_context, model_visit_lengths, sample).unsqueeze(1)), dim=1)
            new_model_context = torch.ones(model_context.size(0), 1, MAX_VISIT_LENGTH, dtype=torch.int).to(device) * (config.code_vocab_size + 1)
            new_model_context[:,0,0] = config.code_vocab_size
            new_visit_lengths = np.zeros((model_visit_lengths.shape[0], 1), np.int8)
            for i in range(record_context.size(0)):
                present_codes = torch.nonzero(record_context[i,-1,:], as_tuple=True)[0].tolist()
                new_model_context[i,0,1:min(len(present_codes), MAX_VISIT_LENGTH-1)+1] = torch.IntTensor(present_codes[:MAX_VISIT_LENGTH-1]).to(device)
                new_visit_lengths[i] = len(present_codes) + 1
            model_context = torch.cat((model_context, new_model_context), dim=1)
            model_visit_lengths = np.concatenate((model_visit_lengths, new_visit_lengths), axis=1)

    return record_context

ehr_dataset = []
with torch.no_grad():
    for i in tqdm(range(0, len(dataset), config.sample_batch_size)):
        batch_ehr, _, batch_visit_lens, batch_ehr_label, _ = get_batch(i, config.batch_size, dataset)
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.int).to(device)
        batch_ehr_label = torch.tensor(batch_ehr_label, dtype=torch.float32).to(device)
        
        predictions = finish_sequence(model, config.n_ctx, batch_ehr[:,:11,:], batch_visit_lens[:,:11], batch_ehr_label[:,:11,:], batch_size=batch_ehr.size(0), device=device, sample=True)
        batch_converted_ehr = convert_ehr(predictions.detach().cpu().numpy())
        ehr_dataset += batch_converted_ehr
    
pickle.dump(ehr_dataset, open('./results/datasets_temporal_baselines/extended_conan.pkl', 'wb'))