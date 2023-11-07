import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model import MediSimModel
from config_note import MediSimConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

train_ehr_dataset = pickle.load(open('data_note/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('data_note/valDataset.pkl', 'rb'))
MAX_ITERATION = 25
PATIENCE = 5
QUALITY_EPOCHS = 10
EMBEDDING_DIM = 256
LSTM_DIM = 128
QUALITY_BATCH_SIZE = 512
VAL_SPLIT = 0.1
PROB_REPLACE = 0.75
NUM_SAMPLE = 10000
NUM_SIMILAR = 25
  
def get_batch(dataset, loc, batch_size):
  ehr = dataset[loc:loc+batch_size]
  batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
  batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
  for i, p in enumerate(ehr):
    visits = p['visits']
    for j, v in enumerate(visits):
      batch_ehr[i,j+1][v[0]] = 1
      batch_mask[i,j+1] = 1
    batch_ehr[i,len(visits),config.code_vocab_size+1] = 1 # Set the final visit to have the end token
    batch_ehr[i,len(visits)+1:,config.code_vocab_size+2] = 1 # Set the rest to the padded visit token
  
  batch_ehr[:,0,config.code_vocab_size] = 1 # Set the first visits to be the start token
  batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
  return batch_ehr, batch_mask

def shuffle_training_data(train_ehr_dataset):
  np.random.shuffle(train_ehr_dataset)

def train_model(model, optimizer, train_dataset, val_dataset, global_loss):
  # Train Model
  curr_patience = 0
  for e in tqdm(range(config.epoch)):
    if curr_patience >= PATIENCE:
      break
    shuffle_training_data(train_ehr_dataset)
    model.train()
    train_losses = []
    for i in range(0, len(train_dataset), config.batch_size):
      batch_ehr, batch_mask = get_batch(train_dataset, i, config.batch_size)
      batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
      batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
      
      optimizer.zero_grad()
      loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.cpu().detach().item())
      
    print("Epoch %d Training Loss:%.6f"%(e, np.mean(train_losses)))
    model.eval()
    with torch.no_grad():
      val_l = []
      for v_i in range(0, len(val_dataset), config.batch_size):
        batch_ehr, batch_mask = get_batch(val_dataset, v_i, config.batch_size)
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)

        val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
        val_l.append(val_loss.cpu().detach().numpy())
        
      cur_val_loss = np.mean(val_l)
      print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
      if cur_val_loss < global_loss:
        curr_patience = 0
        global_loss = cur_val_loss
        state = {
              'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'iteration': 0
          }
        torch.save(state, './save/medisim_model_note_ss')
        print('\n------------ Save best model ------------\n')
      else:
        curr_patience += 1

  model.load_state_dict(torch.load('./save/medisim_model_note_ss')['model'])
  return model, global_loss

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

def generate_dataset(model, amount):
  synthetic_dataset = []
  stoken = np.zeros(config.total_vocab_size)
  stoken[config.code_vocab_size] = 1
  for i in range(0, amount, config.sample_batch_size):
    bs = min([amount-i, config.sample_batch_size])
    batch_synthetic_ehrs = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
    batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
    synthetic_dataset += batch_synthetic_ehrs
  return synthetic_dataset

class QualityModel(nn.Module):
    def __init__(self, config):
        super(QualityModel, self).__init__()
        self.embedding = nn.Linear(config.total_vocab_size, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=LSTM_DIM,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2*LSTM_DIM, EMBEDDING_DIM)
        self.output = nn.Linear(EMBEDDING_DIM, 1)

    def forward(self, input_visits, lengths):
        visit_emb = self.embedding(input_visits)
        visit_emb = self.dropout(visit_emb)
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :LSTM_DIM]
        out_reverse = output[:, 0, LSTM_DIM:]
        out_combined = torch.cat((out_forward, out_reverse), 1)

        patient_embedding = self.output(torch.relu(self.fc(out_combined)))
        patient_embedding = torch.squeeze(patient_embedding, 1)
        prob = torch.sigmoid(patient_embedding)
        
        return prob

def get_quality_batch(dataset, loc, batch_size):
  ehr = dataset[loc:loc+batch_size]
  batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
  batch_labels = np.array([p[1] for p in ehr])
  batch_lens = np.zeros(len(ehr))
  for i, p in enumerate(ehr):
    visits = p[0]['visits']
    batch_lens[i] = len(visits) + 1
    for j, v in enumerate(visits):
      batch_ehr[i,j+1][v[0]] = 1 # set the visit codes  
    batch_ehr[i,len(visits),config.code_vocab_size+1] = 1 # Set the final visit to have the end token

  batch_ehr[:,0,config.code_vocab_size] = 1 # Set the first visits to be the start token
  return batch_ehr, batch_labels, batch_lens

def train_quality_model(quality_dataset):
    quality_loss = 1e10
    model = QualityModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    np.random.shuffle(quality_dataset)
    num_val = int(len(quality_dataset) * VAL_SPLIT)
    train_dataset = quality_dataset[:-num_val]
    val_dataset = quality_dataset[-num_val:]
    bce = nn.BCELoss()
    for e in range(QUALITY_EPOCHS):
      np.random.shuffle(train_dataset)
      train_losses = []
      for i in range(0, len(train_dataset), QUALITY_BATCH_SIZE):
        model.train()
        batch_ehr, batch_labels, batch_lens = get_quality_batch(train_dataset, i, QUALITY_BATCH_SIZE)
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        prob = model(batch_ehr, batch_lens)
        loss = bce(prob, batch_labels)
        train_losses.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()

      cur_train_loss = np.mean(train_losses)
      print("Epoch %d Training Loss:%.5f"%(e, cur_train_loss))
    
      model.eval()
      with torch.no_grad():
        val_losses = []
        for v_i in range(0, len(val_dataset), QUALITY_BATCH_SIZE):
          batch_ehr, batch_labels, batch_lens = get_quality_batch(val_dataset, v_i, QUALITY_BATCH_SIZE)
          batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
          batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
          prob = model(batch_ehr, batch_lens)
          val_loss = bce(prob, batch_labels)
          val_losses.append(val_loss.cpu().detach().numpy())
        cur_val_loss = np.mean(val_losses)
        print("Epoch %d Validation Loss:%.5f"%(e, cur_val_loss))
        if cur_val_loss < quality_loss:
          quality_loss = cur_val_loss
          state = {
              'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'iteration': 0
          }
          torch.save(state, f'./save/quality_model_note')
          print('\n------------ Save best model ------------\n')

    model.load_state_dict(torch.load('./save/quality_model_note')['model'])
    return model

def getLens(data):
  batch_lens = np.ones(len(data)) * config.n_ctx
  stop_tokens = data[:,:,config.code_vocab_size+1]
  for i in range(len(data)):
    vals = stop_tokens[i].nonzero()
    if vals.numel() > 0:
      batch_lens[i] = vals.min().cpu().item() + 1
  return batch_lens

def train_rl(model, quality_model, val_dataset, global_loss, iteration):
  # Train Model
  curr_patience = 0
  num_batch = 0
  mean_rewards = []
  quality_model.eval()
  optimizer = torch.optim.Adam(model.parameters(), lr=config.lr/1000)
  quality_model.eval()
  while curr_patience < PATIENCE:
    num_batch += 1
    model.eval()
    with torch.no_grad():
      generated = torch.zeros(config.batch_size, 1, config.total_vocab_size, device=device, dtype=torch.float32)
      generated[:,:,config.code_vocab_size] = 1
      for i in range(config.n_ctx-1):
        generated = model.sample(torch.cat((generated,torch.zeros((config.batch_size,1,config.total_vocab_size), device=device, dtype=torch.float32)), dim=1), True)
        if torch.sum(torch.sum(generated[:,:,config.code_vocab_size+1], dim=1).bool().int(), dim=0).item() == config.batch_size:
          break

    generated = torch.cat((generated, torch.zeros((config.batch_size, config.n_ctx - generated.size(1),config.total_vocab_size), device=device, dtype=torch.float32)), 1)
    generated_lens = getLens(generated)
    with torch.no_grad():
      rewards = quality_model(generated, generated_lens)

    model.train()
    generated_mask = torch.ones(config.batch_size, config.n_ctx-1, 1, device=device, dtype=torch.float32)
    for i in range(len(generated_lens)):
      if generated_lens[i] < config.n_ctx:
        generated_mask[i,int(generated_lens[i]-1):] = 0

    _, predictions, labels = model(generated, position_ids=None, ehr_labels=generated, ehr_masks=generated_mask)
    probs = torch.log(torch.abs(labels - 1.0 + predictions)).sum(-1).sum(-1)
    normalized_probs = probs / probs.sum()

    mean_rewards.append(rewards.mean().cpu().item())

    optimizer.zero_grad()
    loss = (normalized_probs * rewards).sum()
    loss.backward()
    optimizer.step()
      
    if num_batch % 10 == 0:
      print("Iteration %d Mean Reward Probability:%.6f"%(num_batch, np.mean(mean_rewards)))
      mean_rewards = []
      model.eval()
      with torch.no_grad():
        val_l = []
        for v_i in range(0, len(val_dataset), config.batch_size):
          batch_ehr, batch_mask = get_batch(val_dataset, v_i, config.batch_size)
          batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
          batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)

          val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
          val_l.append(val_loss.cpu().detach().numpy())
          
        cur_val_loss = np.mean(val_l)
        print("Iteration %d Validation Loss:%.7f"%(num_batch, cur_val_loss))
        if cur_val_loss < global_loss:
          curr_patience = 0
          global_loss = cur_val_loss
          state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration
            }
          torch.save(state, './save/medisim_model_note_ss')
          print('\n------------ Save best model ------------\n')
        else:
          curr_patience += 1

  model.load_state_dict(torch.load('./save/medisim_model_note_ss')['model'])
  return model, global_loss



model = MediSimModel(config).to(device)
if os.path.exists("./save/medisim_model_note_base"):
  print("Loading previous model")
  checkpoint = torch.load('./save/medisim_model_note_base', map_location=torch.device(device))
  model.load_state_dict(checkpoint['model'])
  iteration = 1#checkpoint['iteration'] + 1
else:
  iteration = 0

global_loss = 0.0019723 #1e10
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# if iteration == 0:
#   print("Training Model")
#   model, global_loss = train_model(model, optimizer, train_ehr_dataset, val_ehr_dataset, global_loss)
#   print(f'Global Loss - {global_loss}')

size = len(train_ehr_dataset)

while iteration < MAX_ITERATION:
  print(f'Beginning Iteration {iteration}')

  print("Generating Synthetic Data")
  synthetic_dataset = generate_dataset(model, size)
  quality_train_dataset = [(e, 0) for e in synthetic_dataset] + [(e, 1) for e in   train_ehr_dataset]

  print("Training Quality Model")
  quality_model = train_quality_model(quality_train_dataset)

  print("Training Core Model With Quality Model")
  model, global_loss = train_rl(model, quality_model, val_ehr_dataset, global_loss, iteration)

  print(f'Global Loss - {global_loss}')
  iteration += 1