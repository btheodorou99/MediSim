import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm

from config_note import MediSimConfig
from model_note import DecoderModel

SEED = 4
cudaNum = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device(f"cuda:{cudaNum}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_dataset = pickle.load(open('./data_note/trainDataset.pkl', 'rb'))
train_dataset = [v for p in train_dataset for v in p['visits']]
train_dataset = [(v[0], v[1][1:config.n_gen_positions]) for v in train_dataset]
val_dataset = pickle.load(open('./data_note/valDataset.pkl', 'rb'))
val_dataset = [v for p in val_dataset for v in p['visits']]
val_dataset = [(v[0], v[1][1:config.n_gen_positions]) for v in val_dataset]

def get_batch(dataset, loc, batch_size):
    notes = dataset[loc:loc+batch_size]
    bs = len(notes)
    batch_context = torch.zeros(bs, 1, config.code_vocab_size, dtype=torch.float, device=device)
    batch_note = torch.ones(bs, config.n_gen_positions-1, dtype=torch.long, device=device) * config.word_vocab_size
    for i, n in enumerate(notes):
        codes, text = n
        batch_context[i, 0, codes] = 1
        batch_note[i,:len(text)] = torch.IntTensor(text)
        
    return batch_context, batch_note

def shuffle_training_data(train_ehr_dataset):
    np.random.shuffle(train_ehr_dataset)

model = DecoderModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists(f"./save/model_note_gen"):
    print("Loading previous model")
    checkpoint = torch.load(f'./save/model_note_gen', map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Train Model
global_loss = 1e10
curr_patience = 0
for e in tqdm(range(config.epoch_gen)):
    shuffle_training_data(train_dataset)
    train_losses = []
    model.train()
    for i in range(0, len(train_dataset), config.batch_size_gen):
        batch_context, batch_words = get_batch(train_dataset, i, config.batch_size_gen)
        optimizer.zero_grad()
        loss, _ = model(batch_context, batch_words, gen_loss=True)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.cpu().detach().item())
        
    print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, np.mean(train_losses)))
    model.eval()
    with torch.no_grad():
        val_losses = []
        for v_i in range(0, len(val_dataset), config.batch_size_gen):
            batch_context, batch_words = get_batch(val_dataset, v_i, config.batch_size_gen)                 
            val_loss, _ = model(batch_context, batch_words, gen_loss=True)
            val_losses.append((val_loss).cpu().detach().item())
        
        cur_val_loss = np.mean(val_losses)
        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
        if cur_val_loss < global_loss:
            curr_patience = 0
            global_loss = cur_val_loss
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, f'./save/model_note_gen')
            print('\n------------ Save best model ------------\n')
        else:
            curr_patience += 1
            if curr_patience >= config.patience:
                print("Early stopping")
                break