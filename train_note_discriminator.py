import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from config_note import MediSimConfig
from transformers import AutoTokenizer
from model_discriminator import NoteDiscriminator
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

SEED = 4
TEMPERATURE = 0.5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# train_note_dataset = [(t, 1) for t in pickle.load(open('data_text/realTextTrain.pkl', 'rb'))] + [(t, 0) for t in pickle.load(open(f'data_text/sampleTextTrain_{TEMPERATURE}.pkl', 'rb'))]
# train_note_dataset = [(tokenizer.encode(t)[:config.n_embed_positions], l) for (t,l) in train_note_dataset]
# print(len(train_note_dataset))
# val_note_dataset = [(t, 1) for t in pickle.load(open('data_text/realTextVal.pkl', 'rb'))] + [(t, 0) for t in pickle.load(open(f'data_text/sampleTextVal_{TEMPERATURE}.pkl', 'rb'))]
# val_note_dataset = [(tokenizer.encode(t)[:config.n_embed_positions], l) for (t,l) in val_note_dataset]
test_note_dataset = [(t, 1) for t in pickle.load(open('data_text/realTextTest.pkl', 'rb'))] + [(t, 0) for t in pickle.load(open(f'data_text/sampleTextTest_{TEMPERATURE}.pkl', 'rb'))]
test_note_dataset = [(tokenizer.encode(t)[:config.n_embed_positions], l) for (t,l) in test_note_dataset]

def get_batch(dataset, loc, batch_size):
  notes = dataset[loc:loc+batch_size]
  batch_note = np.ones((len(notes), config.n_embed_positions)) * config.word_vocab_size
  batch_label = np.zeros((len(notes), 1))
  for i, (t, l) in enumerate(notes):
    batch_note[i,:len(t)] = t
    batch_label[i,0] = l
  
  return batch_note, batch_label

def shuffle_training_data(train_note_dataset):
  np.random.shuffle(train_note_dataset)

model = NoteDiscriminator(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists("./save/note_discriminator"):
  print("Loading previous model")
  checkpoint = torch.load('./save/note_discriminator', map_location=torch.device(device))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])

# # Train Model
# PATIENCE = 0
# global_loss = 1e10
loss_fn = torch.nn.BCELoss()
# for e in tqdm(range(config.epoch)):
#   model.train()
#   shuffle_training_data(train_note_dataset)
#   train_l = []
#   for i in range(0, len(train_note_dataset), config.discriminator_batch_size):
#     batch_note, batch_label = get_batch(train_note_dataset, i, config.discriminator_batch_size)
#     batch_note = torch.tensor(batch_note, dtype=torch.long).to(device)
#     batch_label = torch.tensor(batch_label, dtype=torch.float32).to(device)
    
#     optimizer.zero_grad()
#     preds = model(batch_note)
#     loss = loss_fn(preds, batch_label)
#     loss.backward()
#     optimizer.step()
#     train_l.append((loss).cpu().detach().numpy())
    
#   print("Epoch %d Training Loss:%.7f"%(e, np.mean(train_l)))
#   model.eval()
#   with torch.no_grad():
#     val_l = []
#     for v_i in range(0, len(val_note_dataset), config.discriminator_batch_size):
#       batch_note, batch_label = get_batch(val_note_dataset, v_i, config.discriminator_batch_size)
#       batch_note = torch.tensor(batch_note, dtype=torch.long).to(device)
#       batch_label = torch.tensor(batch_label, dtype=torch.float32).to(device)
    
#       preds = model(batch_note)
#       val_loss = loss_fn(preds, batch_label)
#       val_l.append((val_loss).cpu().detach().numpy())
      
#     cur_val_loss = np.mean(val_l)
#     print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
#     if cur_val_loss < global_loss:
#       PATIENCE = 0
#       global_loss = cur_val_loss
#       state = {
#             'model': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#         }
#       torch.save(state, './save/note_discriminator')
#       print('\n------------ Save best model ------------\n')
#     else:
#       PATIENCE += 1
#       if PATIENCE > config.patience:
#         print("Early stopping")
#         break
      
# Test Model
# model.load_state_dict(torch.load('./save/note_discriminator', map_location=torch.device(device))['model'])
model.eval()
with torch.no_grad():
  test_l = []
  preds = []
  labels = []
  for t_i in tqdm(range(0, len(test_note_dataset), config.batch_size)):
    batch_note, batch_label = get_batch(test_note_dataset, t_i, config.batch_size)
    batch_note = torch.tensor(batch_note, dtype=torch.long).to(device)
    batch_label = torch.tensor(batch_label, dtype=torch.float32).to(device)
    
    batch_preds = model(batch_note)
    test_loss = loss_fn(batch_preds, batch_label)
    test_l.append((test_loss).cpu().detach().numpy())
    preds += batch_preds.squeeze(-1).cpu().detach().numpy().tolist()
    labels += batch_label.squeeze(-1).cpu().detach().numpy().tolist()
    
  print("Test Loss:%.7f"%(np.mean(test_l)))
  preds = np.array(preds)
  labels = np.array(labels)
  rounded_preds = np.round(preds)
  print("Test Accuracy:%.7f"%(accuracy_score(labels, rounded_preds)))
  print("Test Precision:%.7f"%(precision_score(labels, rounded_preds)))
  print("Test Recall:%.7f"%(recall_score(labels, rounded_preds)))
  print("Test F1:%.7f"%(f1_score(labels, rounded_preds)))
  print("Test AUC:%.7f"%(roc_auc_score(labels, rounded_preds)))