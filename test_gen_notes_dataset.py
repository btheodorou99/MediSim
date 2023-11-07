import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from config_note import MediSimConfig
from transformers import AutoTokenizer
from model_note import DecoderModel

SEED = 4
cudaNum = 6
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
test_dataset = pickle.load(open('./data_note/testDataset.pkl', 'rb'))
test_dataset = [v for p in test_dataset for v in p['visits']]
test_dataset = [(v[0], v[1][1:config.n_gen_positions]) for v in test_dataset]

def get_batch(dataset, loc, batch_size):
    notes = dataset[loc:loc+batch_size]
    bs = len(notes)
    batch_context = torch.zeros(bs, 1, config.code_vocab_size, dtype=torch.float, device=device)
    batch_note = torch.ones(bs, config.n_gen_positions-1, dtype=torch.long, device=device) * config.word_vocab_size
    batch_num_words = 0
    for i, n in enumerate(notes):
        codes, text = n
        batch_context[i, 0, codes] = 1
        batch_note[i,:len(text)] = torch.IntTensor(text)
        batch_num_words += len(text)
        
    return batch_context, batch_note, batch_num_words
  
model = DecoderModel(config).to(device)
print("Loading previous model")
checkpoint = torch.load(f'./save/model_note_gen', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model.eval()
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Generate Samples
# real_text_train = [tokenizer.decode(v[1]) for v in train_dataset]
# pickle.dump(real_text_train, open(f'data_text/realTextTrain.pkl', 'wb'))
# real_text_val = [tokenizer.decode(v[1]) for v in val_dataset]
# pickle.dump(real_text_val, open(f'data_text/realTextVal.pkl', 'wb'))
# real_text_test = [tokenizer.decode(v[1]) for v in test_dataset]
# pickle.dump(real_text_test, open(f'data_text/realTextTest.pkl', 'wb'))

TEMPERATURES = [0.75, 0.5] #[1.0, 0.75, 0.5]
with torch.no_grad():
    for t in tqdm(TEMPERATURES):
        sample_dataset = [] 
        for i in tqdm(range(0, len(train_dataset), config.batch_size_gen), leave=False):
            sample_contexts, _, _ = get_batch(train_dataset, i, config.batch_size_gen)
            sample_text = torch.ones(sample_contexts.size(0), 0, dtype=torch.long, device=device) * config.word_vocab_size
            for _ in tqdm(range(config.n_gen_positions), leave=False):
                next_logits = model(sample_contexts, sample_text, gen_loss=False)[:,-1,:config.word_vocab_size]
                next_logits = next_logits / t
                next_probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(next_probs, num_samples=1)
                sample_text = torch.cat((sample_text, next_tokens), dim=1)
        
            for j in range(sample_text.size(0)):
                sample_note = []
                for k in range(config.n_gen_positions):
                    if sample_text[j,k] == 102:
                        break
                    
                    sample_note.append(sample_text[j,k])
                sample_dataset.append(tokenizer.decode(sample_note))
        pickle.dump(sample_dataset, open(f'data_text/sampleTextTrain_{t}.pkl', 'wb'))
        
        sample_dataset = [] 
        for i in tqdm(range(0, len(val_dataset), config.batch_size_gen), leave=False):
            sample_contexts, _, _ = get_batch(val_dataset, i, config.batch_size_gen)
            sample_text = torch.ones(sample_contexts.size(0), 0, dtype=torch.long, device=device) * config.word_vocab_size
            for _ in tqdm(range(config.n_gen_positions), leave=False):
                next_logits = model(sample_contexts, sample_text, gen_loss=False)[:,-1,:config.word_vocab_size]
                next_logits = next_logits / t
                next_probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(next_probs, num_samples=1)
                sample_text = torch.cat((sample_text, next_tokens), dim=1)
        
            for j in range(sample_text.size(0)):
                sample_note = []
                for k in range(config.n_gen_positions):
                    if sample_text[j,k] == 102:
                        break
                    
                    sample_note.append(sample_text[j,k])
                sample_dataset.append(tokenizer.decode(sample_note))
        pickle.dump(sample_dataset, open(f'data_text/sampleTextVal_{t}.pkl', 'wb'))
        
        sample_dataset = [] 
        for i in tqdm(range(0, len(test_dataset), config.batch_size_gen), leave=False):
            sample_contexts, _, _ = get_batch(test_dataset, i, config.batch_size_gen)
            sample_text = torch.ones(sample_contexts.size(0), 0, dtype=torch.long, device=device) * config.word_vocab_size
            for _ in tqdm(range(config.n_gen_positions), leave=False):
                next_logits = model(sample_contexts, sample_text, gen_loss=False)[:,-1,:config.word_vocab_size]
                next_logits = next_logits / t
                next_probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(next_probs, num_samples=1)
                sample_text = torch.cat((sample_text, next_tokens), dim=1)
        
            for j in range(sample_text.size(0)):
                sample_note = []
                for k in range(config.n_gen_positions):
                    if sample_text[j,k] == 102:
                        break
                    
                    sample_note.append(sample_text[j,k])
                sample_dataset.append(tokenizer.decode(sample_note))
        pickle.dump(sample_dataset, open(f'data_text/sampleTextTest_{t}.pkl', 'wb'))