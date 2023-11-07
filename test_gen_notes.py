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
cudaNum = 0
NUM_SAMPLES = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device(f"cuda:{cudaNum}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

# # Test Model
# global_loss = 1e10
# model.eval()
# nlls = []
# num_tokens = 0
# with torch.no_grad():
#     for i in tqdm(range(0, len(test_dataset), config.batch_size)):
#         batch_context, batch_words, batch_num_words = get_batch(test_dataset, i, config.batch_size)
#         loss, _ = model(batch_context, batch_words, gen_loss=True)
#         log_likelihood = loss * batch_num_words
#         num_tokens += batch_num_words
#         nlls.append(log_likelihood.cpu().detach().item())
        
# ppl = np.exp(np.sum(nlls) / num_tokens)
# print(f"Perplexity: {ppl}")
# pickle.dump(ppl, open(f'results/note_generation/perplexity.pkl', 'wb'))



# Generate Samples
TEMPERATURES = [1.0, 0.75, 0.5, 0.25]
# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# sampleIdx = np.random.choice(len(test_dataset), size=NUM_SAMPLES)
# sample_contexts = [test_dataset[i] for i in sampleIdx]
# real_text = [tokenizer.decode(v[1]) for v in sample_contexts]
# pickle.dump(real_text, open(f'results/note_generation/realText.pkl', 'wb'))
# for n in real_text:
#     print(n)
# print("\n\n\n")

# sample_contexts, _, _ = get_batch(sample_contexts, 0, NUM_SAMPLES)
# for t in TEMPERATURES:
#     with torch.no_grad():
#         sample_text = torch.ones(NUM_SAMPLES, 0, dtype=torch.long, device=device) * config.word_vocab_size
#         for _ in tqdm(range(config.n_gen_positions), leave=False):
#             next_logits = model(sample_contexts, sample_text, gen_loss=False)[:,-1,:config.word_vocab_size]
#             next_logits = next_logits / t
#             next_probs = F.softmax(next_logits, dim=-1)
#             next_tokens = torch.multinomial(next_probs, num_samples=1)
#             sample_text = torch.cat((sample_text, next_tokens), dim=1)
       
#     sample_dataset = [] 
#     for i in range(NUM_SAMPLES):
#         sample_note = []
#         for j in range(config.n_gen_positions):
#             if sample_text[i,j] == 102:
#                 break
            
#             sample_note.append(sample_text[i,j])
#         sample_dataset.append(tokenizer.decode(sample_note))
#     pickle.dump(sample_dataset, open(f'results/note_generation/sampleText_{t}.pkl', 'wb'))
#     print(f"Temperature {t}:")
#     for n in sample_dataset:
#         print(n)
#     print("\n\n\n") 
    
# PRINT EVERYTHING
texts = [pickle.load(open('results/note_generation/realText.pkl', 'rb'))] + [pickle.load(open(f'results/note_generation/sampleText_{t}.pkl', 'rb')) for t in TEMPERATURES]
for i in range(len(texts[0])):
    print(f"Real Text {i}: {texts[0][i]}")
    for j in range(1, len(texts)):
        print(f"Sample Text {j-1} {i}: {texts[j][i]}")
    print("\n\n\n")
            