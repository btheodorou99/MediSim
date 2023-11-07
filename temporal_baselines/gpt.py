import os
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from config import MediSimConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_ehr_dataset = pickle.load(open('data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('data/valDataset.pkl', 'rb'))
test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
  
def get_batch(loc, batch_size, mode):
    if mode == 'train':
        ehr = train_ehr_dataset[loc:loc+batch_size]
    elif mode == 'valid':
        ehr = val_ehr_dataset[loc:loc+batch_size]
    else:
        ehr = test_ehr_dataset[loc:loc+batch_size]
    
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

def shuffle_training_data(train_ehr_dataset):
    np.random.shuffle(train_ehr_dataset)

import copy
import math
import torch
import torch.nn as nn

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.total_vocab_size

        self.vis_embed_mat = nn.Linear(config.total_vocab_size, config.n_embd, bias=False)
        self.pos_embed_mat = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_visits, position_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_visits.size(1) + past_length, dtype=torch.long,
                                        device=input_visits.device)
            position_ids = position_ids.unsqueeze(0).expand(input_visits.size(0), input_visits.size(1))

        inputs_embeds = self.vis_embed_mat(input_visits)
        position_embeds = self.pos_embed_mat(position_ids)
        hidden_states = inputs_embeds + position_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents

class GPT4EHRHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT4EHRHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[0], embed_shape[1], bias=False)
        self.decoder.weight = nn.Parameter(model_embeddings_weights.transpose(0, 1))  # Tied weights

    def forward(self, hidden_state):
        code_logits = self.decoder(hidden_state)
        return code_logits

class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.transformer = GPT2Model(config)
        self.ehr_head = GPT4EHRHead(self.transformer.vis_embed_mat.weight, config)

    def set_tied(self):
        """Make sure we are sharing the embeddings"""
        self.ehr_head.set_embeddings_weights(self.transformer.vis_embed_mat.weight)

    def forward(self, input_visits, position_ids=None, ehr_labels=None, ehr_masks=None, past=None, pos_loss_weight=None):
        hidden_states, presents = self.transformer(input_visits, position_ids, past)
        code_logits = self.ehr_head(hidden_states)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        if ehr_labels is not None:    
            shift_probs = code_probs[..., :-1, :].contiguous()
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            loss_weights = None
            if pos_loss_weight is not None:
                loss_weights = torch.ones(shift_probs.shape, device=code_probs.device)
                loss_weights = loss_weights + (pos_loss_weight-1) * shift_labels
            if ehr_masks is not None:
                shift_probs = shift_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks
                if pos_loss_weight is not None:
                    loss_weights = loss_weights * ehr_masks

            bce = nn.BCELoss(weight=loss_weights)
            loss = bce(shift_probs, shift_labels)
            return loss, shift_probs, shift_labels
        
        return code_probs
    
    def sample(self, input_visits, random=True, position_ids=None, past=None):
        hidden_states, _ = self.transformer(input_visits, position_ids, past)
        next_logits = self.ehr_head(hidden_states[:,-1:,:])
        sig = nn.Sigmoid()
        next_probs = sig(next_logits)
        if random:
            visit = torch.bernoulli(next_probs)
        else:
            visit = torch.round(next_probs)

        return visit

model = GPT(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists("./save/temporal/gpt"):
    print("Loading previous model")
    checkpoint = torch.load('./save/temporal/gpt', map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Train Model
global_loss = 1e10
for e in tqdm(range(config.epoch)):
    shuffle_training_data(train_ehr_dataset)
    for i in range(0, len(train_ehr_dataset), config.batch_size):
        model.train()
        
        batch_ehr, batch_mask = get_batch(i, config.batch_size, 'train')
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        loss, _, _ = model(batch_ehr, ehr_labels=batch_ehr, ehr_masks=batch_mask)
        loss.backward()
        optimizer.step()
        
        if i % (250*config.batch_size) == 0:
            print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss))
        if i % (500*config.batch_size) == 0:
            if i == 0:
                continue
        
            model.eval()
            with torch.no_grad():
                val_l = []
                for v_i in range(0, len(val_ehr_dataset), config.batch_size):
                    batch_ehr, batch_mask = get_batch(v_i, config.batch_size, 'valid')
                    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
            
                    val_loss, _, _ = model(batch_ehr, ehr_labels=batch_ehr, ehr_masks=batch_mask)
                    val_l.append((val_loss).cpu().detach().numpy())
                    
                cur_val_loss = np.mean(val_l)
                print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
                if cur_val_loss < global_loss:
                    global_loss = cur_val_loss
                    state = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iteration': i
                        }
                    torch.save(state, './save/temporal/gpt')
                    print('\n------------ Save best model ------------\n')



###############
### TESTING ###
###############

model = GPT(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
checkpoint = torch.load('./save/temporal/gpt', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

def conf_mat(x, y):
    totaltrue = np.sum(x)
    totalfalse = len(x) - totaltrue
    truepos, totalpos = np.sum(x & y), np.sum(y)
    falsepos = totalpos - truepos
    return np.array([[totalfalse - falsepos, falsepos], #true negatives, false positives
                    [totaltrue - truepos, truepos]]) #false negatives, true positives

confusion_matrix = [None] * (config.n_ctx - 1)
probability_list = [[] for _ in range(config.n_ctx - 1)]
fully_correct = torch.zeros(config.n_ctx - 1)
n_visits = torch.zeros(config.n_ctx - 1)
n_pos_codes = torch.zeros(config.n_ctx - 1)
n_total_codes = torch.zeros(config.n_ctx - 1)
model.eval()
with torch.no_grad():
    for v_i in tqdm(range(0, len(test_ehr_dataset), config.batch_size)):
        # Get batch inputs
        batch_ehr, batch_mask = get_batch(v_i, config.batch_size, 'test')
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
        
        # Get batch outputs
        test_loss, predictions, labels = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
        rounded_preds = torch.round(predictions) 
        rounded_preds = rounded_preds + batch_mask - 1 # Setting the masked visits to be -1 to be ignored by the confusion matrix
        true_values = labels + batch_mask - 1 # Setting the masked visits to be -1 to be ignored by the confusion matrix
        
        # Add number of visits and codes
        n_visits += torch.sum(batch_mask, 0).squeeze().cpu()
        n_pos_codes += torch.sum(torch.sum(labels, 0), -1).cpu()
        n_total_codes += (torch.sum(batch_mask,0).squeeze() * config.total_vocab_size).cpu()

        # Add confusion matrix
        batch_cmatrix = [conf_mat((true_values[:,i,:] == 1).cpu().numpy().flatten(), (rounded_preds[:,i,:] == 1).cpu().numpy().flatten()) for i in range(config.n_ctx - 1)] 
        for i in range(config.n_ctx - 1):
            batch_cmatrix[i][0][0] = torch.sum(batch_mask[:,i]) * config.total_vocab_size - batch_cmatrix[i][0][1] - batch_cmatrix[i][1][0] - batch_cmatrix[i][1][1] # Remove the masked values
            confusion_matrix[i] = batch_cmatrix[i] if confusion_matrix[i] is None else confusion_matrix[i] + batch_cmatrix[i]

        # Calculate and add probabilities 
        # Note that the masked codes will have probability 1 and be ignored
        label_probs = torch.abs(labels - 1.0 + predictions)
        log_prob = torch.log(label_probs)
        for i in range(config.n_ctx - 1):
            probability_list[i].append(torch.sum(log_prob[:,i]).cpu().item())
        
        for j in range(len(labels)):
            for i in range(config.n_ctx - 1):
                if batch_mask[j,i] == 1 and (labels[j,i] == rounded_preds[j,i]).all():
                    fully_correct[i] += 1

# Save intermediate values in case of error
intermediate = {}
intermediate["Fully Correct"] = fully_correct
intermediate["Confusion Matrix"] = confusion_matrix
intermediate["Probabilities"] = probability_list
intermediate["Num Visits"] = n_visits
intermediate["Num Positive Codes"] = n_pos_codes
intermediate["Num Total Codes"] = n_total_codes
pickle.dump(intermediate, open("./results/temporal_completion_stats/GPT_intermediate_results.pkl", "wb"))

#Extract, save, and display test metrics
full_acc = []
acc = []
prc = []
rec = []
f1 = []
log_probability = []
pp_visit = []
pp_positive = []
pp_possible = []
for i in range(config.n_ctx - 1):
    tn, fp, fn, tp = confusion_matrix[i].ravel()
    full_acc.append(fully_correct[i]/n_visits[i])
    acc.append((tn + tp)/(tn+fp+fn+tp))
    prc.append(tp/(tp+fp))
    rec.append(tp/(tp+fn))
    f1.append((2 * prc[i] * rec[i])/(prc[i] + rec[i]))
    log_probability.append(np.sum(probability_list[i]))
    pp_visit.append(np.exp(-log_probability[i]/n_visits[i]))
    pp_positive.append(np.exp(-log_probability[i]/n_pos_codes[i]))
    pp_possible.append(np.exp(-log_probability[i]/n_total_codes[i]))

confusion_matrix_overall = np.array(confusion_matrix).sum(0)
tn, fp, fn, tp = confusion_matrix_overall.ravel()
full_acc_overall = fully_correct.sum()/n_visits.sum()
acc_overall = (tn + tp)/(tn+fp+fn+tp)
prc_overall = tp/(tp+fp)
rec_overall = tp/(tp+fn)
f1_overall = (2 * prc_overall * rec_overall)/(prc_overall + rec_overall)
log_probability_overall = np.sum([p for i in range(config.n_ctx - 1) for p in probability_list[i]])
pp_visit_overall = np.exp(-log_probability_overall/n_visits.sum())
pp_positive_overall = np.exp(-log_probability_overall/n_pos_codes.sum())
pp_possible_overall = np.exp(-log_probability_overall/n_total_codes.sum())
 
metrics_dict = {}
metrics_dict['Confusion Matrix'] = confusion_matrix
metrics_dict['Full Visit Accuracy'] = full_acc
metrics_dict['Accuracy'] = acc
metrics_dict['Precision'] = prc
metrics_dict['Recall'] = rec
metrics_dict['F1 Score'] = f1
metrics_dict['Test Log Probability'] = log_probability
metrics_dict['Perplexity Per Visit'] = pp_visit
metrics_dict['Perplexity Per Positive Code'] = pp_positive
metrics_dict['Perplexity Per Possible Code'] = pp_possible
metrics_dict['Confusion Matrix Overall'] = confusion_matrix_overall
metrics_dict['Full Visit Accuracy Overall'] = full_acc_overall
metrics_dict['Accuracy Overall'] = acc_overall
metrics_dict['Precision Overall'] = prc_overall
metrics_dict['Recall Overall'] = rec_overall
metrics_dict['F1 Score Overall'] = f1_overall
metrics_dict['Test Log Probability Overall'] = log_probability_overall
metrics_dict['Perplexity Per Visit Overall'] = pp_visit_overall
metrics_dict['Perplexity Per Positive Code Overall'] = pp_positive_overall
metrics_dict['Perplexity Per Possible Code Overall'] = pp_possible_overall
pickle.dump(metrics_dict, open("./results/temporal_completion_stats/GPT_Metrics.pkl", "wb"))

print("Confusion Matrix: ", confusion_matrix)
print('Full Visit Accuracy: ', full_acc)
print('Accuracy: ', acc)
print('Precision: ', prc)
print('Recall: ', rec)
print('F1 Score: ', f1)
print('Test Log Probability: ', log_probability)
print('Perplexity Per Visit: ', pp_visit)
print('Perplexity Per Positive Code: ', pp_positive)
print('Perplexity Per Possible Code: ', pp_possible)
print("Confusion Matrix Overall: ", confusion_matrix_overall)
print('Full Visit Accuracy Overall: ', full_acc_overall)
print('Accuracy Overall: ', acc_overall)
print('Precision Overall: ', prc_overall)
print('Recall Overall: ', rec_overall)
print('F1 Score Overall: ', f1_overall)
print('Test Log Probability Overall: ', log_probability_overall)
print('Perplexity Per Visit Overall: ', pp_visit_overall)
print('Perplexity Per Positive Code Overall: ', pp_positive_overall)
print('Perplexity Per Possible Code Overall: ', pp_possible_overall)