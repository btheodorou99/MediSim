import os
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
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
    batch_lens = batch_mask.sum(1).squeeze(-1).astype(np.int8)
    return batch_ehr, batch_mask, batch_lens

def shuffle_training_data(train_ehr_dataset):
    np.random.shuffle(train_ehr_dataset)

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

model = RETAIN(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists("./save/temporal/retain"):
    print("Loading previous model")
    checkpoint = torch.load('./save/temporal/retain', map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Train Model
global_loss = 1e10
for e in tqdm(range(config.epoch * 10)):
    shuffle_training_data(train_ehr_dataset)
    for i in range(0, len(train_ehr_dataset), config.batch_size):
        model.train()
        
        batch_ehr, batch_mask, batch_lens = get_batch(i, config.batch_size, 'train')
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        loss, _, _ = model(batch_ehr, batch_lens, ehr_labels=batch_ehr, ehr_masks=batch_mask)
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
                    batch_ehr, batch_mask, batch_lens = get_batch(v_i, config.batch_size, 'valid')
                    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
            
                    val_loss, _, _ = model(batch_ehr, batch_lens, ehr_labels=batch_ehr, ehr_masks=batch_mask)
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
                    torch.save(state, './save/temporal/retain')
                    print('\n------------ Save best model ------------\n')



###############
### TESTING ###
###############

model = RETAIN(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
checkpoint = torch.load('./save/temporal/retain', map_location=torch.device(device))
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
        batch_ehr, batch_mask, batch_lens = get_batch(v_i, config.batch_size, 'test')
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
        
        # Get batch outputs
        test_loss, predictions, labels = model(batch_ehr, batch_lens, ehr_labels=batch_ehr, ehr_masks=batch_mask)
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
pickle.dump(intermediate, open("./results/temporal_completion_stats/RETAIN_intermediate_results.pkl", "wb"))

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
pickle.dump(metrics_dict, open("./results/temporal_completion_stats/RETAIN_Metrics.pkl", "wb"))

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