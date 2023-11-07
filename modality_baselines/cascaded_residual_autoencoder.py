import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
from config import MediSimConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
LR = 0.0001
EPOCHS = 500
BATCH_SIZE = 1024
EMBEDDING_DIM = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

config = MediSimConfig()
train_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
train_dataset = [v for p in train_dataset for v in p['visits']]
val_dataset = pickle.load(open('./data/valDataset.pkl', 'rb'))
val_dataset = [v for p in val_dataset for v in p['visits']]
test_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
test_dataset = [v for p in test_dataset for v in p['visits']]

class ResidualAutoencoder(nn.Module):
  def __init__(self, config):
    super(ResidualAutoencoder, self).__init__()
    self.encoder = nn.Linear(config.total_vocab_size-config.special_vocab_size, EMBEDDING_DIM)
    self.decoder = nn.Linear(EMBEDDING_DIM, config.total_vocab_size-config.special_vocab_size)
    
  def forward(self, input):
    return self.decoder(torch.relu(self.encoder(input)))

class CascadedResidualAutoencoder(nn.Module):
  def __init__(self, config):
    super(CascadedResidualAutoencoder, self).__init__()
    self.ae1 = ResidualAutoencoder(config)
    self.ae2 = ResidualAutoencoder(config)
    self.ae3 = ResidualAutoencoder(config)
    self.diagnosis_vocab_size = config.diagnosis_vocab_size
    self.procedure_vocab_size = config.procedure_vocab_size
    self.medication_vocab_size = config.medication_vocab_size

  def forward(self, input):
    masked_input = torch.zeros(input.size(0), self.diagnosis_vocab_size+self.procedure_vocab_size+self.medication_vocab_size).to(input.device)
    masked_input[:,:self.diagnosis_vocab_size] = input
    
    output_1 = self.ae1(masked_input)
    output_1 = output_1 + masked_input
    
    output_2 = self.ae2(output_1)
    output_2 = output_2 + output_1
    
    output_3 = self.ae3(output_2)
    output_3 = output_3 + output_2
    return (torch.sigmoid(output_1), torch.sigmoid(output_2), torch.sigmoid(output_3)), torch.sigmoid(output_3[:,self.diagnosis_vocab_size:])

def get_batch(ehr_dataset, loc, batch_size):
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), config.diagnosis_vocab_size), np.int8)
    batch_labels = np.zeros((len(ehr), config.procedure_vocab_size+config.medication_vocab_size), np.int8)
    for i, v in enumerate(ehr):
        batch_ehr[i][[c for c in v if c < config.diagnosis_vocab_size]] = 1
        batch_labels[i][[c - config.diagnosis_vocab_size for c in v if c >= config.diagnosis_vocab_size]] = 1

    return batch_ehr, batch_labels

def train_model(model, train_dataset, val_dataset, save_name):
    global_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    for e in tqdm(range(EPOCHS)):
        np.random.shuffle(train_dataset)
        train_losses = []
        for i in range(0, len(train_dataset), BATCH_SIZE):
            model.train()
            batch_ehr, batch_labels = get_batch(train_dataset, i, BATCH_SIZE)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            (prob1, prob2, prob3), _ = model(batch_ehr)
            full_labels = torch.cat((batch_ehr, batch_labels), dim=-1)
            loss = bce(prob1, full_labels) + bce(prob2, full_labels) + bce(prob3, full_labels)
            train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        cur_train_loss = np.mean(train_losses)
        print("Epoch %d Training Loss:%.5f"%(e, cur_train_loss))
    
        model.eval()
        with torch.no_grad():
            val_losses = []
            for v_i in range(0, len(val_dataset), BATCH_SIZE):
                batch_ehr, batch_labels = get_batch(val_dataset, v_i, BATCH_SIZE)
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
                (prob1, prob2, prob3), _ = model(batch_ehr)
                full_labels = torch.cat((batch_ehr, batch_labels), dim=-1)
                val_loss = bce(prob1, full_labels) + bce(prob2, full_labels) + bce(prob3, full_labels)
                val_losses.append(val_loss.cpu().detach().numpy())
            cur_val_loss = np.mean(val_losses)
            print("Epoch %d Validation Loss:%.5f"%(e, cur_val_loss))
            if cur_val_loss < global_loss:
                global_loss = cur_val_loss
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, f'./save/modality/{save_name}')
                print('------------ Save best model ------------')

    model.load_state_dict(state['model'])

  

def test_model(model, test_dataset):
    loss_list = []
    fully_correct = []
    probability_list = []
    n_visits = len(test_dataset)
    n_pos_codes = 0
    n_total_codes = n_visits * (config.procedure_vocab_size + config.medication_vocab_size)
    cmatrix = None
    bce = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), BATCH_SIZE)):
            # Get batch inputs
            batch_ehr, batch_labels = get_batch(test_dataset, i, BATCH_SIZE)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            
            # Get batch outputs
            _, predictions = model(batch_ehr)
            rounded_preds = torch.round(predictions)

            # Get Loss
            val_loss = bce(predictions, batch_labels)
            loss_list.append(val_loss.cpu().detach().numpy())
            
            # Add number of visits and codes
            n_pos_codes += batch_labels.sum().cpu().item()
            
            # Add confusion matrix
            batch_cmatrix = metrics.confusion_matrix(batch_labels.cpu().detach().numpy().flatten(), rounded_preds.cpu().detach().numpy().flatten())
            cmatrix = batch_cmatrix if cmatrix is None else cmatrix + batch_cmatrix

            # Calculate and add probabilities 
            # Note that the masked codes will have probability 1 and be ignored
            label_probs = torch.abs(batch_labels - 1.0 + predictions)
            log_prob = torch.log(label_probs)
            log_prob[log_prob == -float("Inf")] = -100
            probability_list.append(log_prob.sum().cpu().item())
                
            for j in range(len(batch_labels)):
                if (batch_labels[j] == rounded_preds[j]).all():
                    fully_correct.append(1)
                else:
                    fully_correct.append(0)

    # Extract, save, and display test metrics
    avg_loss = np.mean(loss_list)
    tn, fp, fn, tp = cmatrix.ravel()
    full_acc = np.mean(fully_correct)
    acc = (tn + tp)/(tn+fp+fn+tp)
    prc = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = (2 * prc * rec)/(prc + rec)
    log_probability_overall = np.sum(probability_list)
    pp_visit_overall = np.exp(-log_probability_overall/n_visits)
    pp_positive_overall = np.exp(-log_probability_overall/n_pos_codes)
    pp_possible_overall = np.exp(-log_probability_overall/n_total_codes)
    
    metrics_dict = {}
    metrics_dict['Test Loss'] = avg_loss
    metrics_dict['Confusion Matrix'] = cmatrix
    metrics_dict['Accuracy'] = acc
    metrics_dict['Precision'] = prc
    metrics_dict['Recall'] = rec
    metrics_dict['F1 Score'] = f1
    metrics_dict['Full Visit Accuracy'] = full_acc
    metrics_dict['Test Log Probability Overall'] = log_probability_overall
    metrics_dict['Perplexity Per Visit Overall'] = pp_visit_overall
    metrics_dict['Perplexity Per Positive Code Overall'] = pp_positive_overall
    metrics_dict['Perplexity Per Possible Code Overall'] = pp_possible_overall
    
    print('Test Loss: ', avg_loss)
    print('Confusion Matrix: ', cmatrix)
    print('Accuracy: ', acc)
    print('Precision: ', prc)
    print('Recall: ', rec)
    print('F1 Score: ', f1)
    print('Full Visit Accuracy: ', full_acc)
    print('Test Log Probability Overall', log_probability_overall)
    print('Perplexity Per Visit Overall', pp_visit_overall)
    print('Perplexity Per Positive Code Overall', pp_positive_overall)
    print('Perplexity Per Possible Code Overall', pp_possible_overall)

    return metrics_dict

model = CascadedResidualAutoencoder(config).to(device)
train_model(model, train_dataset, val_dataset, f"cascaded_residual_autoencoder")
state = torch.load(f'./save/modality/cascaded_residual_autoencoder')
model.load_state_dict(state['model'])
results = test_model(model, test_dataset)
pickle.dump(results, open(f"results/modality_completion_stats/cascaded_residual_autoencoder_stats.pkl", "wb"))