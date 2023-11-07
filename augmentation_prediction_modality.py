import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
from config import MediSimConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

LR = 0.001
EPOCHS = 150
BATCH_SIZE = 64
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_VAL_EXAMPLES = 500
PATIENCE = 3
NOISE = 1
indexSets = pickle.load(open('data/indexSets.pkl', 'rb'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = MediSimConfig()
real_diagnosis_full = pickle.load(open('./results/datasets/real_diagnosis_full.pkl', 'rb'))
real_diagnosis_full = [p for p in real_diagnosis_full if len(p['visits']) > 1]
real_full = pickle.load(open('./results/datasets/real_full.pkl', 'rb'))
real_full = [p for p in real_full if len(p['visits']) > 1]
modality_added = pickle.load(open('./results/datasets/modality_added.pkl', 'rb'))
modality_added = [p for p in modality_added if len(p['visits']) > 1]
modality_added_ss = pickle.load(open('./results/datasets/modality_added_ss.pkl', 'rb'))
modality_added_ss = [p for p in modality_added_ss if len(p['visits']) > 1]
modality_added_lr = pickle.load(open('./results/datasets_modality_baselines/modality_added_logistic_regression.pkl', 'rb'))
modality_added_lr = [p for p in modality_added_lr if len(p['visits']) > 1]
modality_added_nn = pickle.load(open('./results/datasets_modality_baselines/modality_added_neural_network.pkl', 'rb'))
modality_added_nn = [p for p in modality_added_nn if len(p['visits']) > 1]
modality_added_cra = pickle.load(open('./results/datasets_modality_baselines/modality_added_cascaded_residual_autoencoder.pkl', 'rb'))
modality_added_cra = [p for p in modality_added_cra if len(p['visits']) > 1]
test_dataset = pickle.load(open('./results/datasets/test_dataset.pkl', 'rb'))
test_dataset = [p for p in test_dataset if len(p['visits']) > 1]

# Add Noise
all_diags = set(range(config.diagnosis_vocab_size))
for p in real_diagnosis_full + real_full + modality_added + modality_added_ss + modality_added_lr + modality_added_nn + modality_added_cra + test_dataset:
    new_visits = []
    old_visits = p['visits']
    for i in range(len(old_visits)):
        visit = old_visits[i]
        if i == len(old_visits) - 1:
            new_visits.append(visit)
        else:
            diags = [c for c in visit if c < config.diagnosis_vocab_size]
            otherCodes = [c for c in visit if c >= config.diagnosis_vocab_size]
            noisy_diags = np.random.choice(diags, max(0, len(diags)-NOISE), replace=False).tolist() + np.random.choice(list(all_diags - set(diags)), NOISE, replace=False).tolist()
            noisy_codes = noisy_diags + otherCodes
            new_visits.append(noisy_codes)
    p['visits'] = new_visits

class PredictionModel(nn.Module):
    def __init__(self, config):
        super(PredictionModel, self).__init__()
        self.embedding = nn.Linear(config.code_vocab_size, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=LSTM_HIDDEN_DIM,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2*LSTM_HIDDEN_DIM, 1)

    def forward(self, input_visits, lengths):
        visit_emb = self.embedding(input_visits)
        visit_emb = self.dropout(visit_emb)
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :LSTM_HIDDEN_DIM]
        out_reverse = output[:, 0, LSTM_HIDDEN_DIM:]
        out_combined = torch.cat((out_forward, out_reverse), 1)

        patient_embedding = self.fc(out_combined)
        patient_embedding = torch.squeeze(patient_embedding, 1)
        prob = torch.sigmoid(patient_embedding)
        
        return prob

def get_batch(ehr_dataset, loc, batch_size, label_codes):
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.code_vocab_size))
    batch_labels = np.zeros(len(ehr))
    batch_lens = np.zeros(len(ehr))
    for i, p in enumerate(ehr):
        visits = p['visits'][:-1]
        batch_lens[i] = len(visits)
        for j, v in enumerate(visits):
            batch_ehr[i,j][v] = 1

        label = 0
        label_visits = p['visits'][-1:]
        for v in label_visits:
            for c in v:
                if c in label_codes:
                    label = 1

        batch_labels[i] = label

    return batch_ehr, batch_labels, batch_lens

def train_model(model, train_dataset, save_name, label_codes):
    global_loss = 1e10
    val_dataset = train_dataset[:NUM_VAL_EXAMPLES]
    train_dataset = train_dataset[NUM_VAL_EXAMPLES:]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    for e in range(EPOCHS):
        np.random.shuffle(train_dataset)
        train_losses = []
        for i in range(0, len(train_dataset), BATCH_SIZE):
            model.train()
            batch_ehr, batch_labels, batch_lens = get_batch(train_dataset, i, BATCH_SIZE, label_codes)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            prob = model(batch_ehr, batch_lens)
            loss = bce(prob, batch_labels)
            train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        cur_train_loss = np.mean(train_losses)
        if e % 5 == 0:
            print("\nEpoch %d Training Loss:%.5f"%(e, cur_train_loss))
    
        model.eval()
        with torch.no_grad():
            val_losses = []
            for v_i in range(0, len(val_dataset), BATCH_SIZE):
                batch_ehr, batch_labels, batch_lens = get_batch(val_dataset, v_i, BATCH_SIZE, label_codes)
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
                prob = model(batch_ehr, batch_lens)
                val_loss = bce(prob, batch_labels)
                val_losses.append(val_loss.cpu().detach().numpy())
            cur_val_loss = np.mean(val_losses)
            if e % 5 == 0:
                print("Epoch %d Validation Loss:%.5f"%(e, cur_val_loss))
            if cur_val_loss < global_loss:
                patience = 0
                global_loss = cur_val_loss
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, f'./save/augmentation/prediction/{save_name}')
            else:
                patience += 1
                if patience == PATIENCE:
                    break

    model.load_state_dict(state['model'])

def test_model(model, test_dataset, label_codes):
    loss_list = []
    pred_list = []
    true_list = []
    bce = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_dataset), BATCH_SIZE):  
            batch_ehr, batch_labels, batch_lens = get_batch(test_dataset, i, BATCH_SIZE, label_codes)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            prob = model(batch_ehr, batch_lens)
            val_loss = bce(prob, batch_labels)
            loss_list.append(val_loss.cpu().detach().numpy())
            pred_list += list(prob.cpu().detach().numpy())
            true_list += list(batch_labels.cpu().detach().numpy())
    
    round_list = np.around(pred_list)

    # Extract, save, and display test metrics
    avg_loss = np.mean(loss_list)
    cmatrix = metrics.confusion_matrix(true_list, round_list)
    acc = metrics.accuracy_score(true_list, round_list)
    prc = metrics.precision_score(true_list, round_list)
    rec = metrics.recall_score(true_list, round_list)
    f1 = metrics.f1_score(true_list, round_list)
    auroc = metrics.roc_auc_score(true_list, pred_list)
    (precisions, recalls, _) = metrics.precision_recall_curve(true_list, pred_list)
    auprc = metrics.auc(recalls, precisions)
    
    metrics_dict = {}
    metrics_dict['Test Loss'] = avg_loss
    metrics_dict['Confusion Matrix'] = cmatrix
    metrics_dict['Accuracy'] = acc
    metrics_dict['Precision'] = prc
    metrics_dict['Recall'] = rec
    metrics_dict['F1 Score'] = f1
    metrics_dict['AUROC'] = auroc
    metrics_dict['AUPRC'] = auprc
    
    print('Test Loss: ', avg_loss)
    print('Confusion Matrix: ', cmatrix)
    print('Accuracy: ', acc)
    print('Precision: ', prc)
    print('Recall: ', rec)
    print('F1 Score: ', f1)
    print('AUROC: ', auroc)
    print('AUPRC: ', auprc)
    print("\n")

    return metrics_dict

results = {}
for (label, label_codes) in indexSets.items():
    print(f'\n\n{label}\n\n')
    label_results = {}

    # Perform the different experiments
    print('Real Diagnosis Full')
    model_real_diagnosis_full = PredictionModel(config).to(device)
    train_model(model_real_diagnosis_full, real_diagnosis_full, f"real_diagnosis_full_missing_{NOISE}_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/real_diagnosis_full_missing_{NOISE}_{label}')
    model_real_diagnosis_full.load_state_dict(state['model'])
    test_results_real_diagnosis_full = test_model(model_real_diagnosis_full, test_dataset, label_codes)
    label_results[f'Real Diagnosis Full'] = test_results_real_diagnosis_full

    print('Real Full')
    model_real_full = PredictionModel(config).to(device)
    train_model(model_real_full, real_full, f"real_full_missing_{NOISE}_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/real_full_missing_{NOISE}_{label}')
    model_real_full.load_state_dict(state['model'])
    test_results_real_full = test_model(model_real_full, test_dataset, label_codes)
    label_results[f'Real Full'] = test_results_real_full

    print('Modality Added')
    model_modality_added = PredictionModel(config).to(device)
    train_model(model_modality_added, modality_added, f"modality_added_missing_{NOISE}_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/modality_added_missing_{NOISE}_{label}')
    model_modality_added.load_state_dict(state['model'])
    test_results_modality_added = test_model(model_modality_added, test_dataset, label_codes)
    label_results[f'Modality Added'] = test_results_modality_added

    print('Modality Added SS')
    model_modality_added_ss = PredictionModel(config).to(device)
    train_model(model_modality_added_ss, modality_added_ss, f"modality_added_ss_missing_{NOISE}_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/modality_added_ss_missing_{NOISE}_{label}')
    model_modality_added_ss.load_state_dict(state['model'])
    test_results_modality_added_ss = test_model(model_modality_added_ss, test_dataset, label_codes)
    label_results[f'Modality Added SS'] = test_results_modality_added_ss

    print('Modality Added LR')
    model_modality_added_lr = PredictionModel(config).to(device)
    train_model(model_modality_added_lr, modality_added_lr, f"modality_added_lr_missing_{NOISE}_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/modality_added_lr_missing_{NOISE}_{label}')
    model_modality_added_lr.load_state_dict(state['model'])
    test_results_modality_added_lr = test_model(model_modality_added_lr, test_dataset, label_codes)
    label_results[f'Modality Added LR'] = test_results_modality_added_lr

    print('Modality Added CRA')
    model_modality_added_cra = PredictionModel(config).to(device)
    train_model(model_modality_added_cra, modality_added_cra, f"modality_added_cra_missing_{NOISE}_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/modality_added_cra_missing_{NOISE}_{label}')
    model_modality_added_cra.load_state_dict(state['model'])
    test_results_modality_added_cra = test_model(model_modality_added_cra, test_dataset, label_codes)
    label_results[f'Modality Added CRA'] = test_results_modality_added_cra

    print('Modality Added NN')
    model_modality_added_nn = PredictionModel(config).to(device)
    train_model(model_modality_added_nn, modality_added_nn, f"modality_added_nn_missing_{NOISE}_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/modality_added_nn_missing_{NOISE}_{label}')
    model_modality_added_nn.load_state_dict(state['model'])
    test_results_modality_added_nn = test_model(model_modality_added_nn, test_dataset, label_codes)
    label_results[f'Modality Added NN'] = test_results_modality_added_nn

    results[label] = label_results

pickle.dump(results, open(f"results/augmentation_stats/prediction_stats_missing_{NOISE}.pkl", "wb"))
results['Average'] = {k: {m: np.mean([results[i][k][m] for i in results]) for m in list(list(results.values())[0].values())[0].keys() if m != 'Confusion Matrix'} for k in list(results.values())[0].keys()}
print(results['Average'])
pickle.dump(results, open(f"results/augmentation_stats/prediction_stats_missing_{NOISE}.pkl", "wb"))