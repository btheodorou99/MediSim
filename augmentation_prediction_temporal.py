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
EPOCHS = 100
BATCH_SIZE = 32
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_VAL_EXAMPLES = 5000
PATIENCE = 3

idToLabel = {}
idToLabel[0] = "Alzheimer or related disorders or senile"
idToLabel[1] = "Heart Failure"
idToLabel[2] = "Chronic Kidney Disease"
idToLabel[3] = "Cancer"
idToLabel[4] = "Chronic Obstructive Pulmonary Disease"
idToLabel[5] = "Depression"
idToLabel[6] = "Diabetes"
idToLabel[7] = "Ischemic Heart Disease"
idToLabel[8] = "Osteoporosis"
idToLabel[9] = "Rheumatoid Arthritis and Osteoarthritis (RA/OA)"
idToLabel[10] = "Stroke/transient Ischemic Attack"

idToCodes = {}
idToCodes[0] = pickle.load(open('./data/alzheimersIdx.pkl', 'rb'))
idToCodes[1] = pickle.load(open('./data/heartFailureIdx.pkl', 'rb'))
idToCodes[2] = pickle.load(open('./data/kidneyDiseaseIdx.pkl', 'rb'))
idToCodes[3] = pickle.load(open('./data/cancerIdx.pkl', 'rb'))
idToCodes[4] = pickle.load(open('./data/copdIdx.pkl', 'rb'))
idToCodes[5] = pickle.load(open('./data/depressionIdx.pkl', 'rb'))
idToCodes[6] = pickle.load(open('./data/diabetesIdx.pkl', 'rb'))
idToCodes[7] = pickle.load(open('./data/heartDiseaseIdx.pkl', 'rb'))
idToCodes[8] = pickle.load(open('./data/osteoporosisIdx.pkl', 'rb'))
idToCodes[9] = pickle.load(open('./data/arthritisIdx.pkl', 'rb'))
idToCodes[10] = pickle.load(open('./data/strokeIdx.pkl', 'rb'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = MediSimConfig()
real_full = pickle.load(open('./results/datasets/real_full.pkl', 'rb'))
real_shortened = pickle.load(open('./results/datasets/real_shortened.pkl', 'rb'))
extended = pickle.load(open('./results/datasets/extended.pkl', 'rb'))
extended_ss = pickle.load(open('./results/datasets/extended_ss.pkl', 'rb'))
extended_lstm = pickle.load(open('./results/datasets_temporal_baselines/extended_lstm.pkl', 'rb'))
extended_gpt = pickle.load(open('./results/datasets_temporal_baselines/extended_gpt.pkl', 'rb'))
extended_retain = pickle.load(open('./results/datasets_temporal_baselines/extended_retain.pkl', 'rb'))
extended_conan = pickle.load(open('./results/datasets_temporal_baselines/extended_conan.pkl', 'rb'))
extended_dipole = pickle.load(open('./results/datasets_temporal_baselines/extended_dipole.pkl', 'rb'))
extended_synteg = pickle.load(open('./results/datasets_temporal_baselines/extended_synteg.pkl', 'rb'))
test_dataset = pickle.load(open('./results/datasets/test_dataset.pkl', 'rb'))

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
        visits = p['visits'][:-5]
        batch_lens[i] = len(visits)
        for j, v in enumerate(visits):
            batch_ehr[i,j][v] = 1

        label = 0
        label_visits = p['visits'][-5:]
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
    patience = 0
    for e in tqdm(range(EPOCHS)):
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
        print("Epoch %d Training Loss:%.5f"%(e, cur_train_loss))
    
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
            print("Epoch %d Validation Loss:%.5f"%(e, cur_val_loss))
            if cur_val_loss < global_loss:
                patience = 0
                global_loss = cur_val_loss
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, f'./save/augmentation/prediction/{save_name}')
                print('------------ Save best model ------------')
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
for (label_idx, label_codes) in idToCodes.items():
    if label_idx not in [10]:
        continue
        
    label = idToLabel[label_idx].replace('(', '').replace(')', '').replace('/','')
    print(label)
    label_results = {}

    # Perform the different experiments
    model_real_full = PredictionModel(config).to(device)
    train_model(model_real_full, real_full, f"real_full_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/real_full_{label}')
    model_real_full.load_state_dict(state['model'])
    test_results_real_full = test_model(model_real_full, test_dataset, label_codes)
    label_results[f'Real Full'] = test_results_real_full

    model_real_shortened = PredictionModel(config).to(device)
    train_model(model_real_shortened, real_shortened, f"real_shortened_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/real_shortened_{label}')
    model_real_shortened.load_state_dict(state['model'])
    test_results_real_shortened = test_model(model_real_shortened, test_dataset, label_codes)
    label_results[f'Real Shortened'] = test_results_real_shortened

    model_extended = PredictionModel(config).to(device)
    train_model(model_extended, extended, f"extended_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/extended_{label}')
    model_extended.load_state_dict(state['model'])
    test_results_extended = test_model(model_extended, test_dataset, label_codes)
    label_results[f'Extended'] = test_results_extended

    model_extended_ss = PredictionModel(config).to(device)
    train_model(model_extended_ss, extended_ss, f"extended_ss_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/extended_ss_{label}')
    model_extended_ss.load_state_dict(state['model'])
    test_results_extended = test_model(model_extended_ss, test_dataset, label_codes)
    label_results[f'Extended SS'] = test_results_extended

    model_extended_lstm = PredictionModel(config).to(device)
    train_model(model_extended_lstm, extended_lstm, f"extended_lstm_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/extended_lstm_{label}')
    model_extended_lstm.load_state_dict(state['model'])
    test_results_extended = test_model(model_extended_lstm, test_dataset, label_codes)
    label_results[f'Extended LSTM'] = test_results_extended

    model_extended_gpt = PredictionModel(config).to(device)
    train_model(model_extended_gpt, extended_gpt, f"extended_gpt_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/extended_gpt_{label}')
    model_extended_gpt.load_state_dict(state['model'])
    test_results_extended = test_model(model_extended_gpt, test_dataset, label_codes)
    label_results[f'Extended GPT'] = test_results_extended

    model_extended_retain = PredictionModel(config).to(device)
    train_model(model_extended_retain, extended_retain, f"extended_retain_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/extended_retain_{label}')
    model_extended_retain.load_state_dict(state['model'])
    test_results_extended = test_model(model_extended_retain, test_dataset, label_codes)
    label_results[f'Extended RETAIN'] = test_results_extended

    model_extended_conan = PredictionModel(config).to(device)
    train_model(model_extended_conan, extended_conan, f"extended_conan_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/extended_conan_{label}')
    model_extended_conan.load_state_dict(state['model'])
    test_results_extended = test_model(model_extended_conan, test_dataset, label_codes)
    label_results[f'Extended CONAN'] = test_results_extended

    model_extended_dipole = PredictionModel(config).to(device)
    train_model(model_extended_dipole, extended_dipole, f"extended_dipole_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/extended_dipole_{label}')
    model_extended_dipole.load_state_dict(state['model'])
    test_results_extended = test_model(model_extended_dipole, test_dataset, label_codes)
    label_results[f'Extended Dipole'] = test_results_extended

    model_extended_synteg = PredictionModel(config).to(device)
    train_model(model_extended_synteg, extended_synteg, f"extended_synteg_{label}", label_codes)
    state = torch.load(f'./save/augmentation/prediction/extended_synteg_{label}')
    model_extended_synteg.load_state_dict(state['model'])
    test_results_extended = test_model(model_extended_synteg, test_dataset, label_codes)
    label_results[f'Extended SynTEG'] = test_results_extended

    results[label] = label_results
    pickle.dump(label_results, open(f'./temp_results/prediction_stats_{label_idx}.pkl', 'wb'))

pickle.dump(results, open(f"results/augmentation_stats/prediction_stats.pkl", "wb"))
results['Average'] = {k: {m: np.mean([results[i][k][m] for i in results]) for m in list(list(results.values())[0].values())[0].keys() if m != 'Confusion Matrix'} for k in list(results.values())[0].keys()}
print(results['Average'])
print('\n\n\n')
print(results['COPD'])
print('\n\n\n')
print(results['Heart Failure'])
pickle.dump(results, open(f"results/augmentation_stats/prediction_stats.pkl", "wb"))