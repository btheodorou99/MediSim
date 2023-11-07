import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from config import MediSimConfig
from scipy.sparse import csr_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = MediSimConfig()
train_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb')) + pickle.load(open('./data/valDataset.pkl', 'rb'))
train_dataset = [v for p in train_dataset for v in p['visits']]
test_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))[:7000]
test_dataset = [v for p in test_dataset for v in p['visits']]

def get_batch(ehr_dataset, loc, batch_size):
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), config.diagnosis_vocab_size), np.int8)
    batch_labels = np.zeros((len(ehr), config.procedure_vocab_size+config.medication_vocab_size), np.int8)
    for i, v in enumerate(ehr):
        batch_ehr[i][[c for c in v if c < config.diagnosis_vocab_size]] = 1
        batch_labels[i][[c - config.diagnosis_vocab_size for c in v if c >= config.diagnosis_vocab_size]] = 1

    batch_ehr = csr_matrix(batch_ehr)
    return batch_ehr, batch_labels

def train_model(model, train_dataset, save_name):
    for i in tqdm(range(0, len(train_dataset), 100000)):
        X, y = get_batch(train_dataset, i, 100000)
        if i == 0:
            model.partial_fit(X, y, classes=[np.arange(2) for _ in range(config.procedure_vocab_size+config.medication_vocab_size)])
        else:
            model.partial_fit(X, y)
    pickle.dump(model, open(f'./save/modality/{save_name}', 'wb'))

def conf_mat(x, y):
    totaltrue = np.sum(x)
    totalfalse = len(x) - totaltrue
    truepos, totalpos = np.sum(x & y), np.sum(y)
    falsepos = totalpos - truepos
    return np.array([[totalfalse - falsepos, falsepos], #true negatives, false positives
                     [totaltrue - truepos, truepos]]) #false negatives, true positives
    
def test_model(model, test_dataset):
    loss_list = []
    full_acc = 0
    cmatrix = None
    log_probability_overall = 0
    n_pos_codes = 0
    bce = nn.BCELoss()
    print("Beginning Test Predictions")
    for i in tqdm(range(0, len(test_dataset), 100000)):
        batch_ehr, batch_labels = get_batch(test_dataset, i, 100000)
        prob = np.array(model.predict_proba(batch_ehr)).transpose((1,0,2))
        val_loss = bce(torch.tensor(prob)[:,:,1], torch.DoubleTensor(batch_labels))
        loss_list.append(val_loss.cpu().detach().numpy())
        pred_list = prob[:,:,1].flatten()
        true_list = batch_labels.flatten()
        round_list = np.around(pred_list)
        for i in range(len(prob)):
            if (np.around(prob[i,:,1]) == batch_labels[i,:]).all():
                full_acc += 1

        cmatrix = conf_mat(true_list.astype(np.int8), round_list.astype(np.int8)) if cmatrix is None else cmatrix + conf_mat(true_list.astype(np.int8), round_list.astype(np.int8))
        log_probability_overall += np.log(np.absolute(batch_labels - 1.0 + prob[:,:,1])).sum()
        n_pos_codes += batch_labels.sum()
      
    n_visits = len(test_dataset)
    n_total_codes = len(test_dataset) * (config.procedure_vocab_size + config.medication_vocab_size)

    # Extract, save, and display test metrics
    avg_loss = np.mean(loss_list)
    tn, fp, fn, tp = cmatrix.ravel()
    acc = (tn + tp)/(tn+fp+fn+tp)
    prc = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = (2 * prc * rec)/(prc + rec)
    full_acc = full_acc / n_visits
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

model = MultiOutputClassifier(estimator=SGDClassifier(loss='log'))
train_model(model, train_dataset, f"logistic_regression")
model = pickle.load(open(f'./save/modality/logistic_regression', 'rb'))
results = test_model(model, test_dataset)
pickle.dump(results, open(f"results/modality_completion_stats/logistic_regression_stats.pkl", "wb"))