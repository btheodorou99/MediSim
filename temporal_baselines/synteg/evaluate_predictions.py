import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from config import MediSimConfig
from temporal_baselines.synteg.synteg import Generator, DependencyModel

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
config.batch_size = 5
MAX_VISIT_LENGTH = 280
Z_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
   torch.cuda.manual_seed_all(SEED)

model = DependencyModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
generator = Generator(config).to(device)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=4e-6, weight_decay=1e-5)
checkpoint1 = torch.load("./save/temporal/synteg_dependency_model", map_location=torch.device(device))
checkpoint2 = torch.load("./save/temporal/synteg_condition_model", map_location=torch.device(device))
model.load_state_dict(checkpoint1['model'])
optimizer.load_state_dict(checkpoint1['optimizer'])
generator.load_state_dict(checkpoint2['generator'])
generator_optimizer.load_state_dict(checkpoint2['generator_optimizer'])
test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))

def get_batch(loc, batch_size, dataset):
    ehr = dataset[loc:loc+batch_size]
    
    batch_ehr = np.ones((len(ehr), config.n_ctx, MAX_VISIT_LENGTH)) * (config.code_vocab_size + 2) # Initialize each code to the padding code
    batch_lens = np.ones((len(ehr), config.n_ctx, 1))
    batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
    batch_num_visits = np.zeros(len(ehr))
    for i, p in enumerate(ehr):
        visits = p['visits']
        for j, v in enumerate(visits):
            batch_mask[i,j+1] = 1
            batch_lens[i,j+1] = len(v) + 1
            for k, c in enumerate(v):
                batch_ehr[i,j+1,k+1] = c
        batch_ehr[i,j+1,len(v)+1] = config.code_vocab_size + 1 # Set the last code in the last visit to be the end record code
        batch_lens[i,j+1] = len(v) + 2
        batch_num_visits[i] = len(visits)
    
    batch_mask[:,1] = 1  # Set the mask to cover the labels
    batch_ehr[:,:,0] = config.code_vocab_size # Set the first code in each visit to be the start/class token
    batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
    return batch_ehr, batch_lens, batch_mask, batch_num_visits


###############
### TESTING ###
###############

def conf_mat(x, y):
    totaltrue = np.sum(x)
    totalfalse = len(x) - totaltrue
    truepos, totalpos = np.sum(x & y), np.sum(y)
    falsepos = totalpos - truepos
    return np.array([[totalfalse - falsepos, falsepos], #true negatives, false positives
                    [totaltrue - truepos, truepos]]) #false negatives, true positives

def generate_predictions(model, generator, context, context_lengths):
    with torch.no_grad():
        condition = model(context, context_lengths, export=True)[:,:-1,:]
        z = torch.randn((condition.size(0), condition.size(1), Z_DIM)).to(device)
        return generator(z.reshape(-1,z.size(-1)), condition.reshape(-1,condition.size(-1))).reshape(condition.size(0), condition.size(1), -1)

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
        batch_ehr, batch_lens, _, batch_num_visits = get_batch(v_i, config.batch_size, test_ehr_dataset)
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.long).to(device)
        batch_lens = torch.tensor(batch_lens, dtype=torch.int).to(device)
        
        labels = torch.sum(F.one_hot(batch_ehr, num_classes=config.total_vocab_size), dim=2)[:,1:,:-1]
        batch_mask = torch.sum(labels, dim=2).bool().float().unsqueeze(-1)
        
        # Get batch outputs
        predictions = generate_predictions(model, generator, batch_ehr, batch_lens)[:,:,:-1]
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
            batch_cmatrix[i][0][0] = torch.sum(batch_mask[:,i]) * (config.total_vocab_size - 1) - batch_cmatrix[i][0][1] - batch_cmatrix[i][1][0] - batch_cmatrix[i][1][1] # Remove the masked values
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
pickle.dump(intermediate, open("./results/temporal_completion_stats/SynTEG_intermediate_results.pkl", "wb"))

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
pickle.dump(metrics_dict, open("./results/temporal_completion_stats/SynTEG_Metrics.pkl", "wb"))

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