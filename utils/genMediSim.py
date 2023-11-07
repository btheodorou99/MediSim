import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

mimic_dir = "../../Datasets/MIMIC-III/"
admissionFile = mimic_dir + "ADMISSIONS.csv"
diagnosisFile = mimic_dir + "DIAGNOSES_ICD.csv"
procedureFile = mimic_dir + "PROCEDURES_ICD.csv"
medicationFile = mimic_dir + "PRESCRIPTIONS.csv"

print("Loading CSVs Into Dataframes")
admissionDf = pd.read_csv(admissionFile, dtype=str)
admissionDf['ADMITTIME'] = pd.to_datetime(admissionDf['ADMITTIME'])
admissionDf = admissionDf.sort_values('ADMITTIME')
admissionDf = admissionDf.reset_index(drop=True)
diagnosisDf = pd.read_csv(diagnosisFile, dtype=str).set_index("HADM_ID")
diagnosisDf = diagnosisDf[diagnosisDf['ICD9_CODE'].notnull()]
diagnosisDf = diagnosisDf[['ICD9_CODE']]
procedureDf = pd.read_csv(procedureFile, dtype=str).set_index("HADM_ID")
procedureDf = procedureDf[procedureDf['ICD9_CODE'].notnull()]
procedureDf = procedureDf[['ICD9_CODE']]
medicationDf = pd.read_csv(medicationFile, dtype=str).set_index("HADM_ID")
medicationDf = medicationDf[medicationDf['NDC'].notnull()]
medicationDf = medicationDf[medicationDf['NDC'] != '0']
medicationDf = medicationDf[['NDC', 'DRUG']]

print("Building Dataset")
data = {}
for row in tqdm(admissionDf.itertuples(), total=admissionDf.shape[0]):          
    #Extracting Admissions Table Info
    hadm_id = row.HADM_ID
    subject_id = row.SUBJECT_ID
            
    # Extracting the Diagnoses
    if hadm_id in diagnosisDf.index: 
        diagnoses = list(set(list(diagnosisDf.loc[[hadm_id]]["ICD9_CODE"])))
        diagnosisDf.loc[[hadm_id]]
    else:
        diagnoses = []
    
    # Extracting the Procedures
    if hadm_id in procedureDf.index: 
        procedures = list(set(list(procedureDf.loc[[hadm_id]]["ICD9_CODE"])))
    else:
        procedures = []
        
    # Extracting the Medications
    if hadm_id in medicationDf.index: 
        medications = list(set(list(medicationDf.loc[[hadm_id]]["NDC"])))
    else:
        medications = []
        
    # Building the hospital admission data point
    if subject_id not in data:
      data[subject_id] = {'visits': [(diagnoses, procedures, medications)]}
    else:
      data[subject_id]['visits'].append((diagnoses, procedures, medications))

print("Shortening Records")
MAX_LEN = 48
for p in data:
    data[p]['visits'] = data[p]['visits'][:MAX_LEN]

code_to_index = {}
all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v[0]]))
np.random.shuffle(all_codes)
for k in all_codes:
    code_to_index[('DIAGNOSIS ICD9_CODE', k)] = len(code_to_index)
print(f"POST-DIAGNOSIS VOCAB SIZE: {len(code_to_index)}")
all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v[1]]))
np.random.shuffle(all_codes)
for k in all_codes:
    code_to_index[('PROCEDURE ICD9_CODE', k)] = len(code_to_index)
print(f"POST-PROCEDURE VOCAB SIZE: {len(code_to_index)}")
all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v[2]]))
np.random.shuffle(all_codes)
for k in all_codes:
    code_to_index[('NDC', k)] = len(code_to_index)
print(f"POST-MEDICATION VOCAB SIZE: {len(code_to_index)}")

index_to_code = {v: k for k, v in code_to_index.items()}

print("Converting Visits")
for p in data:
    new_visits = []
    for v in data[p]['visits']:
        new_visit = []
        for c in v[0]:
            new_visit.append(code_to_index[('DIAGNOSIS ICD9_CODE', c)])
        for c in v[1]:
            new_visit.append(code_to_index[('PROCEDURE ICD9_CODE', c)])
        for c in v[2]:
            new_visit.append(code_to_index[('NDC', c)])
                
        new_visits.append((list(set(new_visit))))
        
    data[p]['visits'] = new_visits    

data = list(data.values())
print(f"MAX LEN: {max([len(p['visits']) for p in data])}")
print(f"AVG LEN: {np.mean([len(p['visits']) for p in data])}")
print(f"MAX VISIT LEN: {max([len(v) for p in data for v in p['visits']])}")
print(f"AVG VISIT LEN: {np.mean([len(v) for p in data for v in p['visits']])}")
print(f"NUM RECORDS: {len(data)}")
print(f"NUM LONGITUDINAL RECORDS: {len([p for p in data if len(p['visits']) > 1])}")

# Train-Val-Test Split
print("Splitting Datasets")
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=4, shuffle=True)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4, shuffle=True)

# Save Everything
print("Saving Everything")
print(len(index_to_code))
pickle.dump(code_to_index, open("../data/codeToIndex.pkl", "wb"))
pickle.dump(index_to_code, open("../data/indexToCode.pkl", "wb"))
pickle.dump(train_dataset, open("../data/trainDataset.pkl", "wb"))
pickle.dump(val_dataset, open("../data/valDataset.pkl", "wb"))
pickle.dump(test_dataset, open("../data/testDataset.pkl", "wb"))