import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

mimic_cxr_path = "/srv/local/data/MIMIC-CXR/"

mimic_cxr_meta = pd.read_csv(os.path.join(mimic_cxr_path, "cxr-record-list.csv"))
cxr_data = pd.read_csv(os.path.join(mimic_cxr_path, "mimic-cxr-2.0.0-metadata.csv"))
cxr_data.StudyDate = cxr_data.StudyDate.astype(str)
cxr_data.StudyTime = cxr_data.StudyTime.astype(str).str.split(".").str[0]
cxr_data["StudyDateTime"] = pd.to_datetime(cxr_data.StudyDate + cxr_data.StudyTime,
                                                     format="%Y%m%d%H%M%S",
                                                     errors="coerce")
cxr_data = cxr_data[["subject_id", "study_id", "dicom_id", "StudyDateTime"]]
cxr_data = cxr_data.drop_duplicates(["subject_id", "study_id"])
cxr_data = cxr_data.dropna().reset_index(drop=True)
cxr_findings = pd.read_csv(os.path.join(mimic_cxr_path, "mimic-cxr-2.0.0-negbio.csv"))
cxr_data = cxr_data.merge(cxr_findings, on=["subject_id", "study_id"], how="inner")
cxr_data = cxr_data.sort_values("StudyDateTime").reset_index(drop=True)
cxr_modalities = {
    "Cardiopulmonary Conditions": ["Cardiomegaly", "Edema", "Enlarged Cardiomediastinum", "Pneumonia", "Pneumothorax"],
    "Pulmonary Parenchymal Abnormalities": ["Atelectasis", "Consolidation", "Lung Lesion", "Lung Opacity"],
    "Pleural Conditions": ["Pleural Effusion", "Pleural Other"],
    "Miscellaneous Findings": ["Fracture", "Support Devices", "No Finding"]
}
cxr_data = cxr_data.rename(columns={c: c.replace(' ', '') for c in cxr_data.columns})
cxr_modalities = {k: [c.replace(' ', '') for c in v] for k, v in cxr_modalities.items()}

print("Building Dataset")
data = {}
for row in tqdm(cxr_data.itertuples(), total=cxr_data.shape[0]):          
    #Extracting Admissions Table Info
    subject_id = row.subject_id
    dicom_id = row.dicom_id
    valMap = {col: getattr(row, col) for mod in cxr_modalities for col in cxr_modalities[mod]}

    # Building the hospital admission data point
    if subject_id not in data:
      data[subject_id] = {'visits': [(valMap, dicom_id)]}
    else:
      data[subject_id]['visits'].append((valMap, dicom_id))

print("Shortening Records")
MAX_LEN = 11
for p in data:
    data[p]['visits'] = data[p]['visits'][:MAX_LEN]

code_to_index = {}
for mod in cxr_modalities:
    all_codes = [(mod, c, val) for c in cxr_modalities[mod] for val in [0, 1, -1, 'missing']]
    np.random.shuffle(all_codes)
    for k in all_codes:
        code_to_index[k] = len(code_to_index)

index_to_code = {v: k for k, v in code_to_index.items()}

print("Converting Visits")
for p in data:
    new_visits = []
    for v in data[p]['visits']:
        new_visit = []
        for mod in cxr_modalities:
            for c in cxr_modalities[mod]:
                val = v[0][c]
                if val != val:
                    val = 'missing'
                    
                new_visit.append(code_to_index[(mod, c, val)])
        new_visits.append((new_visit, v[1]))
        
    data[p]['visits'] = new_visits

data = list(data.values())
print(f"MAX LEN: {max([len(p['visits']) for p in data])}")
print(f"AVG LEN: {np.mean([len(p['visits']) for p in data])}")
print(f"MAX VISIT LEN: {max([len(v[0]) for p in data for v in p['visits']])}")
print(f"AVG VISIT LEN: {np.mean([len(v[0]) for p in data for v in p['visits']])}")
print(f"NUM RECORDS: {len(data)}")
print(f"NUM LONGITUDINAL RECORDS: {len([p for p in data if len(p['visits']) > 1])}")

# Train-Val-Test Split
print("Splitting Datasets")
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=4, shuffle=True)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4, shuffle=True)

# Save Everything
print("Saving Everything")
print(len(index_to_code))
pickle.dump(code_to_index, open("../data_image/codeToIndex.pkl", "wb"))
pickle.dump(index_to_code, open("../data_image/indexToCode.pkl", "wb"))
pickle.dump(train_dataset, open("../data_image/trainDataset.pkl", "wb"))
pickle.dump(val_dataset, open("../data_image/valDataset.pkl", "wb"))
pickle.dump(test_dataset, open("../data_image/testDataset.pkl", "wb"))