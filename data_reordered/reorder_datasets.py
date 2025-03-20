import os
import pickle
from config import MediSimConfig

def reorder_codes(visits, config):
    """
    Reorder codes from [diagnosis, procedure, medication] to [medication, diagnosis, procedure]
    """
    new_visits = []
    for visit in visits:
        new_visit = []
        # Get codes for each modality using the original ordering
        diagnosis_codes = [code for code in visit if code < config.diagnosis_vocab_size]
        procedure_codes = [code for code in visit if config.diagnosis_vocab_size <= code < config.diagnosis_vocab_size + config.procedure_vocab_size]
        medication_codes = [code for code in visit if config.diagnosis_vocab_size + config.procedure_vocab_size <= code < config.code_vocab_size]
        
        # Adjust indices for new ordering [medication, diagnosis, procedure]
        new_medication_codes = [code - (config.diagnosis_vocab_size + config.procedure_vocab_size) for code in medication_codes]
        new_diagnosis_codes = [code + config.medication_vocab_size for code in diagnosis_codes]
        new_procedure_codes = [code - config.diagnosis_vocab_size + (config.medication_vocab_size + config.diagnosis_vocab_size) for code in procedure_codes]
        
        # Combine in new order
        new_visit.extend(new_medication_codes)
        new_visit.extend(new_diagnosis_codes)
        new_visit.extend(new_procedure_codes)
        new_visits.append(new_visit)
    return new_visits

def main():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    config = MediSimConfig()
    
    # Load original datasets
    train_data = pickle.load(open('../data/trainDataset.pkl', 'rb'))
    val_data = pickle.load(open('../data/valDataset.pkl', 'rb'))
    test_data = pickle.load(open('../data/testDataset.pkl', 'rb'))
    
    # Reorder each dataset
    new_train_data = []
    new_val_data = []
    new_test_data = []
    
    for patient in train_data:
        new_patient = {'visits': reorder_codes(patient['visits'], config)}
        new_train_data.append(new_patient)
        
    for patient in val_data:
        new_patient = {'visits': reorder_codes(patient['visits'], config)}
        new_val_data.append(new_patient)
        
    for patient in test_data:
        new_patient = {'visits': reorder_codes(patient['visits'], config)}
        new_test_data.append(new_patient)
    
    # Save reordered datasets
    pickle.dump(new_train_data, open('./data/trainDataset.pkl', 'wb'))
    pickle.dump(new_val_data, open('./data/valDataset.pkl', 'wb'))
    pickle.dump(new_test_data, open('./data/testDataset.pkl', 'wb'))
    
    print("Datasets have been reordered and saved to data/")

if __name__ == "__main__":
    main()
