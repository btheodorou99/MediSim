import pickle
from config import MediSimConfig

config = MediSimConfig()
NUM_TRAINING = 7000

full = pickle.load(open('./data/testDataset.pkl', 'rb'))
dataset = [{'visits': [[c for c in v if c < config.diagnosis_vocab_size] for v in p['visits'][0:1]]} for p in full[:NUM_TRAINING]]
pickle.dump(dataset, open('./results/datasets/real_diagnosis_shortened.pkl', 'wb'))