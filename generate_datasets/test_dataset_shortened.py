import pickle

NUM_TRAINING = 7000
NUM_TESTING = 2000

full = pickle.load(open('./data/testDataset.pkl', 'rb'))
dataset = [{'visits': p['visits'][0:1]} for p in full[NUM_TRAINING:NUM_TRAINING+NUM_TESTING]]
pickle.dump(dataset, open('./results/datasets/test_dataset_shortened.pkl', 'wb'))