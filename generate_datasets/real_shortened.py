import pickle

NUM_TRAINING = 7000

full = pickle.load(open('./data/testDataset.pkl', 'rb'))
dataset = [{'visits': p['visits'][0:1]} for p in full[:NUM_TRAINING]]
pickle.dump(dataset, open('./results/datasets/real_shortened.pkl', 'wb'))