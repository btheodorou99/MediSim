import pickle

NUM_TRAINING = 7000

full = pickle.load(open('./data/testDataset.pkl', 'rb'))
dataset = full[:NUM_TRAINING]
pickle.dump(dataset, open('./results/datasets/real_full.pkl', 'wb'))