import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from config_image import MediSimConfig
from model_image import DiffusionModel

SEED = 4
cudaNum = 6
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device(f"cuda:{cudaNum}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_dataset = pickle.load(open('./data_image/trainDataset.pkl', 'rb'))
train_dataset = [v for p in train_dataset for v in p['visits']]
train_dataset = [(v[0], v[1]) for v in train_dataset]
val_dataset = pickle.load(open('./data_image/valDataset.pkl', 'rb'))
val_dataset = [v for p in val_dataset for v in p['visits']]
val_dataset = [(v[0], v[1]) for v in val_dataset]
test_dataset = pickle.load(open('./data_image/testDataset.pkl', 'rb'))
test_dataset = [v for p in test_dataset for v in p['visits']]
test_dataset = [(v[0], v[1]) for v in test_dataset]
image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2*x - 1)  # Normalize to [-1, 1]
    ])

def load_image(image_path):
    with Image.open(f'{config.image_dir}/{image_path}.jpg') as img:
        return image_transform(img)

def get_batch(dataset, loc, batch_size):
    images = dataset[loc:loc+batch_size]
    bs = len(images)
    batch_context = torch.zeros(bs, config.code_vocab_size, dtype=torch.float, device=device)
    batch_image = torch.zeros(bs, config.n_channels, config.image_dim, config.image_dim, dtype=torch.float, device=device)
    for i, n in enumerate(images):
        codes, image = n
        batch_context[i, codes] = 1
        batch_image[i] = load_image(image)
        
    return batch_context, batch_image

def tensor_to_image(tensor):
    # First de-normalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    # Convert to PIL image
    img = transforms.ToPILImage()(tensor)
    return img
        
model = DiffusionModel(config).to(device)
print("Loading previous model")
checkpoint = torch.load(f'./save/model_image_gen', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model.eval()

with torch.no_grad():
    sample_dataset = [] 
    for i in tqdm(range(0, len(train_dataset), config.batch_size_gen), leave=False):
        sample_contexts, _ = get_batch(train_dataset, i, config.batch_size_gen)
        sample_images = model.generate(sample_contexts)
        for j in range(sample_images.size(0)):
            sample_dataset.append(sample_images[j].cpu().numpy())
    pickle.dump(sample_dataset, open(f'data_image/sampleImagesTrain.pkl', 'wb'))
    
    sample_dataset = [] 
    for i in tqdm(range(0, len(val_dataset), config.batch_size_gen), leave=False):
        sample_contexts, _ = get_batch(val_dataset, i, config.batch_size_gen)
        sample_images = model.generate(sample_contexts)
        for j in range(sample_images.size(0)):
            sample_dataset.append(sample_images[j].cpu().numpy())
    pickle.dump(sample_dataset, open(f'data_image/sampleImagesVal.pkl', 'wb'))
    
    sample_dataset = [] 
    for i in tqdm(range(0, len(test_dataset), config.batch_size_gen), leave=False):
        sample_contexts, _ = get_batch(test_dataset, i, config.batch_size_gen)
        sample_images = model.generate(sample_contexts)
        for j in range(sample_images.size(0)):
            sample_dataset.append(sample_images[j].cpu().numpy())
    pickle.dump(sample_dataset, open(f'data_image/sampleImagesTest.pkl', 'wb'))