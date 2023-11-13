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
cudaNum = 4
NUM_SAMPLES = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MediSimConfig()
device = torch.device(f"cuda:{cudaNum}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

test_dataset = pickle.load(open('./data_image/testDataset.pkl', 'rb'))
test_dataset = [v for p in test_dataset for v in p['visits']]
test_dataset = [(v[0], v[1]) for v in test_dataset]
indexToCode = pickle.load(open('./data_image/indexToCode.pkl', 'rb'))
image_transform = transforms.Compose([
        transforms.Resize((config.image_dim_gen, config.image_dim_gen)),
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
    batch_image = torch.ones(bs, config.n_channels, config.image_dim_gen, config.image_dim_gen, dtype=torch.float, device=device)
    batch_codes = []
    for i, n in enumerate(images):
        codes, image = n
        batch_context[i, codes] = 1
        batch_image[i] = load_image(image)
        batch_codes.append(codes)
        
    return batch_context, batch_image, batch_codes
  
def tensor_to_image(tensor):
    # First de-normalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    # Convert to PIL image
    img = transforms.ToPILImage()(tensor)
    return img
  
model = DiffusionModel(config).to(device)
print("Loading previous model")
model.load_state_dict(torch.load(f'./save/model_image_gen', map_location=torch.device(device))['model'])

# Generate Samples
sampleIdx = np.random.choice(len(test_dataset), size=NUM_SAMPLES)
sample_data = [test_dataset[i] for i in sampleIdx]
sample_contexts, real_images, real_codes = get_batch(sample_data, 0, NUM_SAMPLES)
for i in tqdm(range(NUM_SAMPLES)):
    with open(f'results/image_generation/realFinding_{i}.txt', 'w') as f:
        codes = real_codes[i]
        for code in codes:
            finding = indexToCode[code]
            if finding[2] in [0,1]:
                f.write(f"{indexToCode[code]}\n")
    

for i in tqdm(range(NUM_SAMPLES)):
    img = tensor_to_image(real_images[i].cpu())
    img.save(f'results/image_generation/realImage_{i}.jpg')

with torch.no_grad():
    sample_images = model.generate(sample_contexts)
    
for i in tqdm(range(NUM_SAMPLES)):
    sample_image = tensor_to_image(sample_images[i].cpu())
    sample_image.save(f'results/image_generation/sampleImage_{i}.jpg')
