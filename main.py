import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import SRCNN_dataset
from model import SRCNN
from solver import train

train_config = {
    'dir_path': 'Train',
    'scale': 3,
    'is_gray': True,
    'input_size': 33,
    'label_size': 21,
    'stride': 21
}

test_config = train_config.copy()
test_config['dir_path'] = 'Test/Set5'

train_dataset = SRCNN_dataset(train_config)
model = SRCNN()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#train(train_dataset, model, loss_fn, 
#      optimizer, num_epochs=4, batch_size=128)

test_dataset = SRCNN_dataset(test_config)
test_loader = DataLoader(test_dataset, 125, True, 4)

for _, (input_sample, label_sample) in enumerate(test_loader):
        print(input_sample.size())
        break

