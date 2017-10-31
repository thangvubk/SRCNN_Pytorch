import torch
from torch.utils.data import Dataset, DataLoader
from utils import make_input

class SRCNN_dataset(Dataset):
    """
    Create dataset for SRCNN

    Args:
        config: config to get dataset from utils
        transfrom: optional transform to be applied on a sample
    """
    def __init__(self, config):
        self.inputs, self.labels = make_input(config)
    
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        label_sample = self.labels[idx]
        
        # transpose channel because
        # numpy image H x W x C
        # torch image C x H x W
        input_sample = input_sample.transpose(2, 0, 1)
        label_sample = label_sample.transpose(2, 0, 1)

        # Wrap with tensor
        input_sample, label_sample = torch.Tensor(input_sample), torch.Tensor(label_sample)
    
        return input_sample, label_sample



