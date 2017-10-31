import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loader import SRCNN_dataset
from model import SRCNN

def train(dataset, model, loss_fn, optimizer, num_epochs, batch_size):
    """
    Train the network

    Args:
        - dataloader: used to load minibatch
        - model: model for compute output
        - loss_fn: loss function
        - optimizer: weight update scheme
        - num_epochs: number of epochs
    """

    # load data
    #dataset = SRCNN_dataset(data_config, transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    # Train the model
    for epoch in range(num_epochs):
        for i, (input_batch, label_batch) in enumerate(dataloader):
            #Wrap with torch Variable
            input_batch, label_batch = Variable(input_batch), Variable(label_batch)
            
            #zero the grad
            optimizer.zero_grad()

            # Forward + Backward + update
            output_batch = model(input_batch)
            loss = loss_fn(output_batch, label_batch)
            if i%10 == 0:
                print('Epoch %d, iter %5d, loss %.5f' \
                        %(epoch + 1, i, loss.data[0]))

            loss.backward()
            optimizer.step()
    return model
           
