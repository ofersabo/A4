import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
from utils import *

## load mnist dataset
use_cuda = torch.cuda.is_available()


# if not exist, download mnist dataset

batch_size = 100
input_size = 30522
hidden_size = 100
output_size = 2
num_epochs = 600 * 4
batch_size = 100
learning_rate = 0.001
drop_out = 0.3



## network
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(2 * input_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1, 2 * input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "MLP"


def train_MLP(train,test):
    ## training
    # my_x = [np.array([[1.0, 2], [3, 4]]), np.array([[5., 6], [7, 8]])]  # a list of numpy arrays
    # my_y = [np.array([4.]), np.array([2.])]  # another list of numpy arrays (targets)
    train_vectors = [np.concatenate(vec[0],vec[1]) for label,vec in train ]
    train_labels = [np.array([label]) for label,vec in train ]
    tensor_x = torch.stack([torch.Tensor(i) for i in train])  # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in train_labels])

    my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
    my_dataloader = utils.DataLoader(my_dataset)  # create your dataloader
    traind_torch =
    model = MLPNet()

    if use_cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # trainning
        ave_loss = 0
        for batch_idx, (x, target) in enumerate(train):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print
                '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx + 1, ave_loss)
        # testing
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(test_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            # smooth average
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
                print
                '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt)

    torch.save(model.state_dict(), model.name())


def main():
    output_file_name = "DL_OUTPUT.txt"
    first_load = False
    where_to_store_date = "data_for_mlp.pickle"

    train, dev = load_from_file(where_to_store_date)
    # order =load_from_file("order")

    train_MLP(train, dev)


if __name__ == '__main__':
    main()
