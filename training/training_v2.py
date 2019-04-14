import os
import argparse
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
from tracker.network import Re3Net
import get_rand_sequence

def train(trainloader, net, criterion, optimizer, device, num_unrolls = 2):
    net = net.train()
    images, labels = trainloader
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    output, _ = net(images, num_unrolls)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(testloader, net, criterion, device, num_unrolls = 2):
    running_loss = 0.0
    with torch.no_grad():
        net = net.eval()
        images, labels = testloader
        images = images.to(device)
        labels = labels.to(device)
        output, _ = net(images, num_unrolls)
        loss = criterion(output, labels)
        running_loss = loss.item()
    return running_loss

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DESTINATION = 'checkpoint.pth'

    num_unrolls = args.num_unrolls
    max_steps = args.max_steps
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    use_net_prob = args.use_net_prob
    PATH = args.model_path

    # Re3Net Set up
    net = Re3Net().to(device)
    if PATH is not None:
        net.load_state_dict(torch.load(PATH))
        net.eval()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

    dataset_train = get_rand_sequence.Dataset(net, num_unrolls, USE_NETWORK_PROB = use_net_prob)
    dataset_val = get_rand_sequence.Dataset(net, num_unrolls, mode = 'val')

    print('Start Training: ')
    for epoch in range(1, max_steps+1):
        train_X = []
        train_y = []
        for _ in range(batch_size):
            X, y = dataset_train.get_data_sequence()
            train_X.append(X)
            train_y.append(y)   
        X = np.concatenate(train_X)
        y = np.concatenate(train_y)
        X = torch.tensor(X, dtype = torch.float)
        y = torch.tensor(y, dtype = torch.float)

        train_loss = train((X, y), net, criterion, optimizer, device, num_unrolls = num_unrolls)

        val_X = []
        val_y = []
        for _ in range(max(batch_size//10, 1)):
            X, y = dataset_val.get_data_sequence()            
            val_X.append(X)
            val_y.append(y)
        X = np.concatenate(val_X)
        y = np.concatenate(val_y)
        X = torch.tensor(X, dtype = torch.float)
        y = torch.tensor(y, dtype = torch.float)

        val_loss = test((X, y), net, criterion, device, num_unrolls = num_unrolls)

        print('[Epoch %d / %d] train_loss: %.5f val_loss: %.5f video id: %d' % (epoch, max_steps, train_loss, val_loss, dataset_train.video_idx))
        
        if epoch % 50 == 0:
            torch.save(net.state_dict(), DESTINATION)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training for Re3.')
    parser.add_argument('-n', '--num_unrolls', action='store', type=int, default=2)
    parser.add_argument('-m', '--max_steps', action='store', type=int, default=100)
    parser.add_argument('-b', '--batch_size', action='store', type=int, default=64)
    parser.add_argument('-l', '--learning_rate', action='store', type=float, default=1e-5)
    parser.add_argument('-u', '--use_net_prob', action='store', type=float, default=0)
    parser.add_argument('-p', '--model_path', action='store', type=str, default='checkpoint.pth')
    args = parser.parse_args()
    main(args)
