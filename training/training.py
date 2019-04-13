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
import get_sequence
import get_rand_sequence

def train(trainloader, net, criterion, optimizer, device, num_unrolls = 2):
    running_loss = 0.0
    cnt = 0
    net = net.train()
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output, _ = net(images, num_unrolls)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()        
        cnt += 1
    return (running_loss/cnt)

def test(testloader, net, criterion, device, num_unrolls = 1):
    running_loss = 0.0
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            output, _ = net(images, num_unrolls)
            loss = criterion(output, labels)
            running_loss += loss.item()
            cnt += 1
    return (running_loss/cnt)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DESTINATION = 'checkpoint.pth'

    num_unrolls = args.num_unrolls
    max_steps = args.max_steps
    learning_rate = args.learning_rate
    num_sequence = args.num_sequence
    PATH = args.model_path

#     start_line = np.loadtxt('start_line.txt', dtype=int)

    # Re3Net Set up
    net = Re3Net().to(device)
    if PATH is not None:
        net.load_state_dict(torch.load(PATH))
        net.eval()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)
#     dataset_train = get_sequence.Dataset(32, start_line = start_line[0])
#     dataset_val = get_sequence.Dataset(32, 'val', start_line = start_line[1])
    dataset_train = get_rand_sequence.Dataset(32)
    dataset_val = get_rand_sequence.Dataset(32, 'val')
    print('Start Training: ')
    for epoch in range(1, max_steps+1):
        # TODO: Data Loader
        # X = np.load('training/Images.npy')
        # y = np.load('training/Labels.npy')
        # X = torch.tensor(X, dtype = torch.float)
        # y = torch.tensor(y, dtype = torch.float)

        # train_loader = [(X[i,...], y[i,...]) for i in range(3)]
        # val_loader = [(X[i,...], y[i,...]) for i in range(10,11)]
        train_loader = []
        val_loader = []
        for _ in range(num_sequence):
            X, y = dataset_train.get_data_sequence()
            X = torch.tensor(X, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
            train_loader.append((X, y))
        for _ in range(num_sequence//10):
            X, y = dataset_val.get_data_sequence()
            X = torch.tensor(X, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
            val_loader.append((X, y))

        train_loss = train(train_loader, net, criterion, optimizer, device, num_unrolls = num_unrolls)
        val_loss = test(val_loader, net, criterion, device, num_unrolls = num_unrolls)

        print('[Epoch %d / %d] train_loss: %.5f val_loss: %.5f video id: %d' % (epoch, max_steps, train_loss, val_loss, dataset_train.video_idx))
        torch.save(net.state_dict(), DESTINATION)
#         start_line[0] = dataset_train.cur_line
#         start_line[1] = dataset_val.cur_line
#         np.savetxt('start_line.txt', start_line, fmt = '%d')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training for Re3.')
    parser.add_argument('-n', '--num_unrolls', action='store', type=int, default=2)
    parser.add_argument('-m', '--max_steps', action='store', type=int, default=100)
    parser.add_argument('-s', '--num_sequence', action='store', type=int, default=20)
    parser.add_argument('-l', '--learning_rate', action='store', type=float, default=1e-5)
    parser.add_argument('-p', '--model_path', action='store', type=str, default='checkpoint.pth')
    args = parser.parse_args()
    main(args)


