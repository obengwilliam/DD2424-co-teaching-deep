# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10
# from data.mnist import MNIST
from NetWork import CNN
import argparse, sys
import numpy as np
import datetime
import shutil

from Loss import loss_coteaching

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default=0.8)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=2, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=3)

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
learning_rate = args.lr
batch_size = 100
remember_rate = 1 - args.forget_rate

# load dataset
if args.dataset == 'cifar10':
    input_channel = 3
    num_classes = 10
    args.top_bn = False
    args.epoch_decay_start = 4
    args.n_epoch = 10
    train_dataset = CIFAR10(root="./data/",
                          train=True,
                          download=True,
                          transform=transforms.ToTensor(),
                          noise_rate=args.noise_rate,
                          noise_type=args.noise_type
                          )
    test_dataset = CIFAR10(root="./data/",
                         train=False,
                         download=True,
                         transform=transforms.ToTensor(),
                         noise_rate=args.noise_rate,
                         noise_type=args.noise_type
                         )

'''
if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 10
    args.top_bn = False
    args.epoch_decay_start = 100
    args.n_epoch = 200
    train_dataset = C1FAR100(root="./data/",
                            train=True,
                            download=True,
                            transform=transforms.ToTensor(),
                            noise_rate=args.noise_rate,
                            noise_type=args.noise_type
                            )
    test_dataset = CIFAR100(root="./data/",
                           train=False,
                           download=True,
                           transform=transforms.ToTensor(),
                           noise_rate=args.noise_rate,
                           noise_type=args.noise_type
                           )
'''

'''
if args.dataset == 'minst':
    input_channel = 1
    num_classes = 10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = MINST(root = "./data/", 
                          train = True,
                          download = True,
                          transform = transforms.ToTensor(),
                          noise_rate = args.noise_rate,
                          noise_type = args.noise_type
    )
    test_dataset = MINST(root="./data/",
                          train=False,
                          download=True,
                          transform=transforms.ToTensor(),
                          noise_rate=args.noise_rate,
                          noise_type=args.noise_type
                          )
'''

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2


def adjust_learning_rate(optimizer, epoch):
    # Only change beta1
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999) # Only change beta1
        

# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*(1-remember_rate)
rate_schedule[:args.num_gradual] = np.linspace(0, (1-remember_rate)**args.exponent, args.num_gradual)


def accuracy(logit, target, k_top=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    max_k = max(k_top)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for i in k_top:
        correct_i = correct[:i].view(-1).float().sum(0, keepdim=True)
        res.append(correct_i.mul_(100.0 / batch_size))
    return res


# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2):
    print('Training ...')

    train_loss_1, train_acc_1 = 0, 0
    train_loss_2, train_acc_2 = 0, 0
    train_size_1 = 0
    train_size_2 = 0

    for batch, (images, labels, indexes) in enumerate(train_loader):
        if batch > args.num_iter_per_epoch:
            break
        X = Variable(images).cuda()
        y = Variable(labels).cuda()

        # Compute prediction error
        pred_1 = model1(X)
        # pred_1 = pred_1.type(torch.FloatTensor)
        pred_2 = model2(X)
        # pred_2 = pred_2.type(torch.FloatTensor)
        # res1 = accuracy(pred_1, y, k_top=(1, 5))
        # res2 = accuracy(pred_2, y, k_top=(1, 5))
        train_size_1 += 1
        train_size_2 += 1
        # train_acc_1 += res1
        # train_acc_2 += res2
        train_loss_1, train_loss_2 = loss_coteaching(pred_1, pred_2, y, rate_schedule[epoch])
        # Forward + Backward + Optimize
        optimizer1.zero_grad()
        train_loss_1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        train_loss_2.backward()
        optimizer2.step()

        '''
        if (batch + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss1: %.4f, Loss2: %.4f'
                  % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, train_loss_1.data[0], train_loss_2.data[0]))
        '''
    train_acc_1 = float(train_acc_1) / float(train_size_1)
    train_acc_2 = float(train_acc_2) / float(train_size_2)

    return train_acc_1, train_acc_2


# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating ...')

    test_loss_1, acc1 = 0, 0
    test_loss_2, acc2 = 0, 0
    test_size = 0

    model1.eval()
    # Change model to 'eval' mode.

    print('Enter the evaluate')
    with torch.no_grad():
        for images, labels, _ in test_loader:
            test_size += labels.size(0)
            X = Variable(images).cuda()

            pred_1 = model1(X)
            output_1 = F.softmax(pred_1, dim = 1)
            _, y_1 = torch.max(output_1.data, 1)
            acc1 += (y_1.cpu() == labels).sum()

    model2.eval()
    # Change model to 'eval' mode.
    with torch.no_grad():
        for images, labels, _ in test_loader:
            X = Variable(images).cuda()

            pred_2 = model2(X)
            output_2 = F.softmax(pred_2, dim = 1)
            _, y_2 = torch.max(output_2.data, 1)
            acc2 += (y_2.cpu() == labels).sum()

    
    acc1 = 100 * float(acc1) / float(test_size)
    acc2 = 100 * float(acc2) / float(test_size)
    return acc1, acc2


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    # part_train, desert_train = torch.utils.data.random_split(train_dataset, [int(1000), len(train_dataset) - int(1000)])
    # part_test, desert_test = torch.utils.data.random_split(test_dataset, [int(200), len(test_dataset) - int(200)])

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          num_workers = args.num_workers,
                                          drop_last = True,
                                          shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          num_workers = args.num_workers,
                                          drop_last = True,
                                          shuffle = False)

    
    # Define models
    print('building model')
    model1 = CNN(input_channel = input_channel, n_outputs = num_classes)
    model1.cuda()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
    model2.cuda()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

    epoch = 0
    train_acc_1 = 0
    train_acc_2 = 0
    test_acc_1, test_acc_2 = evaluate(test_loader, model1, model2)

    '''
    with open(txtfile, "a") as myfile:
        myfile.write(
            'epoch: train_acc1 train_acc2 test_acc1 test_acc2\n')
    
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc_1) +' '  + str(train_acc_2) +' '  + str(test_acc_1) + " " + str(test_acc_2) + "\n")
    '''

    # evaluate models with random weights
    acc1, acc2 = evaluate(test_loader, model1, model2)
    # save results

    print('enter for loop')

    # training
    for epoch in range(1, args.n_epoch):

        model1.train()
        adjust_learning_rate(optimizer1, epoch)

        model2.train()
        adjust_learning_rate(optimizer2, epoch)
        # train models
        train(train_loader, epoch, model1, optimizer1, model2, optimizer2)
        # evaluate models
        test_acc_1, test_acc_2 = evaluate(test_loader, model1, model2)

        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%' %
              (epoch+1, args.n_epoch, len(test_dataset), test_acc_1, test_acc_2))
        '''
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': ' + str(train_acc_1) + ' ' + str(train_acc_2) + ' ' + str(test_acc_1) + " " + str(test_acc_2) + "\n")
        '''


if __name__ == '__main__':
    main()
