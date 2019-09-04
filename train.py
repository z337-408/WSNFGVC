# -*- coding: utf-8 -*

import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from lr_scheduler import lr_scheduler
from bcnn import BCNN
from PIL import ImageFile
from Imagefolder_modified import Imagefolder_modified
import pickle
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)

os.popen('mkdir -p model')
# os.popen('mkdir -p crossentropy_eachepoch')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--drop_rate', type=float, default=0.25)
parser.add_argument('--T_k', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. ')
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--n_classes', type=int, default=200)
parser.add_argument('--step', type=int, default=None)
# parser.add_argument('--start', type=int, default=2)
parser.add_argument('--resume', action='store_true')


args = parser.parse_args()

data_dir = args.dataset
learning_rate = args.base_lr
batch_size = args.batch_size
num_epochs = args.epoch
weight_decay = args.weight_decay
step = args.step
drop_rate= args.drop_rate
T_k=args.T_k
num_classes=args.n_classes
# start=args.start

resume = args.resume

# data_dir = 'data/cub200web'
# learning_rate = 1e-3
# batch_size = 128
# num_epochs = 200
# weight_decay = 1e-8
# step =1
# drop_rate=0.25
# T_k=10

start =2
warmup_epochs = 5

logfile ='training_log.txt'
# cross_entropy_savapath='crossentropy_eachepoch/'
# Load data
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=448),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(size=448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=448),
    torchvision.transforms.CenterCrop(size=448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_data = Imagefolder_modified(os.path.join(data_dir, 'train'), transform=train_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# Adjust learning rate and betas for Adam Optimizer
mom = 0.9
alpha_plan = lr_scheduler(learning_rate, num_epochs, warmup_end_epoch=warmup_epochs, mode='cosine')
rate_schedule = np.ones(num_epochs) * drop_rate
if T_k>1:
    rate_schedule[:T_k] = np.linspace(0, drop_rate, T_k)

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (mom, 0.999)  # only change beta1

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def accuracy(logit, target, topk=(1,)):
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    N = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # (N, maxk)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target is in shape (N,)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)  # size is 1
        res.append(correct_k.mul_(100.0 / N))
    return res

#compute cross-entropy
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P))

def crossentropy_loss(logits_now, Crossentropy,ids, labels, epoch = start):
    """
    :param logits:      shape of (N, 200)
    :param labels:      shape of (N,)
    :param drop_rate:   drop_rate for each epoch
    :return:loss:       loss to update the network
    """
    if Crossentropy==[]:
        return F.cross_entropy(logits_now, labels)

    all_id = [int(x[3]) for x in Crossentropy]
    num_remember = int((1 - rate_schedule[max(0 , epoch-start)]) * len(all_id))

    false_id = all_id[num_remember:]
    loss_update= [i for i in range(len(ids))]
    #find the irrelevant noisy images in this mini-batch by id
    noise = list(set(false_id).intersection(set(ids.numpy())))
    #remove the irrelevant noisy images
    for i in range(len(ids)):
        if ids[i] in noise:
            loss_update.remove(i)

    logits_final = logits_now[loss_update]
    labels_final = labels[loss_update]

    loss = F.cross_entropy(logits_final, labels_final)
    return loss,len(logits_final)

# Train the Model
def train(train_loader, epoch, model, optimizer, logits_softmax_before=[], Cross_entropy_before=[]):
    train_total = 0
    train_correct = 0
    Cross_entropy_now=[]
    initial = False
    # initialization
    if len(logits_softmax_before) == 0:
        initial=True
        logits_softmax_before = [[0 for i in range(200)] for j in range(len(train_data))]
    # training in each mini-batch
    for it, (images, labels, ids, path) in enumerate(train_loader):
        iter_start_time = time.time()
        images = images.cuda()
        labels = labels.cuda()
        # Forward + Backward + Optimize
        logits = model(images)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec.item()
        # The first epoch, only record softmax probability
        if initial == True:
            for i in range(ids.shape[0]):
                logits_softmax = F.softmax(logits, dim=1).cpu().detach().numpy()
                logits_softmax_before[ids[i]] = logits_softmax[i]
        else:
            for i in range(ids.shape[0]):
                # compute cross-entropy
                logits_softmax = F.softmax(logits, dim=1).cpu().detach().numpy()
                output = cross_entropy(logits_softmax_before[ids[i]], logits_softmax[i])
                # record softmax probability
                logits_softmax_before[ids[i]] = logits_softmax[i]
                tmp = []
                tmp.append(output)
                tmp.append(path[i])
                tmp.append(labels[i].cpu().numpy())
                tmp.append(ids[i].cpu().numpy())
                # cross_entropy, image path, image label, image id. These can be output to oberseve the selection result.
                # 'Cross_entropy_now' is computed by softmax probability of epoch T and epoch T-1. It will be used in next epoch T+1
                Cross_entropy_now.append(tmp)

        if len(Cross_entropy_before) >0:
            # update network guided by 'Cross_entropy_before'. It's computed by softmax probability of epoch T-1 and epoch T-2
            loss_entropy , numbers = crossentropy_loss(logits,Cross_entropy_before,ids,labels,epoch)
        else:
            loss_entropy = F.cross_entropy(logits, labels)
            numbers = batch_size

        optimizer.zero_grad()
        loss_entropy.backward()
        optimizer.step()

        iter_end_time = time.time()
        print('Epoch:[{0:03d}/{1:03d}]  Iter:[{2:04d}/{3:04d}]  '
              'Train Accuracy :[{4:6.2f}]  Loss :[{5:4.4f}] '
              'Iter Runtime:[{6:6.2f}]'
              'training number:[{7:03d}]'.format(
            epoch + 1, num_epochs, it + 1, len(train_data)// batch_size,
            prec.item(), loss_entropy.item(),
            iter_end_time - iter_start_time,numbers))

    train_acc = float(train_correct) / float(train_total)
    # sort samples by cross-entropy
    Cross_entropy_now.sort(key=lambda x: x[0])
    return train_acc, logits_softmax_before, Cross_entropy_now

def evaluate(test_loader, model):
    model.eval()  # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum().item()

    acc = 100 * float(correct) / float(total)
    return acc

def main():
    # step = args.step
    print('===> About training in a two-step process! ===')
    print('------\n'
          'drop rate: [{}]\tT_k: [{}]\t'
          'start epoch: [{}]\t'
          '\n------'.format(
        drop_rate,T_k,start))
    # step 1: only train the fc layer
    if step == 1:
        print('===> Step 1 ...')
        bnn = BCNN(pretrained=True, n_classes=num_classes)
        bnn = nn.DataParallel(bnn).cuda()
        optimizer = optim.Adam(bnn.module.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # step 1: train the whole network
    elif step == 2:
        print('===> Step 2 ...')
        bnn = BCNN(pretrained=False, n_classes=num_classes)
        bnn = nn.DataParallel(bnn).cuda()
        optimizer = optim.Adam(bnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise AssertionError('Wrong step argument')

    loadmodel = 'checkpoint.pth'
    # check if it is resume mode
    print('-----------------------------------------------------------------------------')
    if resume:
        assert os.path.isfile(loadmodel), 'please make sure checkpoint.pth exists'
        print('---> loading checkpoint.pth <---')
        checkpoint = torch.load(loadmodel)
        assert step == checkpoint['step'], 'step in checkpoint does not match step in argument'
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        best_epoch = checkpoint['best_epoch']
        bnn.load_state_dict(checkpoint['bnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        Cross_entropy = checkpoint['Cross_entropy']
        logits_softmax = checkpoint['logits_softmax']
    else:
        if step == 2:
            print('--->        step2 checkpoint loaded         <---')
            bnn.load_state_dict(torch.load('model/bnn_step1_vgg16_best_epoch.pth'))
        else:
            print('--->        no checkpoint loaded         <---')
        Cross_entropy = []
        logits_softmax = []
        start_epoch = 0
        best_accuracy = 0.0
        best_epoch = None
    print('-----------------------------------------------------------------------------')

    with open(logfile, "a") as f:
        f.write('------ Step: {} ...\n'.format(step))
        f.write('------\n'
              'drop rate: [{}]\tT_k: [{}]\t'
              'start epoch: [{}]\t'
              '\n------'.format(
            drop_rate, T_k, start))

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        bnn.train()
        adjust_learning_rate(optimizer, epoch)

        #train returns 'Cross_entropy', used in saving checkpoints.
        train_acc ,logits_softmax, Cross_entropy = train(train_loader, epoch, bnn, optimizer, logits_softmax, Cross_entropy)

        # dump the output: cross_entropy, image path, image label, image id. If you want to check the selection result, just use the code blow.
        # if len(Cross_entropy) > 0:
        #     pickle.dump(Cross_entropy, open(cross_entropy_savapath + 'crossentropy_epoch{}_step{}.pkl'.format(epoch + 1,step), 'wb'))

        test_acc = evaluate(test_loader, bnn)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1
            torch.save(bnn.state_dict(), 'model/bnn_step{}_vgg16_best_epoch.pth'.format(step))

        epoch_end_time = time.time()
        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'bnn_state_dict': bnn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_epoch': best_epoch,
            'best_accuracy': best_accuracy,
            'step': step,
            'Cross_entropy' : Cross_entropy,
            'logits_softmax' : logits_softmax
        },filename=loadmodel)

        print('------\n'
              'Epoch: [{:03d}/{:03d}]\tTrain Accuracy: [{:6.2f}]\t'
              'Test Accuracy: [{:6.2f}]\t'
              'Epoch Runtime: [{:6.2f}]\t'\
              '\n------'.format(
            epoch + 1, num_epochs, train_acc, test_acc,
            epoch_end_time - epoch_start_time))
        with open(logfile, "a") as f:
            output = 'Epoch: [{:03d}/{:03d}]\tTrain Accuracy: [{:6.2f}]\t' \
                     'Test Accuracy: [{:6.2f}]\t' \
                     'Epoch Runtime: [{:6.2f}]\t'.format(
                epoch + 1, num_epochs, train_acc, test_acc,
                epoch_end_time - epoch_start_time)
            f.write(output + "\n")

    print('******\n'
          'Best Accuracy 1: [{0:6.2f}], at Epoch [{1:03d}] '
          '\n******'.format(best_accuracy, best_epoch))
    with open(logfile, "a") as f:
        output = '******\n' \
                 'Best Accuracy 1: [{0:6.2f}], at Epoch [{1:03d}]; ' \
                 '\n******'.format(best_accuracy, best_epoch)
        f.write(output + "\n")


if __name__ == '__main__':
    main()
